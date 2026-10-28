// Linux LD_PRELOAD observer. Only intercepted library-call envelopes are timed.
// No training-thread synchronization. Graph capture, nonblocking NCCL and mixed
// NCCL groups are deliberately skipped. Not a complete GPU/kernel collector.
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <nccl.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <charconv>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <mutex>
#include <map>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include "observer_output_buffer.inc"

namespace {
using Clock = std::chrono::steady_clock;
constexpr size_t kBytes = 1024 * 1024, kReserve = 2048;
constexpr size_t kPending = 256, kGroupRows = 64;
constexpr unsigned kIterations = 8;
constexpr double kSeconds = 30;
enum class Lib { Nccl, Blas, Lt };
template<class T> T real(const char* name, Lib lib) {
  void* p = dlsym(RTLD_NEXT, name);
  if (!p) {
    // RTLD_LOCAL PyTorch DSOs need an explicit real-library handle. Absolute
    // overrides also support wheels whose private library path is not searched.
    static std::mutex mu;
    static void* handles[3] = {};
    std::lock_guard<std::mutex> lock(mu);
    unsigned i = static_cast<unsigned>(lib);
    const char* env[] = {"FLASH_V4_NCCL_LIBRARY", "FLASH_V4_CUBLAS_LIBRARY", "FLASH_V4_CUBLASLT_LIBRARY"};
    const char* defaults[] = {"libnccl.so.2", "libcublas.so.12", "libcublasLt.so.12"};
    if (!handles[i]) {
      const char* path = std::getenv(env[i]);
      handles[i] = dlopen(path ? path : defaults[i], RTLD_NOW | RTLD_LOCAL);
      if (!handles[i] && !path && i) handles[i] = dlopen(i == 1 ? "libcublas.so.11" : "libcublasLt.so.11", RTLD_NOW | RTLD_LOCAL);
    }
    if (handles[i]) p = dlsym(handles[i], name);
  }
  if (!p) { std::fprintf(stderr, "FLASH v4: cannot resolve real %s\n", name); std::abort(); }
  return reinterpret_cast<T>(p);
}
struct Row {
  const char* kind = "unknown";
  uint64_t iteration = 0, sequence = 0, stream = 0, comm = 0, numel = 0, group = 0;
  uint64_t m = 0, n = 0, k = 0;
  int dtype = -1, dtype_b = -1, dtype_c = -1, dtype_d = -1, compute = -1, op = -1, batch = 1;
  uint64_t kernel_id = 0, dynamic_shared_bytes = 0;
  unsigned grid[3] = {0,0,0}, block[3] = {0,0,0};
  const char* launch_api = "";
};
#include "observer_row_format.inc"
// Shape-local ordinals support comparisons of like distributions. They do not
// identify corresponding model operator instances across ranks or MoE branches.
using BlasKey = std::tuple<int,uint64_t,uint64_t,uint64_t,int,int,int,int,int,int,int>;
uint64_t stable_blas_hash(const BlasKey& key) {
  // FNV-1a-64 over eleven unsigned 64-bit little-endian words, in BlasKey
  // field order. Signed attributes use C++'s defined modulo-2^64 conversion.
  // No rank, iteration, fault state, std::hash, object padding or host endian.
  uint64_t hash = 14695981039346656037ULL;
  auto word = [&](auto value) {
    uint64_t bits = static_cast<uint64_t>(value);
    for (unsigned byte = 0; byte < 8; ++byte) {
      hash ^= (bits >> (8 * byte)) & 0xffULL;
      hash *= 1099511628211ULL;
    }
  };
  std::apply([&](auto... values) { (word(values), ...); }, key);
  return hash;
}
bool blas_sample_selected(uint64_t ordinal, unsigned offset, unsigned stride) {
  return ordinal % stride == offset;
}
unsigned blas_sampling_offset(uint64_t hash, uint64_t iteration, unsigned stride) {
  // Reduce before adding so UINT64_MAX iterations/hashes cannot overflow.
  return static_cast<unsigned>(((hash % stride) + (iteration % stride)) % stride);
}
bool heavy_blas(uint64_t m, uint64_t n, uint64_t k) {
  // 2*m*n*k >= 1e9, checked by saturating at 5e8 multiplies. Multiplication
  // occurs only while its result is below that cap, including UINT64_MAX dims.
  constexpr uint64_t limit = 500000000;
  if (!m || !n || !k) return false;
  uint64_t product = 1;
  for (uint64_t dimension : {m, n, k}) {
    if (dimension > (limit - 1) / product) return true;
    product *= dimension;
  }
  return false;
}
struct BlasSampling { uint64_t ordinal = 0; unsigned offset = 0; };
#include "observer_blas_cache.inc"
struct Timing {
  cudaEvent_t begin = nullptr, end = nullptr;
  int device = 0;
  std::vector<Row> rows;
};
struct Comm { uint64_t id; int local_rank, members; bool blocking; uint64_t iteration = ~uint64_t(0), sequence = 0; };
struct DeferredInit { ncclComm_t* ptr; Comm info; };
struct Group {
  int depth = 0;
  bool unsupported = false, mixed_stream = false, mixed_device = false, stream_seen = false;
  int device = -1;
  cudaStream_t stream = nullptr;
  Timing timing;
  const char* failure = nullptr;
  std::vector<Row> rows;
  std::vector<DeferredInit> inits;
};
thread_local Group group;

class Observer {
 public:
  std::atomic<bool> enabled{false}, stopped{false};
  std::atomic<const char*> stop_reason{nullptr};
  std::atomic<uint64_t> iteration{0};
  std::atomic<uint64_t> seen{0}, sampled{0}, dropped{0}, skipped{0}, heavy_sampled{0};
  std::atomic<uint64_t> kernel_sequence{0}, kernel_candidates{0}, kernel_sampled{0}, kernel_emitted{0}, kernel_dropped{0}, kernel_skipped{0};
  std::atomic<uint64_t> kernel_nccl_excluded{0}, kernel_unknown_excluded{0}, kernel_blas_excluded{0}, kernel_nested_excluded{0};
  std::mutex mu, output_mu, begin_mu, comm_mu, shape_mu;
  std::map<BlasKey,BlasSampling> shape_ordinals;
  BlasSamplingFrontCache shape_front;
  std::condition_variable cv;
  std::deque<Timing> pending;
  size_t in_flight = 0;
  std::unordered_map<ncclComm_t, Comm> comms;
  std::thread worker;
  flash_v4::detail::ObserverOutputBuffer output_buffer;
#include "observer_event_pool.inc"
  bool started = false, closing = false, flushed = false, output_failed = false, iteration_open = false;
  unsigned iterations = 0, completed_iterations = 0, stride = 16;
  uint64_t last_iteration = ~uint64_t(0), peak = 0, bytes = 0;
  std::atomic<uint64_t> emitted{0};
  Clock::time_point start;
  int fd = -1, rank = -1;
  bool kernel_enabled = false;
  std::atomic<uint64_t> allreduce_profile_calls{0};
  Observer() {
    if (const char* s = std::getenv("RANK")) rank = std::atoi(s);
    if (const char* s = std::getenv("FLASH_V4_SAMPLE_STRIDE")) stride = std::max(1, std::atoi(s));
    if (const char* s = std::getenv("FLASH_V4_KERNEL_PROFILE")) kernel_enabled = std::strcmp(s,"1") == 0;
  }
  void stop(const char* reason) {
    const char* expected = nullptr;
    stop_reason.compare_exchange_strong(expected, reason);
    stopped.store(true); enabled.store(false);
  }
  bool write_line(const char* line, bool reserved = false) {
    size_t length = std::strlen(line);
    std::lock_guard<std::mutex> lock(output_mu);
    if (rank < 0 || output_failed) { ++dropped; return false; }
    if (length > 1024 || bytes + length > kBytes - (reserved ? 0 : kReserve)) {
      stop(length > 1024 ? "event_byte_budget" : "byte_budget"); ++dropped; return false;
    }
    if (fd < 0) {
      const char* dir = std::getenv("FLASH_V4_OUTPUT_DIR");
      std::string path = std::string(dir ? dir : "/tmp") + "/rank_" + std::to_string(rank) + ".jsonl";
      // Never silently append separate runs into one evidence file.
      fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
      if (fd < 0) { std::perror("FLASH v4 output (use a fresh output directory)"); output_failed = true; stop("output_error"); ++dropped; return false; }
    }
    auto writer = [this](const char* data, size_t count) { return ::write(fd, data, count); };
    if (!output_buffer.append(line, length, writer)) {
      output_failed = true; stop("output_error"); ++dropped; return false;
    }
    // Account on acceptance so staged bytes cannot evade the fixed budget.
    bytes += length;
    if (reserved && !output_buffer.flush(writer)) {
      output_failed = true; stop("output_error"); ++dropped; return false;
    }
    return true;
  }
  void drain_output() {
    std::lock_guard<std::mutex> lock(output_mu);
    if (output_failed || !output_buffer.buffered_bytes()) return;
    if (!output_buffer.flush([this](const char* data, size_t count) { return ::write(fd, data, count); })) {
      output_failed = true; stop("output_error"); ++dropped;
    }
  }
  void row(const Row& r, double us, const char* reason = nullptr) {
    char line[1024];
    char elapsed[64];
    if (reason) std::strcpy(elapsed, "null"); else std::snprintf(elapsed, sizeof(elapsed), "%.3f", us);
    const bool formatted = format_observer_row(line, sizeof(line), r, rank, elapsed, reason);
    if (std::strcmp(r.kind, "kernel") == 0) {
      if (!formatted) std::snprintf(line,sizeof(line),"{\"kind\":\"kernel\",\"rank\":%d,\"iteration\":%llu,\"sequence\":%llu,\"kernel_id\":\"%016llx\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"dynamic_shared_bytes\":%llu,\"launch_api\":\"%s\",\"sampling_stride\":256,\"elapsed_us\":%s,\"skip\":\"%s\"}\n",
        rank,(unsigned long long)r.iteration,(unsigned long long)r.sequence,(unsigned long long)r.kernel_id,
        r.grid[0],r.grid[1],r.grid[2],r.block[0],r.block[1],r.block[2],
        (unsigned long long)r.dynamic_shared_bytes,r.launch_api,elapsed,reason?reason:"");
      if(reason) { ++skipped;++kernel_skipped; }
      if(write_line(line)) { ++emitted;++kernel_emitted; } else ++kernel_dropped;
      return;
    }
    if (!formatted) std::snprintf(line, sizeof(line), "{\"kind\":\"%s\",\"rank\":%d,\"iteration\":%llu,\"sequence\":%llu,\"stream\":%llu,\"comm_id\":\"%016llx\",\"op\":%d,\"numel\":%llu,\"dtype\":%d,\"dtype_b\":%d,\"dtype_c\":%d,\"dtype_d\":%d,\"compute\":%d,\"batch\":%d,\"shape\":[%llu,%llu,%llu],\"elapsed_us\":%s,\"group\":%llu,\"timing_scope\":\"%s\",\"skip\":\"%s\"}\n",
      r.kind, rank, (unsigned long long)r.iteration, (unsigned long long)r.sequence,
      (unsigned long long)r.stream, (unsigned long long)r.comm, r.op,
      (unsigned long long)r.numel, r.dtype, r.dtype_b, r.dtype_c, r.dtype_d, r.compute, r.batch, (unsigned long long)r.m,
      (unsigned long long)r.n, (unsigned long long)r.k, elapsed,
      (unsigned long long)r.group, r.group ? "nccl_group_envelope" : "library_call_envelope", reason ? reason : "");
    if (reason) ++skipped;
    if (write_line(line)) ++emitted;
  }
  void begin(uint64_t it, bool profile) {
    std::lock_guard<std::mutex> lock(begin_mu);
    if (iteration_open && (it != last_iteration || !profile)) {
      stop("missing_iteration_end"); iteration_open = false;
    }
    enabled.store(false);
    iteration.store(it);
    if (rank < 0 || !profile || stopped.load() || flushed) return;
    if (!started) { start = Clock::now(); started = true; worker = std::thread([this]{ run(); }); }
    if (last_iteration != it && iterations >= kIterations) { stop("iteration_budget"); return; }
    if (last_iteration != it) {
      ++iterations; last_iteration = it;
      kernel_sequence.store(0);
      std::lock_guard<std::mutex> shapes(shape_mu); shape_ordinals.clear();
      shape_front.clear();
    }
    if (std::chrono::duration<double>(Clock::now() - start).count() >= kSeconds) {
      stop("time_budget"); return;
    }
    iteration_open = true;
    enabled.store(true);
  }
  void end(uint64_t it) {
    std::lock_guard<std::mutex> lock(begin_mu);
    if (!started || !iteration_open || it != last_iteration) return;
    // This is the host schedule FWD/BWD return boundary, without GPU sync.
    // Even if no late intercepted call queried active(), a late return must
    // not certify the final iteration as complete after the deadline.
    if (std::chrono::duration<double>(Clock::now() - start).count() >= kSeconds) stop("time_budget");
    if (enabled.load() && !stopped.load()) ++completed_iterations;
    enabled.store(false); iteration_open = false;
  }
  bool active() {
    if (!enabled.load(std::memory_order_relaxed)) return false;
    if (std::chrono::duration<double>(Clock::now() - start).count() >= kSeconds) { stop("time_budget"); return false; }
    return true;
  }
  bool select(uint64_t seq) { ++seen; if (seq % stride) return false; ++sampled; return true; }
  bool select_blas(Row& r) {
    unsigned offset = 0;
    {
      std::lock_guard<std::mutex> lock(shape_mu);
      BlasKey key{std::strcmp(r.kind, "gemm") == 0 ? 0 : 1, r.m, r.n, r.k,
        r.dtype, r.dtype_b, r.dtype_c, r.dtype_d, r.compute, r.op, r.batch};
      BlasSampling* sampling = shape_front.get(key);
      if (!sampling) {
        auto it = shape_ordinals.find(key);
        if (it == shape_ordinals.end()) {
          // Bound host bookkeeping even when dynamic MoE shapes proliferate.
          if (shape_ordinals.size() >= 4096) { ++seen; ++dropped; return false; }
          it = shape_ordinals.emplace(key, BlasSampling{0, blas_sampling_offset(stable_blas_hash(key), r.iteration, stride)}).first;
        }
        sampling = &it->second;
        shape_front.put(key, sampling);
      }
      r.sequence = sampling->ordinal++;
      offset = sampling->offset;
    }
    ++seen;
    if (heavy_blas(r.m, r.n, r.k)) { ++sampled; ++heavy_sampled; return true; }
    if (!blas_sample_selected(r.sequence, offset, stride)) return false;
    ++sampled;
    return true;
  }
  void destroy(Timing& t) { if (t.begin) cudaEventDestroy(t.begin); if (t.end) cudaEventDestroy(t.end); t.begin = t.end = nullptr; }
  const char* event_start(Timing& t, cudaStream_t stream) {
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone) return "cuda_graph_or_capture_unknown";
    if (cudaGetDevice(&t.device) != cudaSuccess) return "cuda_device_error";
    if (!take_cached(t) &&
        (cudaEventCreate(&t.begin) != cudaSuccess || cudaEventCreate(&t.end) != cudaSuccess)) { destroy(t); return "event_create_error"; }
    if (cudaEventRecord(t.begin, stream) != cudaSuccess) { destroy(t); return "event_record_error"; }
    return nullptr;
  }
  void finish(Timing& t, cudaStream_t stream, const char* failure = nullptr) {
    if (!failure && cudaEventRecord(t.end, stream) != cudaSuccess) failure = "event_record_error";
    if (failure) { for (auto& r : t.rows) row(r, -1, failure); destroy(t); return; }
    std::lock_guard<std::mutex> lock(mu);
    if (closing || pending.size() + in_flight >= kPending) {
      dropped += t.rows.size();
      for(auto& r:t.rows) if(std::strcmp(r.kind,"kernel")==0) ++kernel_dropped;
      destroy(t); return;
    }
    // Wake an idle worker for the first event. While it has outstanding work,
    // retain its existing 2ms polling cadence instead of interrupting that
    // wait on every intercepted launch. Closing still always wakes the worker.
    const bool wake_idle = pending.empty() && in_flight == 0;
    pending.push_back(std::move(t)); t = Timing{}; peak = std::max<uint64_t>(peak, pending.size() + in_flight);
    if (wake_idle) cv.notify_one();
  }
  void run() {
    Clock::time_point closing_start;
    bool draining = false;
    // This worker owns its CUDA current-device state. Re-select on a device
    // change or any failed selection, rather than for every pending event.
    int worker_device = -1;
    std::unique_lock<std::mutex> lock(mu);
    for (;;) {
      if (closing && !draining) { draining = true; closing_start = Clock::now(); }
      if (closing && pending.empty()) break;
      // Query all entries: one blocked stream cannot starve completed streams.
      size_t count = pending.size();
      for (size_t i = 0; i < count; ++i) {
        Timing t = std::move(pending.front()); pending.pop_front(); ++in_flight; lock.unlock();
        cudaError_t rc = cudaSuccess;
        if (worker_device != t.device) {
          rc = cudaSetDevice(t.device);
          worker_device = rc == cudaSuccess ? t.device : -1;
        }
        if (rc == cudaSuccess) rc = cudaEventQuery(t.end);
        bool timeout = draining && std::chrono::duration<double>(Clock::now() - closing_start).count() > 5;
        if (rc == cudaErrorNotReady && !timeout) { lock.lock(); --in_flight; pending.push_back(std::move(t)); continue; }
        float ms = 0;
        const char* reason = nullptr;
        if (timeout && rc == cudaErrorNotReady) reason = "exit_pending_timeout";
        else if (rc != cudaSuccess || cudaEventElapsedTime(&ms, t.begin, t.end) != cudaSuccess) reason = "event_query_error";
        for (auto& r : t.rows) row(r, reason ? -1 : ms * 1000.0, reason);
        if (reason || !cache_completed(t)) destroy(t);
        lock.lock(); --in_flight;
      }
      cv.wait_for(lock, std::chrono::milliseconds(2));
    }
  }
  void flush() {
    std::lock_guard<std::mutex> serial(begin_mu);
    if (flushed) return;
    flushed = true; stopped.store(true); enabled.store(false);
    if (rank < 0) return;
    { std::lock_guard<std::mutex> lock(mu); closing = true; cv.notify_one(); }
    if (worker.joinable()) worker.join();
    // A failed data drain must be visible before certifying completeness.
    drain_output();
    const bool pool_cleared = clear_cached();
    const char* reason = stop_reason.load();
    bool normal_stop = !reason || std::strcmp(reason, "iteration_budget") == 0;
    bool complete = completed_iterations == kIterations && !iteration_open && normal_stop && dropped.load() == 0;
    if (!reason) reason = !started ? "profile_not_started" : (iteration_open ? "flush_open_iteration" :
      (dropped.load() ? "data_dropped" : (complete ? "window_closed" : "requested_window_ended")));
    char line[1024];
    std::snprintf(line, sizeof(line), "{\"kind\":\"summary\",\"rank\":%d,\"profile_seen\":%llu,\"sampled\":%llu,\"heavy_sampled\":%llu,\"skipped\":%llu,\"dropped\":%llu,\"emitted\":%llu,\"pending_peak\":%llu,\"data_bytes\":%llu,\"stride\":%u,\"profile_iterations\":%u,\"completed_profile_iterations\":%u,\"window_complete\":%s,\"stop_reason\":\"%s\",\"max_iterations\":8,\"max_seconds\":30,\"max_bytes\":1048576,\"blas_sampling\":\"heavy1e9_all;shape_fnv+iter\",\"comparison\":\"distributions\",\"row_schema\":\"compute_compact_v1\",\"coverage\":\"GEMM,sparse_kernel_envelopes;AR_calls_only\",\"allreduce\":{\"detailed\":false,\"profile_calls\":%llu},\"kernel\":{\"enabled\":%s,\"candidates\":%llu,\"sampled\":%llu,\"emitted\":%llu,\"dropped\":%llu,\"skipped\":%llu,\"nccl\":%llu,\"unknown\":%llu,\"blas\":%llu,\"nested\":%llu,\"stride\":256,\"id_hash\":\"fnv1a64_device_name\"}}\n",
      rank, (unsigned long long)seen.load(), (unsigned long long)sampled.load(),
      (unsigned long long)heavy_sampled.load(), (unsigned long long)skipped.load(), (unsigned long long)dropped.load(),
      (unsigned long long)emitted.load(), (unsigned long long)peak, (unsigned long long)bytes, stride, std::min(iterations, kIterations), completed_iterations, complete ? "true" : "false", reason,
      (unsigned long long)allreduce_profile_calls.load(),
      kernel_enabled?"true":"false",(unsigned long long)kernel_candidates.load(),(unsigned long long)kernel_sampled.load(),
      (unsigned long long)kernel_emitted.load(),(unsigned long long)kernel_dropped.load(),(unsigned long long)kernel_skipped.load(),
      (unsigned long long)kernel_nccl_excluded.load(),(unsigned long long)kernel_unknown_excluded.load(),
      (unsigned long long)kernel_blas_excluded.load(),(unsigned long long)kernel_nested_excluded.load());
    const size_t summary_tail = std::strlen(line) - 2;  // Replace final } and newline.
    std::snprintf(line + summary_tail, sizeof(line) - summary_tail,
      ",\"event_pool\":{\"reused\":%llu,\"cached\":%llu,\"high_water\":%llu,\"capacity\":%llu,\"cleanup_complete\":%s}}\n",
      (unsigned long long)event_pairs_reused.load(), (unsigned long long)event_pairs_cached.load(),
      (unsigned long long)event_pool_high_water.load(), (unsigned long long)kPending, pool_cleared ? "true" : "false");
    write_line(line, true);
    if (fd >= 0) { close(fd); fd = -1; }
  }
};
Observer& observer() { static Observer* o = new Observer; return *o; }
uint64_t hash_id(const ncclUniqueId& id) { uint64_t h = 1469598103934665603ULL; for (unsigned char c : id.internal) { h ^= c; h *= 1099511628211ULL; } return h; }
void register_comm(ncclComm_t* ptr, Comm info) {
  if (!ptr || !*ptr) return;
  auto& o = observer();
  if (o.rank < 0) return;
  { std::lock_guard<std::mutex> lock(o.comm_mu); o.comms[*ptr] = info; }
  char line[512];
  std::snprintf(line, sizeof(line), "{\"kind\":\"communicator\",\"rank\":%d,\"comm_id\":\"%016llx\",\"comm_rank\":%d,\"members\":%d,\"blocking\":%s,\"mapping\":\"join_global_rank_records_by_comm_id\"}\n", o.rank, (unsigned long long)info.id, info.local_rank, info.members, info.blocking ? "true" : "false");
  o.write_line(line);
}
void init_result(ncclResult_t rc, ncclComm_t* ptr, int n, ncclUniqueId id, int rank, bool blocking) {
  if (rc != ncclSuccess) return;
  Comm info{hash_id(id), rank, n, blocking};
  // Public NCCL group initialization writes *ptr only at outer GroupEnd.
  if (group.depth) {
    if (group.inits.size() < kGroupRows) group.inits.push_back({ptr, info}); else ++observer().dropped;
  } else register_comm(ptr, info);
}
Row base_row(const char* kind, cudaStream_t stream, uint64_t sequence) {
  Row r; r.kind = kind; r.iteration = observer().iteration.load(); r.stream = reinterpret_cast<uintptr_t>(stream); r.sequence = sequence; return r;
}
void gemm_group_mark() { if (group.depth) group.unsupported = true; }
template<class Call> cublasStatus_t timed_blas(Row r, cudaStream_t stream, const char* failure, Call call) {
  auto& o = observer(); gemm_group_mark();
  Timing t; if (!failure) failure = o.event_start(t, stream);
  auto rc = call();
  t.rows.push_back(r);
  o.finish(t, stream, failure ? failure : (rc == CUBLAS_STATUS_SUCCESS ? nullptr : "cublas_call_error"));
  return rc;
}
std::atomic<uint64_t> group_sequence{0};
thread_local uint64_t current_group = 0;
thread_local bool inside_blas = false;
struct BlasGuard { BlasGuard() { inside_blas = true; } ~BlasGuard() { inside_blas = false; } };
} // namespace

extern "C" void flash_v4_begin_iteration(uint64_t iteration, int profile_enabled) { observer().begin(iteration, profile_enabled != 0); }
extern "C" void flash_v4_end_iteration(uint64_t iteration) { observer().end(iteration); }
extern "C" void flash_v4_flush() { observer().flush(); }

extern "C" cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* a, cudaDataType atype, int lda, const void* b, cudaDataType btype, int ldb, const void* beta, void* c, cudaDataType ctype, int ldc, cublasComputeType_t compute, cublasGemmAlgo_t algo) {
  using GemmExFn = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int, const void*, const void*, cudaDataType, int, const void*, cudaDataType,
    int, const void*, void*, cudaDataType, int, cublasComputeType_t, cublasGemmAlgo_t);
  static auto fn = real<GemmExFn>("cublasGemmEx", Lib::Blas);
  auto call = [&]{ return fn(handle, transa, transb, m, n, k, alpha, a, atype, lda, b, btype, ldb, beta, c, ctype, ldc, compute, algo); };
  auto& o = observer(); if (!o.active() || inside_blas) return call();
  BlasGuard guard;
  gemm_group_mark();
  Row r = base_row("gemm", nullptr, 0); r.m = m; r.n = n; r.k = k; r.numel = uint64_t(std::max(m, 0)) * std::max(n, 0);
  r.dtype = atype; r.dtype_b = btype; r.dtype_c = r.dtype_d = ctype; r.compute = compute; r.op = int(transa) * 16 + int(transb);
  if (!o.select_blas(r)) return call();
  static auto get_stream = real<decltype(&cublasGetStream_v2)>("cublasGetStream_v2", Lib::Blas);
  cudaStream_t stream = nullptr; auto sr = get_stream(handle, &stream);
  r.stream = reinterpret_cast<uintptr_t>(stream);
  return timed_blas(r, stream, sr == CUBLAS_STATUS_SUCCESS ? nullptr : "cublas_stream_unknown", call);
}

extern "C" cublasStatus_t cublasLtMatmul(cublasLtHandle_t handle, cublasLtMatmulDesc_t desc, const void* alpha, const void* a, cublasLtMatrixLayout_t al, const void* b, cublasLtMatrixLayout_t bl, const void* beta, const void* c, cublasLtMatrixLayout_t cl, void* d, cublasLtMatrixLayout_t dl, const cublasLtMatmulAlgo_t* algo, void* workspace, size_t workspace_bytes, cudaStream_t stream) {
  static auto fn = real<decltype(&cublasLtMatmul)>("cublasLtMatmul", Lib::Lt);
  auto call = [&]{ return fn(handle, desc, alpha, a, al, b, bl, beta, c, cl, d, dl, algo, workspace, workspace_bytes, stream); };
  auto& o = observer(); if (!o.active() || inside_blas) return call();
  BlasGuard guard;
  gemm_group_mark();
  static auto layout_attr = real<decltype(&cublasLtMatrixLayoutGetAttribute)>("cublasLtMatrixLayoutGetAttribute", Lib::Lt);
  static auto desc_attr = real<decltype(&cublasLtMatmulDescGetAttribute)>("cublasLtMatmulDescGetAttribute", Lib::Lt);
  Row r = base_row("lt_matmul", stream, 0);
  uint64_t ar = 0, ac = 0, dr = 0, dc = 0; int32_t batch = 0, abatch = 0, bbatch = 0, cbatch = 0;
  cudaDataType dtype{}, btype{}, ctype{}, dtype_out{}; cublasOperation_t ta{}, tb{}; cublasComputeType_t compute{}; size_t written = 0;
  bool ok = layout_attr(al, CUBLASLT_MATRIX_LAYOUT_ROWS, &ar, sizeof(ar), &written) == CUBLAS_STATUS_SUCCESS;
  ok &= layout_attr(al, CUBLASLT_MATRIX_LAYOUT_COLS, &ac, sizeof(ac), &written) == CUBLAS_STATUS_SUCCESS;
  ok &= layout_attr(dl, CUBLASLT_MATRIX_LAYOUT_ROWS, &dr, sizeof(dr), &written) == CUBLAS_STATUS_SUCCESS;
  ok &= layout_attr(dl, CUBLASLT_MATRIX_LAYOUT_COLS, &dc, sizeof(dc), &written) == CUBLAS_STATUS_SUCCESS;
  ok &= layout_attr(al, CUBLASLT_MATRIX_LAYOUT_TYPE, &dtype, sizeof(dtype), &written) == CUBLAS_STATUS_SUCCESS;
  ok &= layout_attr(bl, CUBLASLT_MATRIX_LAYOUT_TYPE, &btype, sizeof(btype), &written) == CUBLAS_STATUS_SUCCESS;
  ok &= layout_attr(cl, CUBLASLT_MATRIX_LAYOUT_TYPE, &ctype, sizeof(ctype), &written) == CUBLAS_STATUS_SUCCESS;
  ok &= layout_attr(dl, CUBLASLT_MATRIX_LAYOUT_TYPE, &dtype_out, sizeof(dtype_out), &written) == CUBLAS_STATUS_SUCCESS;
  ok &= layout_attr(dl, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch), &written) == CUBLAS_STATUS_SUCCESS;
  ok &= layout_attr(al, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &abatch, sizeof(abatch), &written) == CUBLAS_STATUS_SUCCESS;
  ok &= layout_attr(bl, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bbatch, sizeof(bbatch), &written) == CUBLAS_STATUS_SUCCESS;
  ok &= layout_attr(cl, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &cbatch, sizeof(cbatch), &written) == CUBLAS_STATUS_SUCCESS;
  ok &= desc_attr(desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta), &written) == CUBLAS_STATUS_SUCCESS;
  ok &= desc_attr(desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb), &written) == CUBLAS_STATUS_SUCCESS;
  ok &= desc_attr(desc, CUBLASLT_MATMUL_DESC_COMPUTE_TYPE, &compute, sizeof(compute), &written) == CUBLAS_STATUS_SUCCESS;
  if (ok) {
    r.m = dr; r.n = dc; r.k = ta == CUBLAS_OP_N ? ac : ar; r.dtype = dtype; r.dtype_b = btype; r.dtype_c = ctype; r.dtype_d = dtype_out;
    r.compute = compute; r.batch = batch; r.op = int(ta) * 16 + int(tb); r.numel = dr && dc > UINT64_MAX / dr ? 0 : dr * dc;
  }
  // Descriptor reads precede shape-local sampling, so their PROFILE overhead
  // must be included in measured end-to-end overhead even for unsampled calls.
  if (!o.select_blas(r)) return call();
  return timed_blas(r, stream, !ok ? "lt_shape_unknown" : (batch != 1 || abatch != 1 || bbatch != 1 || cbatch != 1 ? "lt_batched_layout_unsupported" : nullptr), call);
}

extern "C" ncclResult_t ncclCommInitRank(ncclComm_t* comm, int n, ncclUniqueId id, int rank) {
  static auto fn = real<decltype(&ncclCommInitRank)>("ncclCommInitRank", Lib::Nccl);
  const char* blocking_env = std::getenv("NCCL_COMM_BLOCKING");
  auto rc = fn(comm, n, id, rank); init_result(rc, comm, n, id, rank, !blocking_env || std::strcmp(blocking_env, "0") != 0); return rc;
}
extern "C" ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int n, ncclUniqueId id, int rank, ncclConfig_t* config) {
  static auto fn = real<decltype(&ncclCommInitRankConfig)>("ncclCommInitRankConfig", Lib::Nccl);
  auto rc = fn(comm, n, id, rank, config);
  // Only a confirmed blocking config is supported. Undefined/default config is
  // conservative: skip timing, because NCCL environment overrides may apply.
  init_result(rc, comm, n, id, rank, config && config->blocking == 1); return rc;
}
extern "C" ncclResult_t ncclGroupStart() {
  static auto fn = real<decltype(&ncclGroupStart)>("ncclGroupStart", Lib::Nccl);
  auto rc = fn();
  if (rc == ncclSuccess) { if (!group.depth) { group = Group{}; current_group = ++group_sequence; } ++group.depth; }
  return rc;
}
extern "C" ncclResult_t ncclGroupEnd() {
  static auto fn = real<decltype(&ncclGroupEnd)>("ncclGroupEnd", Lib::Nccl);
  auto rc = fn();
  if (group.depth <= 0 || --group.depth) return rc;
  auto& o = observer();
  if (rc == ncclSuccess) for (auto& init : group.inits) register_comm(init.ptr, init.info);
  if (!group.rows.empty()) {
    group.timing.rows = std::move(group.rows);
    const char* failure = group.failure;
    if (rc != ncclSuccess) failure = "nccl_group_error_or_inprogress";
    else if (group.mixed_device) failure = "nccl_group_multidevice";
    else if (group.mixed_stream) failure = "nccl_group_multistream";
    else if (group.unsupported) failure = "nccl_group_mixed_or_unsupported";
    o.finish(group.timing, group.stream, failure);
  }
  group = Group{}; return rc;
}
extern "C" ncclResult_t ncclAllReduce(const void* in, void* out, size_t count, ncclDataType_t dtype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  static auto fn = real<decltype(&ncclAllReduce)>("ncclAllReduce", Lib::Nccl);
  auto call = [&]{ return fn(in, out, count, dtype, op, comm, stream); };
  auto& o = observer(); if (!o.active()) { if (group.depth && !group.rows.empty()) group.unsupported = true; return call(); }
  ++o.allreduce_profile_calls;
  // R84 compute-slow scope: forward every real collective, count active-window
  // calls, retain the original detailed communication path below as comments.
  return call();
//   Row r = base_row("allreduce", stream, 0); r.dtype = dtype; r.op = op; r.numel = count; r.m = count;
//   bool known = false, blocking = false;
//   { std::lock_guard<std::mutex> lock(o.comm_mu); auto it = o.comms.find(comm);
//     if (it != o.comms.end()) { auto& info = it->second; known = true; blocking = info.blocking; r.comm = info.id; if (info.iteration != r.iteration) { info.iteration = r.iteration; info.sequence = 0; } r.sequence = info.sequence++; }
//   }
//   if (group.depth) {
//     if (!group.rows.empty() && group.rows.front().iteration != r.iteration) group.unsupported = true;
//     int device = -1;
//     if (cudaGetDevice(&device) != cudaSuccess) group.unsupported = true;
//     else if (group.device < 0) group.device = device;
//     else if (group.device != device) group.mixed_device = true;
//     if (group.stream_seen && group.stream != stream) group.mixed_stream = true;
//     if (!group.stream_seen) { group.stream = stream; group.stream_seen = true; }
//     if (!known || !blocking) group.unsupported = true;
//   }
//   if (!o.select(r.sequence)) return call();
//   const char* failure = !known ? "communicator_unregistered" : (!blocking ? "nccl_nonblocking_or_unknown" : nullptr);
//   if (group.depth) {
//     r.group = current_group;
//     if (group.rows.size() >= kGroupRows) { ++o.dropped; group.unsupported = true; return call(); }
//     if (group.rows.empty()) {
//       group.failure = failure ? failure : o.event_start(group.timing, stream);
//       // The cached timing owns empty row storage; collect this group into it.
//       group.rows.swap(group.timing.rows);
//     }
//     if (failure) group.unsupported = true;
//     group.rows.push_back(r);
//     auto rc = call(); if (rc != ncclSuccess) group.unsupported = true; return rc;
//   }
//   Timing t; if (!failure) failure = o.event_start(t, stream);
//   auto rc = call(); t.rows.push_back(r);
//   o.finish(t, stream, failure ? failure : (rc == ncclSuccess ? nullptr : "nccl_call_error_or_inprogress")); return rc;
}
// Common non-AllReduce APIs invalidate a sampled group, including calls before
// its first sampled AllReduce. Unlisted/future APIs are outside coverage.
#define FLASH_UNSUPPORTED_NCCL(name, signature, arguments) \
extern "C" ncclResult_t name signature { \
  static auto fn = real<decltype(&name)>(#name, Lib::Nccl); \
  if (group.depth) group.unsupported = true; \
  return fn arguments; }
FLASH_UNSUPPORTED_NCCL(ncclBroadcast, (const void* a, void* b, size_t n, ncclDataType_t d, int root, ncclComm_t c, cudaStream_t s), (a,b,n,d,root,c,s))
FLASH_UNSUPPORTED_NCCL(ncclReduce, (const void* a, void* b, size_t n, ncclDataType_t d, ncclRedOp_t op, int root, ncclComm_t c, cudaStream_t s), (a,b,n,d,op,root,c,s))
FLASH_UNSUPPORTED_NCCL(ncclAllGather, (const void* a, void* b, size_t n, ncclDataType_t d, ncclComm_t c, cudaStream_t s), (a,b,n,d,c,s))
FLASH_UNSUPPORTED_NCCL(ncclReduceScatter, (const void* a, void* b, size_t n, ncclDataType_t d, ncclRedOp_t op, ncclComm_t c, cudaStream_t s), (a,b,n,d,op,c,s))
FLASH_UNSUPPORTED_NCCL(ncclSend, (const void* a, size_t n, ncclDataType_t d, int peer, ncclComm_t c, cudaStream_t s), (a,n,d,peer,c,s))
FLASH_UNSUPPORTED_NCCL(ncclRecv, (void* a, size_t n, ncclDataType_t d, int peer, ncclComm_t c, cudaStream_t s), (a,n,d,peer,c,s))
#undef FLASH_UNSUPPORTED_NCCL

extern "C" ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  static auto fn = real<decltype(&ncclCommDestroy)>("ncclCommDestroy", Lib::Nccl);
  auto rc = fn(comm);
  if (rc == ncclSuccess) { auto& o = observer(); std::lock_guard<std::mutex> lock(o.comm_mu); o.comms.erase(comm); }
  return rc;
}
extern "C" ncclResult_t ncclCommAbort(ncclComm_t comm) {
  static auto fn = real<decltype(&ncclCommAbort)>("ncclCommAbort", Lib::Nccl);
  auto rc = fn(comm);
  if (rc == ncclSuccess) { auto& o = observer(); std::lock_guard<std::mutex> lock(o.comm_mu); o.comms.erase(comm); }
  return rc;
}

#include "observer_kernels.inc"
