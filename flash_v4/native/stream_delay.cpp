// Independent, process-local proportional compute-stream delay actor.
// ABI/configuration and logs are private actuation data, never detector input.
// Ordinary runtime/driver launches only; graph/cooperative launches are outside
// coverage. Estimates cover intercepted launch envelopes, not kernel-only time.
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <cerrno>

// Expose both historical and current entry-point lookup ABIs explicitly.
#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
extern "C" cudaError_t flash_v4_delay_launch_wait(cudaStream_t, unsigned long long);

namespace {
using Clock = std::chrono::steady_clock;
constexpr size_t kCache = 4096, kPending = 256, kStreams = 128;
constexpr uint64_t kCalibrationEvents = 2048, kSamplesPerKey = 4;
constexpr uint64_t kSamplesPerWarmStepKey = 2, kSamplesPerWarmStepPhase = 512;
#ifndef FLASH_V4_ACTOR_MAX_STEP_US
#define FLASH_V4_ACTOR_MAX_STEP_US 200000
#endif
#if FLASH_V4_ACTOR_MAX_STEP_US < 1 || FLASH_V4_ACTOR_MAX_STEP_US > 1000000
#error "FLASH_V4_ACTOR_MAX_STEP_US must be between 1 and 1000000"
#endif
constexpr double kQuantumUs = 20, kSingleUs = 200, kStepUs = FLASH_V4_ACTOR_MAX_STEP_US;
#ifndef FLASH_V4_ACTOR_MAX_FAULT_ITERATIONS
#define FLASH_V4_ACTOR_MAX_FAULT_ITERATIONS 8
#endif
#if FLASH_V4_ACTOR_MAX_FAULT_ITERATIONS < 1 || FLASH_V4_ACTOR_MAX_FAULT_ITERATIONS > 32
#error "FLASH_V4_ACTOR_MAX_FAULT_ITERATIONS must be between 1 and 32"
#endif
constexpr unsigned kFaultIterations = FLASH_V4_ACTOR_MAX_FAULT_ITERATIONS;
constexpr double kFaultSeconds = 30;
thread_local bool internal = false;
struct Guard { Guard() { internal = true; } ~Guard() { internal = false; } };

void* driver_library() {
  static void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  return library;
}
template<class T> T driver(const char* name) {
  void* p = driver_library() ? dlsym(driver_library(), name) : nullptr;
  return reinterpret_cast<T>(p);
}
template<class T> T actual(const char* name, bool runtime) {
  void* p = dlsym(RTLD_NEXT, name);
  if (!p) {
    static void* rt = nullptr;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (runtime && !rt) {
      const char* path = std::getenv("FLASH_V4_DELAY_CUDART_LIBRARY");
      rt = dlopen(path ? path : "libcudart.so.12", RTLD_NOW | RTLD_LOCAL);
    }
    void* handle = runtime ? rt : driver_library();
    if (handle) p = dlsym(handle, name);
  }
  if (!p) { std::fprintf(stderr, "v4 stream delay: missing real %s\n", name); std::abort(); }
  return reinterpret_cast<T>(p);
}
struct Driver {
  decltype(&cuCtxGetCurrent) current = driver<decltype(current)>("cuCtxGetCurrent");
  decltype(&cuCtxPushCurrent) push = driver<decltype(push)>("cuCtxPushCurrent_v2");
  decltype(&cuCtxPopCurrent) pop = driver<decltype(pop)>("cuCtxPopCurrent_v2");
  decltype(&cuStreamIsCapturing) capture = driver<decltype(capture)>("cuStreamIsCapturing");
  decltype(&cuEventCreate) create = driver<decltype(create)>("cuEventCreate");
  decltype(&cuEventRecord) record = driver<decltype(record)>("cuEventRecord");
  decltype(&cuEventQuery) query = driver<decltype(query)>("cuEventQuery");
  decltype(&cuEventElapsedTime) elapsed = driver<decltype(elapsed)>("cuEventElapsedTime");
  decltype(&cuEventDestroy) destroy = driver<decltype(destroy)>("cuEventDestroy_v2");
  using Name = CUresult (*)(const char**, CUfunction);
  Name name = driver<Name>("cuFuncGetName");
  bool valid() const { return current && push && pop && capture && create && record && query && elapsed && destroy; }
};
Driver& cuda() { static Driver api; return api; }
using RuntimeName = cudaError_t (*)(const char**,const void*);
RuntimeName runtime_name() {
  // Query the same runtime DSO as the real launch entry. A host function from
  // one CUDA runtime instance must not be passed to another runtime instance.
  static RuntimeName name=[] {
    Dl_info info{};
    if(!dladdr(actual<void*>("cudaLaunchKernel",true),&info)||!info.dli_fname) return RuntimeName(nullptr);
    void* handle=dlopen(info.dli_fname,RTLD_NOW|RTLD_LOCAL);
    return handle?reinterpret_cast<RuntimeName>(dlsym(handle,"cudaFuncGetName")):nullptr;
  }();
  return name;
}
// Attribute object representation is a process-local cache discriminator only;
// padding can reduce hits, but never selects a rank or an injection location.
template<class T> uint64_t attrs_hash(const T* attrs, unsigned count) {
  uint64_t hash = 14695981039346656037ULL;
  const unsigned char* p = reinterpret_cast<const unsigned char*>(attrs);
  for (size_t n = 0; n < sizeof(T) * count; ++n) { hash ^= p[n]; hash *= 1099511628211ULL; }
  return hash;
}
using Key = std::tuple<uintptr_t,uintptr_t,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,size_t,uint64_t,int>;
using StreamKey = std::tuple<uintptr_t,uintptr_t,uint64_t>;
bool calibration_iteration(uint64_t iteration) { return iteration == 2 || iteration == 3; }
struct Estimate {
  double us = 0;
  uint64_t samples = 0, pending = 0, warm_iteration = 0, warm_attempts = 0;
};
bool accept_estimate_sample(Estimate& estimate,double us) {
  if(!std::isfinite(us)||us<=0) return false;
  estimate.us=estimate.samples?std::min(estimate.us,us):us;
  ++estimate.samples;return true;
}
struct Credit { double us = 0; };
struct Sample { Key key; CUcontext context = nullptr; CUevent begin = nullptr, end = nullptr; uint64_t iteration = 0; int phase = 0; };
struct Step {
  uint64_t iteration = 0, launches = 0, runtime_launches = 0, driver_launches = 0, compute = 0, nccl = 0, unknown = 0;
  uint64_t cold = 0, estimated = 0, calibration = 0, capture = 0, attrs_unsupported = 0;
  uint64_t cache_full = 0, stream_full = 0, queue_full = 0, event_error = 0;
  uint64_t delays = 0, delay_error = 0, cap_hits = 0, time_cap = 0;
  uint64_t phase0_calibration_skipped = 0, key_calibration_cap_hits = 0;
  uint64_t phase_calibration_reserved[3] = {0,0,0}, phase_calibration_queued[3] = {0,0,0};
  uint64_t phase_calibration_cap_hits[3] = {0,0,0}, phase_estimated_launches[3] = {0,0,0};
  double requested_us = 0, estimated_compute_us = 0, discarded_credit_us = 0;
  double phase_requested_us[3] = {0,0,0};
  double phase_estimated_compute_us[3] = {0,0,0};
  bool calib = false, fault = false;
};
struct Ticket { bool selected = false; Sample sample; };
std::atomic<uint64_t> host_thread_sequence{0};
thread_local uint64_t host_thread_id = ++host_thread_sequence;

class Actor {
 public:
  std::atomic<bool> active{false};
  std::atomic<int> phase{0};
  std::mutex mutex, boundary;
  std::condition_variable cv;
  std::map<Key, Estimate> estimates;
  std::map<std::pair<uintptr_t,int>, int> classification;
  std::map<StreamKey, Credit> credits;
  std::deque<Sample> pending;
  size_t reserved = 0;
  std::thread worker;
  bool configured = false, closing = false, closed = false, open_step = false;
  bool fault_started = false;
  double ratio = 0;
  int fd = -1;
  Step step;
  Clock::time_point fault_start;
  unsigned fault_iterations = 0;
  uint64_t calibration_queued = 0, calibration_completed = 0, calibration_failed = 0;
  uint64_t phase_calibration_completed[3] = {0,0,0}, phase_calibration_failed[3] = {0,0,0};
  double phase_sample_us_min[3] = {0,0,0}, phase_sample_us_max[3] = {0,0,0}, phase_sample_us_sum[3] = {0,0,0};
  uint64_t pending_peak = 0, unknown_api_requests = 0, unclosed_steps = 0, log_errors = 0;
  ~Actor() = delete;
  void log(const char* text) {
    if (fd < 0) return;
    size_t n = std::strlen(text), offset = 0;
    while (offset < n) {
      ssize_t written = ::write(fd, text + offset, n - offset);
      if (written < 0 && errno == EINTR) continue;
      if (written <= 0) { ++log_errors; return; }
      offset += static_cast<size_t>(written);
    }
  }
  int configure(double value, const char* path) {
    std::lock_guard<std::mutex> lock(boundary);
    if (configured || !std::isfinite(value) || value < 0 || value > 8 || !path || !*path) return -1;
    fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) return -2;
    ratio = value; configured = true;
    char line[1024];
    std::snprintf(line,sizeof(line),"{\"kind\":\"configuration\",\"ratio\":%.6f,\"semantics\":\"extra_delay_over_calibrated_launch_envelope\",\"quantum_us\":20,\"max_single_us\":200,\"max_step_us\":%.0f,\"cache_limit\":4096,\"pending_limit\":256,\"stream_limit\":128,\"calibration_event_limit\":2048,\"calibration_policy\":\"warm2_3;phase1_2_each512;phase0_skip;key_per_step2_total4;finite_positive_min\",\"max_fault_iterations\":%u,\"max_fault_seconds\":30,\"unknown_policy\":\"skip\",\"cold_policy\":\"skip_no_guess\",\"wait\":\"one_block_one_thread_globaltimer_nanosleep\"}\n",ratio,kStepUs,kFaultIterations);
    log(line); return 0;
  }
  void begin(uint64_t iteration, bool calib, bool fault) {
    std::lock_guard<std::mutex> serial(boundary);
    active.store(false);
    if (!configured || closed) return;
    std::lock_guard<std::mutex> lock(mutex);
    if (open_step) { ++unclosed_steps; finish_step_locked(); }
    credits.clear(); step = Step{}; step.iteration = iteration; step.calib = calib && calibration_iteration(iteration); step.fault = fault;
    if (fault) {
      if (!fault_started) { fault_started = true; fault_start = Clock::now(); }
      ++fault_iterations;
      if (fault_iterations > kFaultIterations) { step.fault = false; ++step.cap_hits; }
    }
    open_step = true;
    if ((step.calib || step.fault) && !worker.joinable()) worker = std::thread([this]{ run(); });
    active.store(step.calib || step.fault);
  }
  void finish_step_locked() {
    for (auto& entry : credits) step.discarded_credit_us += entry.second.us;
    char line[2048];
    std::snprintf(line,sizeof(line),"{\"kind\":\"iteration\",\"iteration\":%llu,\"calibration_enabled\":%s,\"fault_path_enabled\":%s,\"launches\":%llu,\"runtime_launches\":%llu,\"driver_launches\":%llu,\"compute\":%llu,\"nccl_excluded\":%llu,\"unknown_excluded\":%llu,\"cold\":%llu,\"estimated\":%llu,\"calibration_queued\":%llu,\"capture_excluded\":%llu,\"attrs_unsupported\":%llu,\"cache_full\":%llu,\"stream_full\":%llu,\"queue_full\":%llu,\"event_error\":%llu,\"delay_launches\":%llu,\"delay_errors\":%llu,\"cap_hits\":%llu,\"time_budget_hits\":%llu,\"requested_delay_us\":%.3f,\"estimated_compute_us\":%.3f,\"discarded_credit_us\":%.3f,\"phase_requested_us\":[%.3f,%.3f,%.3f],\"phase_calibration_reserved\":[%llu,%llu,%llu],\"phase_calibration_queued\":[%llu,%llu,%llu],\"phase_calibration_cap_hits\":[%llu,%llu,%llu],\"key_calibration_cap_hits\":%llu,\"phase0_calibration_skipped\":%llu,\"phase_estimated_launches\":[%llu,%llu,%llu],\"phase_estimated_compute_us\":[%.3f,%.3f,%.3f],\"strength_verified\":false}\n",
      (unsigned long long)step.iteration,step.calib?"true":"false",step.fault?"true":"false",
      (unsigned long long)step.launches,(unsigned long long)step.runtime_launches,(unsigned long long)step.driver_launches,
      (unsigned long long)step.compute,(unsigned long long)step.nccl,(unsigned long long)step.unknown,
      (unsigned long long)step.cold,(unsigned long long)step.estimated,(unsigned long long)step.calibration,
      (unsigned long long)step.capture,(unsigned long long)step.attrs_unsupported,(unsigned long long)step.cache_full,
      (unsigned long long)step.stream_full,(unsigned long long)step.queue_full,(unsigned long long)step.event_error,
      (unsigned long long)step.delays,(unsigned long long)step.delay_error,(unsigned long long)step.cap_hits,(unsigned long long)step.time_cap,
      step.requested_us,step.estimated_compute_us,step.discarded_credit_us,
      step.phase_requested_us[0],step.phase_requested_us[1],step.phase_requested_us[2],
      (unsigned long long)step.phase_calibration_reserved[0],(unsigned long long)step.phase_calibration_reserved[1],(unsigned long long)step.phase_calibration_reserved[2],
      (unsigned long long)step.phase_calibration_queued[0],(unsigned long long)step.phase_calibration_queued[1],(unsigned long long)step.phase_calibration_queued[2],
      (unsigned long long)step.phase_calibration_cap_hits[0],(unsigned long long)step.phase_calibration_cap_hits[1],(unsigned long long)step.phase_calibration_cap_hits[2],
      (unsigned long long)step.key_calibration_cap_hits,(unsigned long long)step.phase0_calibration_skipped,
      (unsigned long long)step.phase_estimated_launches[0],(unsigned long long)step.phase_estimated_launches[1],(unsigned long long)step.phase_estimated_launches[2],
      step.phase_estimated_compute_us[0],step.phase_estimated_compute_us[1],step.phase_estimated_compute_us[2]);
    log(line); open_step = false;
  }
  void end(uint64_t iteration) {
    std::lock_guard<std::mutex> serial(boundary); active.store(false);
    std::lock_guard<std::mutex> lock(mutex);
    if (open_step && step.iteration == iteration) finish_step_locked();
    credits.clear(); phase.store(0);
  }
  int classify(uintptr_t function, int api) {
    std::pair<uintptr_t,int> id{function,api};
    { std::lock_guard<std::mutex> lock(mutex); auto it=classification.find(id); if(it!=classification.end()) return it->second; }
    std::string name, owner;
    if (api == 0) {
      Dl_info info{};
      if (dladdr(reinterpret_cast<void*>(function), &info)) { if(info.dli_sname) name=info.dli_sname; if(info.dli_fname) owner=info.dli_fname; }
      const char* device_name=nullptr;
      if(runtime_name() && runtime_name()(&device_name,reinterpret_cast<const void*>(function))==cudaSuccess && device_name) {
        // Keep host ownership as a second NCCL exclusion signal.
        name=std::string(device_name)+" "+name;
      }
    } else if (cuda().name) {
      const char* result=nullptr;
      if(cuda().name(&result,reinterpret_cast<CUfunction>(function))==CUDA_SUCCESS && result) name=result;
    }
    std::string lower=name+" "+owner;
    std::transform(lower.begin(),lower.end(),lower.begin(),[](unsigned char c){return char(std::tolower(c));});
    int result=0;
    if(lower.find("nccl")!=std::string::npos) result=2;
    else if(!name.empty()) result=1;
    std::lock_guard<std::mutex> lock(mutex);
    if(classification.size()<kCache) classification.emplace(id,result);
    return result;
  }
  Ticket before(uintptr_t function, unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,size_t smem,uint64_t attrs,int api,CUstream stream) {
    Ticket ticket;
    if(!active.load()) return ticket;
    ticket.sample.phase=std::max(0,std::min(2,phase.load()));
    { std::lock_guard<std::mutex> lock(mutex); ++step.launches;if(api==0)++step.runtime_launches;else ++step.driver_launches; }
    if(!cuda().valid()) { std::lock_guard<std::mutex> lock(mutex); ++step.event_error; return ticket; }
    int category=classify(function,api);
    { std::lock_guard<std::mutex> lock(mutex);
      if(category!=1) { if(category==2) ++step.nccl; else ++step.unknown; return ticket; }
      ++step.compute;
      if(step.calib && ticket.sample.phase==0) { ++step.phase0_calibration_skipped;return ticket; }
    }
    CUcontext context=nullptr; CUstreamCaptureStatus capture;
    if(cuda().current(&context)!=CUDA_SUCCESS || !context || cuda().capture(stream,&capture)!=CUDA_SUCCESS) {
      std::lock_guard<std::mutex> lock(mutex); ++step.event_error; return ticket;
    }
    if(capture!=CU_STREAM_CAPTURE_STATUS_NONE) { std::lock_guard<std::mutex> lock(mutex); ++step.capture; return ticket; }
    ticket.sample.key={reinterpret_cast<uintptr_t>(context),function,gx,gy,gz,bx,by,bz,smem,attrs,api}; ticket.sample.context=context;
    std::lock_guard<std::mutex> lock(mutex);
    auto it=estimates.find(ticket.sample.key);
    if(it==estimates.end()) {
      if(estimates.size()>=kCache) { ++step.cache_full; return ticket; }
      it=estimates.emplace(ticket.sample.key,Estimate{}).first;
    }
    if(step.fault && !it->second.samples) ++step.cold;
    ticket.selected=reserve_calibration_locked(it->second,ticket.sample);
    return ticket;
  }
  bool reserve_calibration_locked(Estimate& estimate,Sample& sample) {
    if(!step.calib || !calibration_iteration(step.iteration)) return false;
    if(sample.phase==0) { ++step.phase0_calibration_skipped;return false; }
    if(estimate.warm_iteration!=step.iteration) { estimate.warm_iteration=step.iteration;estimate.warm_attempts=0; }
    if(estimate.samples+estimate.pending>=kSamplesPerKey || estimate.warm_attempts>=kSamplesPerWarmStepKey) { ++step.key_calibration_cap_hits;return false; }
    if(step.phase_calibration_reserved[sample.phase]>=kSamplesPerWarmStepPhase) { ++step.phase_calibration_cap_hits[sample.phase];return false; }
    if(pending.size()+reserved>=kPending || calibration_queued+reserved>=kCalibrationEvents) { ++step.queue_full;return false; }
    ++estimate.pending;++estimate.warm_attempts;++reserved;++step.phase_calibration_reserved[sample.phase];
    sample.iteration=step.iteration;return true;
  }
  void cancel(Ticket& ticket) {
    if(ticket.sample.begin) cuda().destroy(ticket.sample.begin);
    if(ticket.sample.end) cuda().destroy(ticket.sample.end);
    std::lock_guard<std::mutex> lock(mutex);
    if(ticket.selected) { --reserved; --estimates[ticket.sample.key].pending; ++step.event_error; }
    ticket.selected=false;
  }
  void start(Ticket& ticket,CUstream stream) {
    if(!ticket.selected) return;
    if(cuda().create(&ticket.sample.begin,CU_EVENT_DEFAULT)!=CUDA_SUCCESS ||
       cuda().create(&ticket.sample.end,CU_EVENT_DEFAULT)!=CUDA_SUCCESS ||
       cuda().record(ticket.sample.begin,stream)!=CUDA_SUCCESS) cancel(ticket);
  }
  void after(Ticket& ticket,CUstream stream,bool success) {
    if(ticket.selected) {
      if(!success || cuda().record(ticket.sample.end,stream)!=CUDA_SUCCESS) cancel(ticket);
      else {
        std::lock_guard<std::mutex> lock(mutex);
        --reserved; pending.push_back(std::move(ticket.sample)); ticket.selected=false;
        ++calibration_queued; ++step.calibration;++step.phase_calibration_queued[ticket.sample.phase];
        pending_peak=std::max<uint64_t>(pending_peak,pending.size()+reserved); cv.notify_one();
      }
    }
    if(!success || !active.load() || !std::get<0>(ticket.sample.key)) return;
    double delay=0;
    StreamKey stream_key{std::get<0>(ticket.sample.key),reinterpret_cast<uintptr_t>(stream),stream==CU_STREAM_PER_THREAD?host_thread_id:0};
    {
      std::lock_guard<std::mutex> lock(mutex);
      if(!step.fault) return;
      if(std::chrono::duration<double>(Clock::now()-fault_start).count()>=kFaultSeconds) { ++step.time_cap; return; }
      auto it=estimates.find(ticket.sample.key);
      if(it==estimates.end() || !it->second.samples) return;
      ++step.estimated; step.estimated_compute_us+=it->second.us;
      ++step.phase_estimated_launches[ticket.sample.phase];step.phase_estimated_compute_us[ticket.sample.phase]+=it->second.us;
      auto c=credits.find(stream_key);
      if(c==credits.end()) { if(credits.size()>=kStreams) { ++step.stream_full; return; } c=credits.emplace(stream_key,Credit{}).first; }
      c->second.us=std::min(kStepUs,c->second.us+ratio*it->second.us);
      if(c->second.us<kQuantumUs) return;
      double remaining=kStepUs-step.requested_us;
      if(remaining<kQuantumUs) { ++step.cap_hits; return; }
      delay=std::min({std::floor(c->second.us),kSingleUs,remaining});
      c->second.us-=delay; step.requested_us+=delay; // reserve process-wide step cap
    }
    auto result=flash_v4_delay_launch_wait(reinterpret_cast<cudaStream_t>(stream),static_cast<unsigned long long>(delay*1000));
    std::lock_guard<std::mutex> lock(mutex);
    if(result==cudaSuccess) { ++step.delays; step.phase_requested_us[ticket.sample.phase]+=delay; }
    else { ++step.delay_error; step.requested_us-=delay; credits[stream_key].us+=delay; }
  }
  void run() {
    internal=true;
    Clock::time_point draining_start; bool draining=false;
    std::unique_lock<std::mutex> lock(mutex);
    for(;;) {
      if(closing&&!draining) { draining=true; draining_start=Clock::now(); }
      if(closing&&pending.empty()) break;
      size_t count=pending.size();
      for(size_t i=0;i<count;++i) {
        Sample sample=std::move(pending.front());pending.pop_front();++reserved;lock.unlock();
        bool pushed=cuda().push(sample.context)==CUDA_SUCCESS;
        CUresult result=pushed?cuda().query(sample.end):CUDA_ERROR_INVALID_CONTEXT;
        bool timeout=draining&&std::chrono::duration<double>(Clock::now()-draining_start).count()>3;
        if(result==CUDA_ERROR_NOT_READY&&!timeout) {
          if(pushed) { CUcontext old;cuda().pop(&old); }
          lock.lock();--reserved;pending.push_back(std::move(sample));continue;
        }
        float ms=0;
        bool good=result==CUDA_SUCCESS&&cuda().elapsed(&ms,sample.begin,sample.end)==CUDA_SUCCESS;
        if(pushed) { cuda().destroy(sample.begin);cuda().destroy(sample.end);CUcontext old;cuda().pop(&old); }
        lock.lock();--reserved;auto& estimate=estimates[sample.key];--estimate.pending;
        good=good&&accept_estimate_sample(estimate,ms*1000.0);
        if(good) {
          ++calibration_completed;++phase_calibration_completed[sample.phase];double us=ms*1000.0;
          auto& minimum=phase_sample_us_min[sample.phase];minimum=minimum?std::min(minimum,us):us;
          phase_sample_us_max[sample.phase]=std::max(phase_sample_us_max[sample.phase],us);phase_sample_us_sum[sample.phase]+=us;
        } else { ++calibration_failed;++phase_calibration_failed[sample.phase]; }
      }
      cv.wait_for(lock,std::chrono::milliseconds(2));
    }
  }
  void flush() {
    std::lock_guard<std::mutex> serial(boundary);active.store(false);
    if(!configured||closed) return;
    { std::lock_guard<std::mutex> lock(mutex);if(open_step) { ++unclosed_steps;finish_step_locked(); } closing=true;cv.notify_one(); }
    if(worker.joinable()) worker.join();
    char line[2048];
    std::snprintf(line,sizeof(line),"{\"kind\":\"summary\",\"cache_entries\":%zu,\"calibration_queued\":%llu,\"calibration_completed\":%llu,\"calibration_failed\":%llu,\"pending_peak\":%llu,\"unsupported_api_requests\":%llu,\"unclosed_steps\":%llu,\"log_errors\":%llu,\"coverage\":\"ordinary_launches_only;named_non_nccl;no_graph_or_cooperative_coverage\",\"estimate\":\"warm2_3_same_function_launchshape_finite_positive_min;async_event_envelope\",\"phase_calibration_completed\":[%llu,%llu,%llu],\"phase_calibration_failed\":[%llu,%llu,%llu],\"phase_sample_us_min\":[%.3f,%.3f,%.3f],\"phase_sample_us_max\":[%.3f,%.3f,%.3f],\"phase_sample_us_sum\":[%.3f,%.3f,%.3f],\"strength_verified\":false}\n",
      estimates.size(),(unsigned long long)calibration_queued,(unsigned long long)calibration_completed,(unsigned long long)calibration_failed,
      (unsigned long long)pending_peak,(unsigned long long)unknown_api_requests,(unsigned long long)unclosed_steps,(unsigned long long)log_errors,
      (unsigned long long)phase_calibration_completed[0],(unsigned long long)phase_calibration_completed[1],(unsigned long long)phase_calibration_completed[2],
      (unsigned long long)phase_calibration_failed[0],(unsigned long long)phase_calibration_failed[1],(unsigned long long)phase_calibration_failed[2],
      phase_sample_us_min[0],phase_sample_us_min[1],phase_sample_us_min[2],
      phase_sample_us_max[0],phase_sample_us_max[1],phase_sample_us_max[2],
      phase_sample_us_sum[0],phase_sample_us_sum[1],phase_sample_us_sum[2]);
    log(line);close(fd);fd=-1;closed=true;
  }
};
Actor& actor() { static Actor* instance=new Actor;return *instance; }
// Every outer launch owns the guard across real launch and injected wait; a
// runtime->driver call and the actor's own launch are therefore forwarded once.
template<class F> auto launch(uintptr_t function,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,size_t smem,uint64_t attrs,int api,CUstream stream,F call) -> decltype(call()) {
  if(internal||!actor().active.load()) return call();
  Guard guard;
  Ticket ticket=actor().before(function,gx,gy,gz,bx,by,bz,smem,attrs,api,stream);
  actor().start(ticket,stream);auto result=call();actor().after(ticket,stream,result==0);return result;
}
template<class F> auto unsupported_attrs(F call) -> decltype(call()) {
  if(internal||!actor().active.load()) return call();
  Guard guard;
  { std::lock_guard<std::mutex> lock(actor().mutex);++actor().step.launches;++actor().step.attrs_unsupported; }
  return call();
}
} // namespace

extern "C" int flash_v4_delay_configure(double ratio,const char* logpath) { return actor().configure(ratio,logpath); }
extern "C" void flash_v4_delay_begin(uint64_t iteration,int calib,int fault) { actor().begin(iteration,calib!=0,fault!=0); }
extern "C" void flash_v4_delay_phase(int phase) { actor().phase.store(std::max(0,std::min(2,phase))); }
extern "C" void flash_v4_delay_end(uint64_t iteration) { actor().end(iteration); }
extern "C" void flash_v4_delay_flush() { actor().flush(); }

extern "C" cudaError_t cudaLaunchKernel(const void* function,dim3 grid,dim3 block,void** args,size_t smem,cudaStream_t stream) {
  using Fn=cudaError_t (*)(const void*,dim3,dim3,void**,size_t,cudaStream_t);
  static auto fn=actual<Fn>("cudaLaunchKernel",true);
  return launch(reinterpret_cast<uintptr_t>(function),grid.x,grid.y,grid.z,block.x,block.y,block.z,smem,0,0,reinterpret_cast<CUstream>(stream),[&]{return fn(function,grid,block,args,smem,stream);});
}
extern "C" cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config,const void* function,void** args) {
  using Fn=cudaError_t (*)(const cudaLaunchConfig_t*,const void*,void**);
  static auto fn=actual<Fn>("cudaLaunchKernelExC",true);
  if(!config||config->numAttrs>16||(config->numAttrs&&!config->attrs)) return unsupported_attrs([&]{return fn(config,function,args);});
  return launch(reinterpret_cast<uintptr_t>(function),config->gridDim.x,config->gridDim.y,config->gridDim.z,config->blockDim.x,config->blockDim.y,config->blockDim.z,config->dynamicSmemBytes,attrs_hash(config->attrs,config->numAttrs),0,reinterpret_cast<CUstream>(config->stream),[&]{return fn(config,function,args);});
}

using DriverLaunch=CUresult (*)(CUfunction,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,CUstream,void**,void**);
using DriverLaunchEx=CUresult (*)(const CUlaunchConfig*,CUfunction,void**,void**);
extern "C" CUresult cuLaunchKernel(CUfunction function,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned smem,CUstream stream,void** args,void** extra) {
  static auto fn=actual<DriverLaunch>("cuLaunchKernel",false);
  return launch(reinterpret_cast<uintptr_t>(function),gx,gy,gz,bx,by,bz,smem,0,1,stream,[&]{return fn(function,gx,gy,gz,bx,by,bz,smem,stream,args,extra);});
}
extern "C" CUresult cuLaunchKernelEx(const CUlaunchConfig* config,CUfunction function,void** args,void** extra) {
  static auto fn=actual<DriverLaunchEx>("cuLaunchKernelEx",false);
  if(!config||config->numAttrs>16||(config->numAttrs&&!config->attrs)) return unsupported_attrs([&]{return fn(config,function,args,extra);});
  return launch(reinterpret_cast<uintptr_t>(function),config->gridDimX,config->gridDimY,config->gridDimZ,config->blockDimX,config->blockDimY,config->blockDimZ,config->sharedMemBytes,attrs_hash(config->attrs,config->numAttrs),1,config->hStream,[&]{return fn(config,function,args,extra);});
}
// PTDS lookup must preserve the caller's default-stream semantics. The special
// stream handle is passed explicitly to timing and to the injected wait.
extern "C" CUresult cuLaunchKernel_ptsz(CUfunction function,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned smem,CUstream stream,void** args,void** extra) {
  static auto fn=actual<DriverLaunch>("cuLaunchKernel_ptsz",false);
  CUstream observed=stream?stream:CU_STREAM_PER_THREAD;
  return launch(reinterpret_cast<uintptr_t>(function),gx,gy,gz,bx,by,bz,smem,0,1,observed,[&]{return fn(function,gx,gy,gz,bx,by,bz,smem,stream,args,extra);});
}
extern "C" CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig* config,CUfunction function,void** args,void** extra) {
  static auto fn=actual<DriverLaunchEx>("cuLaunchKernelEx_ptsz",false);
  if(!config||config->numAttrs>16||(config->numAttrs&&!config->attrs)) return unsupported_attrs([&]{return fn(config,function,args,extra);});
  CUstream observed=config->hStream?config->hStream:CU_STREAM_PER_THREAD;
  return launch(reinterpret_cast<uintptr_t>(function),config->gridDimX,config->gridDimY,config->gridDimZ,config->blockDimX,config->blockDimY,config->blockDimZ,config->sharedMemBytes,attrs_hash(config->attrs,config->numAttrs),1,observed,[&]{return fn(config,function,args,extra);});
}
namespace {
void redirect(const char* symbol,void** function,cuuint64_t flags) {
  if(!symbol||!function||!*function) return;
  bool ptsz=(flags&CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)!=0;
  if(std::strcmp(symbol,"cuLaunchKernel")==0) *function=ptsz?reinterpret_cast<void*>(&cuLaunchKernel_ptsz):reinterpret_cast<void*>(&cuLaunchKernel);
  else if(std::strcmp(symbol,"cuLaunchKernelEx")==0) *function=ptsz?reinterpret_cast<void*>(&cuLaunchKernelEx_ptsz):reinterpret_cast<void*>(&cuLaunchKernelEx);
  else if(std::strncmp(symbol,"cuLaunch",8)==0) { std::lock_guard<std::mutex> lock(actor().mutex);++actor().unknown_api_requests; }
}
}
extern "C" CUresult cuGetProcAddress(const char* symbol,void** function,int version,cuuint64_t flags) {
  using Fn=CUresult (*)(const char*,void**,int,cuuint64_t);
  static auto fn=actual<Fn>("cuGetProcAddress",false);
  auto result=fn(symbol,function,version,flags);if(result==CUDA_SUCCESS) redirect(symbol,function,flags);return result;
}
extern "C" CUresult cuGetProcAddress_v2(const char* symbol,void** function,int version,cuuint64_t flags,CUdriverProcAddressQueryResult* status) {
  using Fn=CUresult (*)(const char*,void**,int,cuuint64_t,CUdriverProcAddressQueryResult*);
  static auto fn=actual<Fn>("cuGetProcAddress_v2",false);
  auto result=fn(symbol,function,version,flags,status);if(result==CUDA_SUCCESS) redirect(symbol,function,flags);return result;
}
