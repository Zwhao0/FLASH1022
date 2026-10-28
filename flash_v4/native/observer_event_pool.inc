// Observer member fragment: completed CUDA event pairs, kept per device.
struct CachedEventPair {
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  // Empty storage only: completed rows have already been emitted by the worker.
  std::vector<Row> rows;
};
std::mutex event_pool_mu;
std::unordered_map<int, std::vector<CachedEventPair>> event_pool;
std::atomic<std::uint64_t> event_pairs_reused{0}, event_pairs_cached{0};
std::atomic<std::size_t> event_pool_high_water{0};  // Maximum cached for one device.

bool take_cached(Timing& timing) {
  if (timing.begin || timing.end || !timing.rows.empty()) return false;
  std::lock_guard<std::mutex> lock(event_pool_mu);
  auto found = event_pool.find(timing.device);
  if (found == event_pool.end()) return false;
  auto& pairs = found->second;
  std::size_t selected = pairs.size();
  for (std::size_t i = pairs.size(); i > 0; --i) {
    if (pairs[i - 1].begin && pairs[i - 1].end) {
      selected = i - 1;
      break;
    }
  }
  if (selected == pairs.size()) return false;
  CachedEventPair pair = std::move(pairs[selected]);
  pairs.erase(pairs.begin() + selected);
  timing.begin = pair.begin;
  timing.end = pair.end;
  timing.rows.swap(pair.rows);
  event_pairs_reused.fetch_add(1, std::memory_order_relaxed);
  // Retain the empty per-device bucket and its vector allocation until flush.
  return true;
}

// Call only after end query and elapsed-time retrieval both succeeded.
bool cache_completed(Timing& timing) {
  if (!timing.begin || !timing.end) return false;
  std::lock_guard<std::mutex> lock(event_pool_mu);
  auto& pairs = event_pool[timing.device];
  if (pairs.size() >= kPending) return false;
  pairs.push_back(CachedEventPair{timing.begin, timing.end, {}});
  timing.rows.clear();
  // Bound retained host storage as well as the number of CUDA event pairs.
  if (timing.rows.capacity() <= kGroupRows) pairs.back().rows.swap(timing.rows);
  timing.begin = nullptr;
  timing.end = nullptr;
  event_pairs_cached.fetch_add(1, std::memory_order_relaxed);
  std::size_t current = pairs.size();
  std::size_t high = event_pool_high_water.load(std::memory_order_relaxed);
  while (high < current && !event_pool_high_water.compare_exchange_weak(
             high, current, std::memory_order_relaxed, std::memory_order_relaxed)) {}
  return true;
}

// Call after the event worker has joined. Failed destroys remain available for
// a later cleanup attempt; the caller's current device is restored on exit.
bool clear_cached() {
  int original_device = -1;
  if (cudaGetDevice(&original_device) != cudaSuccess) return false;
  bool complete = true;
  std::lock_guard<std::mutex> lock(event_pool_mu);
  for (auto it = event_pool.begin(); it != event_pool.end();) {
    if (it->second.empty()) {
      it = event_pool.erase(it);
      continue;
    }
    if (cudaSetDevice(it->first) != cudaSuccess) {
      complete = false;
      ++it;
      continue;
    }
    auto& pairs = it->second;
    for (auto pair = pairs.begin(); pair != pairs.end();) {
      if (pair->begin && cudaEventDestroy(pair->begin) == cudaSuccess) pair->begin = nullptr;
      else if (pair->begin) complete = false;
      if (pair->end && cudaEventDestroy(pair->end) == cudaSuccess) pair->end = nullptr;
      else if (pair->end) complete = false;
      if (!pair->begin && !pair->end) pair = pairs.erase(pair);
      else ++pair;
    }
    if (pairs.empty()) it = event_pool.erase(it);
    else ++it;
  }
  if (cudaSetDevice(original_device) != cudaSuccess) complete = false;
  return complete && event_pool.empty();
}
