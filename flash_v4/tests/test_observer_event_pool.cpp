#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;

struct FakeEvent {
  int device;
  int id;
};
using cudaEvent_t = FakeEvent*;

int current_device = 0;
int next_event_id = 1;
std::vector<int> destroyed_ids;
std::vector<int> devices_selected;

cudaError_t cudaGetDevice(int* device) {
  *device = current_device;
  return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
  current_device = device;
  devices_selected.push_back(device);
  return cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
  if (!event || event->device != current_device) return 1;
  destroyed_ids.push_back(event->id);
  delete event;
  return cudaSuccess;
}

struct Row { int sequence = 0; };
struct Timing {
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  int device = 0;
  std::vector<Row> rows;
};

struct EventPoolHarness {
  static constexpr std::size_t kPending = 2;
  static constexpr std::size_t kGroupRows = 64;
#include "../native/observer_event_pool.inc"
};

Timing make_timing(int device) {
  Timing timing;
  timing.device = device;
  timing.begin = new FakeEvent{device, next_event_id++};
  timing.end = new FakeEvent{device, next_event_id++};
  return timing;
}

void reset_fake_cuda() {
  current_device = 0;
  destroyed_ids.clear();
  devices_selected.clear();
}

void test_same_device_reuse() {
  reset_fake_cuda();
  EventPoolHarness pool;
  Timing cached = make_timing(2);
  assert(pool.cache_completed(cached));
  assert(cached.begin == nullptr && cached.end == nullptr);

  Timing requested;
  requested.device = 2;
  assert(pool.take_cached(requested));
  assert(requested.begin && requested.begin->device == 2);
  assert(requested.end && requested.end->device == 2);
  assert(pool.event_pairs_reused.load() == 1);

  assert(cudaSetDevice(2) == cudaSuccess);
  assert(cudaEventDestroy(requested.begin) == cudaSuccess);
  assert(cudaEventDestroy(requested.end) == cudaSuccess);
}

void test_device_isolation() {
  reset_fake_cuda();
  EventPoolHarness pool;
  Timing device_two = make_timing(2);
  Timing device_five = make_timing(5);
  assert(pool.cache_completed(device_two));
  assert(pool.cache_completed(device_five));

  Timing requested;
  requested.device = 5;
  assert(pool.take_cached(requested));
  assert(requested.begin->device == 5 && requested.end->device == 5);
  // A Timing that already owns events must not be overwritten or leaked.
  assert(!pool.take_cached(requested));
  Timing requested_two;
  requested_two.device = 2;
  assert(pool.take_cached(requested_two));
  assert(requested_two.begin->device == 2 && requested_two.end->device == 2);
  assert(cudaSetDevice(5) == cudaSuccess);
  assert(cudaEventDestroy(requested.begin) == cudaSuccess);
  assert(cudaEventDestroy(requested.end) == cudaSuccess);
  assert(cudaSetDevice(2) == cudaSuccess);
  assert(cudaEventDestroy(requested_two.begin) == cudaSuccess);
  assert(cudaEventDestroy(requested_two.end) == cudaSuccess);
  assert(pool.clear_cached());
}

void test_capacity_and_ownership() {
  reset_fake_cuda();
  EventPoolHarness pool;
  Timing first = make_timing(3);
  Timing second = make_timing(3);
  Timing overflow = make_timing(3);
  assert(pool.cache_completed(first));
  assert(pool.cache_completed(second));
  assert(!pool.cache_completed(overflow));
  assert(overflow.begin && overflow.end);
  assert(pool.event_pool.at(3).size() == EventPoolHarness::kPending);
  assert(pool.event_pairs_cached.load() == 2);
  assert(pool.event_pool_high_water.load() == EventPoolHarness::kPending);

  assert(cudaSetDevice(3) == cudaSuccess);
  assert(cudaEventDestroy(overflow.begin) == cudaSuccess);
  assert(cudaEventDestroy(overflow.end) == cudaSuccess);
  current_device = 9;
  assert(pool.clear_cached());
  assert(current_device == 9);
  assert(pool.event_pool.empty());
  assert(destroyed_ids.size() == 6);
}

void test_empty_storage_reuse() {
  reset_fake_cuda();
  EventPoolHarness pool;
  Timing timing = make_timing(2);
  timing.rows.reserve(64);
  auto* storage = timing.rows.data();
  for (int iteration = 0; iteration < 2000; ++iteration) {
    for (int i = 0; i < 64; ++i) timing.rows.push_back(Row{iteration * 64 + i});
    assert(pool.cache_completed(timing));
    assert(timing.rows.empty());
    Timing next;
    next.device = 2;
    assert(pool.take_cached(next));
    assert(next.rows.empty() && next.rows.data() == storage);
    assert(pool.event_pool.count(2) == 1 && pool.event_pool.at(2).empty());
    // Model the NCCL group transfer without allocating or retaining old rows.
    std::vector<Row> group_rows;
    group_rows.swap(next.rows);
    group_rows.push_back(Row{iteration});
    next.rows = std::move(group_rows);
    assert(next.rows.size() == 1 && next.rows[0].sequence == iteration);
    next.rows.clear();
    timing = std::move(next);
  }
  assert(pool.cache_completed(timing));
  assert(pool.clear_cached());
  assert(pool.event_pool.empty() && destroyed_ids.size() == 2);
}

void test_storage_ownership_and_bound() {
  reset_fake_cuda();
  EventPoolHarness pool;
  Timing first = make_timing(3), second = make_timing(3), overflow = make_timing(3);
  first.rows.push_back(Row{11}); second.rows.push_back(Row{22}); overflow.rows.push_back(Row{33});
  auto* overflow_storage = overflow.rows.data();
  assert(pool.cache_completed(first) && pool.cache_completed(second));
  assert(!pool.cache_completed(overflow));
  assert(overflow.rows.data() == overflow_storage && overflow.rows[0].sequence == 33);
  Timing occupied; occupied.device = 3; occupied.rows.push_back(Row{44});
  assert(!pool.take_cached(occupied));
  assert(occupied.rows[0].sequence == 44 && pool.event_pool.at(3).size() == 2);
  assert(cudaSetDevice(3) == cudaSuccess);
  assert(cudaEventDestroy(overflow.begin) == cudaSuccess && cudaEventDestroy(overflow.end) == cudaSuccess);
  assert(pool.clear_cached());
  Timing oversized = make_timing(4); oversized.rows.reserve(128); oversized.rows.push_back(Row{55});
  assert(pool.cache_completed(oversized));
  assert(pool.event_pool.at(4).back().rows.capacity() == 0);
  Timing requested; requested.device = 4;
  assert(pool.take_cached(requested));
  assert(requested.rows.empty() && requested.rows.capacity() == 0);
  assert(pool.cache_completed(requested) && pool.clear_cached());
}

int main() {
  test_same_device_reuse();
  test_device_isolation();
  test_capacity_and_ownership();
  test_empty_storage_reuse();
  test_storage_ownership_and_bound();
  std::cout << "event pool CPU test passed\n";
  return 0;
}
