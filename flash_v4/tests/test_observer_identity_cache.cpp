#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

struct KernelIdentity { uint64_t hash = 0; int category = 0; };
#include "../native/observer_identity_cache.inc"

int main() {
  KernelIdentityFrontCache cache;
  KernelIdentity value;

  assert(!cache.get(0, false, value));
  cache.put(0, false, {17, 1});
  assert(cache.get(0, false, value) && value.hash == 17 && value.category == 1);
  assert(!cache.get(0, true, value));
  cache.put(0, true, {23, 2});
  assert(cache.get(0, true, value) && value.hash == 23 && value.category == 2);
  assert(cache.get(0, false, value) && value.hash == 17);

  KernelIdentityFrontCache collision_cache;
  std::vector<uintptr_t> collisions;
  for (uintptr_t function = 16;
       collisions.size() < KernelIdentityFrontCache::kProbes + 1;
       function += 16) {
    if (KernelIdentityFrontCache::index(function, false) ==
        KernelIdentityFrontCache::index(16, false)) {
      collisions.push_back(function);
    }
  }

  for (size_t i = 0; i < KernelIdentityFrontCache::kProbes; ++i) {
    collision_cache.put(collisions[i], false, {100 + i, int(i % 3)});
  }
  for (size_t i = 0; i < KernelIdentityFrontCache::kProbes; ++i) {
    assert(collision_cache.get(collisions[i], false, value));
    assert(value.hash == 100 + i && value.category == int(i % 3));
  }

  // The seventeenth key in a full probe set is not cached; existing entries stay.
  collision_cache.put(collisions.back(), false, {999, 1});
  assert(!collision_cache.get(collisions.back(), false, value));
  assert(collision_cache.get(collisions[0], false, value) && value.hash == 100);

  // Updating an existing key replaces its value without evicting its neighbors.
  collision_cache.put(collisions[0], false, {777, 2});
  assert(collision_cache.get(collisions[0], false, value));
  assert(value.hash == 777 && value.category == 2);
  for (size_t i = 1; i < KernelIdentityFrontCache::kProbes; ++i) {
    assert(collision_cache.get(collisions[i], false, value));
    assert(value.hash == 100 + i && value.category == int(i % 3));
  }

  std::cout << "identity cache CPU test passed\n";
}
