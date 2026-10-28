#include <cassert>
#include <cstdint>
#include <cstddef>
#include <map>
#include <tuple>
#include <vector>
struct KernelIdentity { uint64_t hash = 0; int category = 0; };
#include "../native/observer_identity_cache.inc"
using BlasKey = std::tuple<int,uint64_t,uint64_t,uint64_t,int,int,int,int,int,int,int>;
struct BlasSampling { uint64_t ordinal = 0; unsigned offset = 0; };
#include "../native/observer_blas_cache.inc"

int main() {
  KernelIdentityFrontCache kernels;
  std::vector<uintptr_t> collisions;
  for (uintptr_t p = 16; collisions.size() < kernels.kProbes + 1; p += 16)
    if (kernels.index(p, false) == kernels.index(16, false)) collisions.push_back(p);
  KernelIdentity value;
  for (size_t i = 0; i < kernels.kProbes; ++i)
    kernels.put(collisions[i], false, KernelIdentity{100 + i, int(i % 3)});
  kernels.put(collisions.back(), false, KernelIdentity{999, 1});
  assert(!kernels.get(collisions.back(), false, value));
  for (size_t i = 0; i < kernels.kProbes; ++i) {
    assert(kernels.get(collisions[i], false, value));
    assert(value.hash == 100 + i && value.category == int(i % 3));
  }
  assert(!kernels.get(collisions[0], true, value));
  kernels.put(collisions[0], false, KernelIdentity{0, 0});
  assert(kernels.get(collisions[0], false, value) && value.hash == 0 && value.category == 0);

  BlasSamplingFrontCache front;
  std::map<BlasKey, BlasSampling> actual, reference;
  for (unsigned iteration = 0; iteration < 8; ++iteration) {
    front.clear(); actual.clear(); reference.clear();
    for (unsigned i = 0; i < 12000; ++i) {
      // More keys than front-cache entries exercise exact-key fallbacks.
      unsigned id = (i * 37) % 900;
      BlasKey key{int(id % 2), id * 8 + 8, 1024, 2048,
                  int(id % 3), -1, -1, -1, 0, int(id % 4), 1};
      auto* found = front.get(key);
      auto inserted = actual.emplace(key, BlasSampling{0, (id + iteration) % 16});
      if (!found) { found = &inserted.first->second; front.put(key, found); }
      assert(found == &inserted.first->second);
      auto ref = reference.emplace(key, BlasSampling{0, (id + iteration) % 16}).first;
      assert(found->ordinal == ref->second.ordinal && found->offset == ref->second.offset);
      assert((found->ordinal % 16 == found->offset) ==
             (ref->second.ordinal % 16 == ref->second.offset));
      ++found->ordinal; ++ref->second.ordinal;
    }
    front.clear();
    for (const auto& item : actual) assert(front.get(item.first) == nullptr);
  }
}
