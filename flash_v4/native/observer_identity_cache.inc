// Bounded per-launch-thread cache. Probe collisions before falling back to the
// authoritative shared cache; do not evict alternating hot launch functions.
struct KernelIdentityFrontCache {
  static constexpr size_t kCapacity = 4096, kProbes = 16;
  struct Entry {
    uintptr_t function = 0;
    bool runtime = false, valid = false;
    KernelIdentity identity;
  };
  Entry entries[kCapacity]{};
  static size_t index(uintptr_t function, bool runtime) {
    return ((function >> 4) ^ (function >> 12) ^ size_t(runtime)) & (kCapacity - 1);
  }
  bool get(uintptr_t function, bool runtime, KernelIdentity& identity) const {
    const size_t first = index(function, runtime);
    for (size_t probe = 0; probe < kProbes; ++probe) {
      const auto& entry = entries[(first + probe) & (kCapacity - 1)];
      if (!entry.valid) return false;
      if (entry.function == function && entry.runtime == runtime) {
        identity = entry.identity;
        return true;
      }
    }
    return false;
  }
  void put(uintptr_t function, bool runtime, KernelIdentity identity) {
    const size_t first = index(function, runtime);
    for (size_t probe = 0; probe < kProbes; ++probe) {
      auto& entry = entries[(first + probe) & (kCapacity - 1)];
      if (!entry.valid || (entry.function == function && entry.runtime == runtime)) {
        entry = Entry{function, runtime, true, identity};
        return;
      }
    }
    // A full probe set only loses a speedup; the caller still returns the
    // authoritative shared result and retries that path on the next launch.
  }
};
