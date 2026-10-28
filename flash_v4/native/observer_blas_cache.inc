// Observer-owned front cache, accessed only while holding shape_mu. Map node
// addresses remain stable until the iteration boundary clears both caches.
struct BlasSamplingFrontCache {
  struct Entry { BlasKey key{}; BlasSampling* value = nullptr; };
  static constexpr size_t kCapacity = 512;
  Entry entries[kCapacity]{};
  static size_t index(const BlasKey& key) {
    uint64_t hash = 14695981039346656037ULL;
    std::apply([&](auto... values) {
      ((hash = (hash ^ static_cast<uint64_t>(values)) * 1099511628211ULL), ...);
    }, key);
    return (hash ^ (hash >> 32) ^ (hash >> 16)) & (kCapacity - 1);
  }
  BlasSampling* get(const BlasKey& key) const {
    const auto& entry = entries[index(key)];
    return entry.value && entry.key == key ? entry.value : nullptr;
  }
  void put(const BlasKey& key, BlasSampling* value) {
    entries[index(key)] = Entry{key, value};
  }
  void clear() { for (auto& entry : entries) entry.value = nullptr; }
};
