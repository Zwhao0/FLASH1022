// CPU-only snapshot of GREYHOUND's producer ring under its existing mutex.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include "shm_storage.hpp"

namespace bip = boost::interprocess;
namespace {
constexpr size_t fields = 14;
constexpr size_t header_words = METADATA_FIELDS;
static_assert(Record::numFields() == fields, "snapshot record ABI changed");
}

extern "C" int64_t flash_snapshot(const char* record_name, const char* lock_name,
                                 int64_t last_event, uint64_t* out,
                                 size_t out_capacity_records, uint64_t* missed) {
    if (missed) *missed = 0;
    if (!record_name || !lock_name || !missed || last_event < -1) return -2;
    try {
        bip::shared_memory_object records(bip::open_only, record_name, bip::read_only);
        bip::offset_t record_bytes = 0;
        if (!records.get_size(record_bytes)) return -3;
        if (record_bytes < static_cast<bip::offset_t>(header_words * sizeof(uint64_t))) return -1;
        bip::mapped_region record_map(records, bip::read_only);
        const auto* header = static_cast<const uint64_t*>(record_map.get_address());
        // Producer constructs Lock before publishing header[0]/[1]. Do not
        // touch a possibly unconstructed mutex while a new ring starts up.
        const uint64_t initial_fields = __atomic_load_n(header, __ATOMIC_ACQUIRE);
        const uint64_t initial_capacity = __atomic_load_n(header + 1, __ATOMIC_ACQUIRE);
        const uint64_t initial_magic = __atomic_load_n(header + 5, __ATOMIC_ACQUIRE);
        if (initial_magic != BUFFER_MAGIC || !initial_fields || !initial_capacity) return -1;
        if (initial_fields != fields) return -2;
        const size_t available_words = record_map.get_size() / sizeof(uint64_t);
        if (initial_capacity > (available_words - header_words) / fields) return -2;

        bip::shared_memory_object locks(bip::open_only, lock_name, bip::read_write);
        bip::offset_t lock_bytes = 0;
        if (!locks.get_size(lock_bytes)) return -3;
        if (lock_bytes < static_cast<bip::offset_t>(sizeof(Lock))) return -1;
        bip::mapped_region lock_map(locks, bip::read_write);
        auto* lock = static_cast<Lock*>(lock_map.get_address());
        // Only metadata and fresh rows are read while holding the producer's
        // mutex. No retries, full-ring copies, allocations or mapping here.
        bip::scoped_lock<bip::interprocess_mutex> guard(lock->mutex);
        const uint64_t capacity = header[1], size = header[2], head = header[3];
        const uint64_t next_event = header[6];
        if (header[0] != fields || header[5] != BUFFER_MAGIC || !capacity ||
            capacity != initial_capacity || size > capacity || head >= capacity ||
            header[4] >= capacity || next_event < size ||
            next_event > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) return -2;
        const uint64_t first_available = next_event - size;
        const uint64_t requested = static_cast<uint64_t>(last_event) + 1;
        const uint64_t from = std::max(requested, first_available);
        const uint64_t fresh = from < next_event ? next_event - from : 0;
        if (fresh > out_capacity_records || (fresh && !out)) return -2;
        if (fresh) {
            const uint64_t offset = (head + from - first_available) % capacity;
            const uint64_t first_part = std::min(fresh, capacity - offset);
            const auto* data = header + header_words;
            std::memcpy(out, data + offset * fields, first_part * fields * sizeof(uint64_t));
            if (fresh > first_part)
                std::memcpy(out + first_part * fields, data,
                            (fresh - first_part) * fields * sizeof(uint64_t));
        }
        *missed = first_available > requested ? first_available - requested : 0;
        return static_cast<int64_t>(fresh);
    } catch (const bip::interprocess_exception& error) {
        return error.get_error_code() == bip::not_found_error ? -1 : -3;
    } catch (...) {
        return -3;
    }
}
