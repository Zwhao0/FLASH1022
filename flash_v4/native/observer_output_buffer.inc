#pragma once

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <sys/types.h>

namespace flash_v4 {
namespace detail {

// Fixed-size output staging for observer rows. The caller owns synchronization
// and byte-budget accounting, and must call flush() during shutdown.
struct ObserverOutputBuffer {
 public:
  static constexpr std::size_t kCapacity = 16 * 1024;
  static constexpr std::size_t kMaxLine = 1024;
  // Bound pathological writers that return EINTR forever.
  static constexpr unsigned kMaxConsecutiveEintr = 16;

  ObserverOutputBuffer() = default;
  ObserverOutputBuffer(const ObserverOutputBuffer&) = delete;
  ObserverOutputBuffer& operator=(const ObserverOutputBuffer&) = delete;

  std::size_t buffered_bytes() const { return used_; }

  template <typename Writer>
  bool append(const char* line, std::size_t length, Writer&& writer) {
    if (length > kMaxLine || (length != 0 && line == nullptr)) return false;
    if (used_ + length > kCapacity && !flush(writer)) return false;
    if (used_ + length > kCapacity) return false;
    if (length != 0) std::memcpy(data_ + used_, line, length);
    used_ += length;
    return true;
  }

  // Drains all currently buffered bytes. Positive partial writes advance the
  // cursor; EINTR is retried with a finite consecutive-retry limit. On error,
  // the unwritten suffix remains buffered so the caller can report or retry.
  template <typename Writer>
  bool flush(Writer&& writer) {
    std::size_t offset = 0;
    unsigned consecutive_eintr = 0;
    while (offset < used_) {
      errno = 0;
      const ssize_t written = writer(data_ + offset, used_ - offset);
      if (written < 0) {
        if (errno == EINTR && consecutive_eintr < kMaxConsecutiveEintr) {
          ++consecutive_eintr;
          continue;
        }
        retain_unwritten_suffix(offset);
        return false;
      }
      if (written == 0 || static_cast<std::size_t>(written) > used_ - offset) {
        retain_unwritten_suffix(offset);
        return false;
      }
      offset += static_cast<std::size_t>(written);
      consecutive_eintr = 0;
    }
    used_ = 0;
    return true;
  }

 private:
  void retain_unwritten_suffix(std::size_t written) {
    if (written == 0) return;
    used_ -= written;
    std::memmove(data_, data_ + written, used_);
  }

  char data_[kCapacity]{};
  std::size_t used_ = 0;
};

}  // namespace detail
}  // namespace flash_v4
