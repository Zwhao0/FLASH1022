#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../native/observer_output_buffer.inc"

struct CapturingWriter {
  std::string output;
  std::size_t max_chunk = static_cast<std::size_t>(-1);
  unsigned eintr_remaining = 0;
  unsigned calls = 0;
  bool fail_permanently = false;

  ssize_t operator()(const char* data, std::size_t length) {
    ++calls;
    if (eintr_remaining != 0) {
      --eintr_remaining;
      errno = EINTR;
      return -1;
    }
    if (fail_permanently) {
      errno = EIO;
      return -1;
    }
    const std::size_t count = std::min(length, max_chunk);
    output.append(data, count);
    return static_cast<ssize_t>(count);
  }
};

using Buffer = flash_v4::detail::ObserverOutputBuffer;

void test_partial_writes_and_eintr() {
  Buffer buffer;
  CapturingWriter writer;
  writer.max_chunk = 2;
  writer.eintr_remaining = 1;

  assert(buffer.append("first\n", 6, writer));
  assert(buffer.append("second\n", 7, writer));
  assert(buffer.flush(writer));
  assert(writer.output == "first\nsecond\n");
  assert(writer.calls > 2);
  assert(buffer.buffered_bytes() == 0);
}

void test_capacity_crossing_order_and_final_flush() {
  Buffer buffer;
  CapturingWriter writer;
  std::vector<std::string> lines;
  std::string expected;
  for (std::size_t i = 0; i < 32; ++i) {
    std::string line = std::to_string(i) + ":";
    line.append(Buffer::kMaxLine - line.size(), static_cast<char>('A' + (i % 26)));
    lines.push_back(std::move(line));
    expected += lines.back();
  }

  for (std::size_t i = 0; i < 16; ++i) {
    assert(buffer.append(lines[i].data(), lines[i].size(), writer));
  }
  assert(buffer.buffered_bytes() == Buffer::kCapacity);
  assert(writer.output.empty());

  // The next append flushes the full batch before copying the new line.
  assert(buffer.append(lines[16].data(), lines[16].size(), writer));
  assert(writer.output == expected.substr(0, Buffer::kCapacity));
  assert(buffer.buffered_bytes() == Buffer::kMaxLine);

  for (std::size_t i = 17; i < lines.size(); ++i) {
    assert(buffer.append(lines[i].data(), lines[i].size(), writer));
  }
  assert(buffer.buffered_bytes() == Buffer::kCapacity);
  // Shutdown drain flushes the final full batch.
  assert(buffer.flush(writer));
  assert(writer.output == expected);
  assert(buffer.buffered_bytes() == 0);
}

void test_permanent_failure_returns_and_keeps_pending_bytes() {
  Buffer buffer;
  CapturingWriter writer;
  writer.fail_permanently = true;

  assert(buffer.append("pending\n", 8, writer));
  assert(!buffer.flush(writer));
  assert(writer.calls == 1);
  assert(writer.output.empty());
  assert(buffer.buffered_bytes() == 8);
}

void test_partial_failure_retains_suffix_for_retry() {
  Buffer buffer;
  CapturingWriter writer;
  writer.max_chunk = 2;
  assert(buffer.append("abcdef", 6, writer));

  struct PartialThenFail {
    CapturingWriter& sink;
    unsigned calls = 0;
    ssize_t operator()(const char* data, std::size_t length) {
      ++calls;
      if (calls == 1) {
        const std::size_t count = std::min<std::size_t>(2, length);
        sink.output.append(data, count);
        return static_cast<ssize_t>(count);
      }
      errno = EIO;
      return -1;
    }
  } writer_with_failure{writer};

  assert(!buffer.flush(writer_with_failure));
  assert(writer.output == "ab");
  assert(buffer.buffered_bytes() == 4);
  assert(buffer.flush(writer));
  assert(writer.output == "abcdef");
  assert(buffer.buffered_bytes() == 0);
}

void test_repeated_eintr_is_bounded() {
  Buffer buffer;
  CapturingWriter writer;
  writer.eintr_remaining = Buffer::kMaxConsecutiveEintr + 10;

  assert(buffer.append("x", 1, writer));
  assert(!buffer.flush(writer));
  assert(writer.calls == Buffer::kMaxConsecutiveEintr + 1);
  assert(buffer.buffered_bytes() == 1);
}

void test_line_length_limit() {
  Buffer buffer;
  CapturingWriter writer;
  const std::string over_limit(Buffer::kMaxLine + 1, 'x');
  assert(!buffer.append(over_limit.data(), over_limit.size(), writer));
  assert(buffer.buffered_bytes() == 0);
  assert(writer.calls == 0);
}

int main() {
  test_partial_writes_and_eintr();
  test_capacity_crossing_order_and_final_flush();
  test_permanent_failure_returns_and_keeps_pending_bytes();
  test_partial_failure_retains_suffix_for_retry();
  test_repeated_eintr_is_bounded();
  test_line_length_limit();
  std::cout << "observer output buffer CPU tests passed\n";
  return 0;
}
