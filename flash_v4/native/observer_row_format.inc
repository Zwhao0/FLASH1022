// Bounded integer formatting only. Floating elapsed formatting stays snprintf.
// False leaves the caller on the original snprintf path, including truncation.
struct ObserverRowWriter {
  char* cursor;
  char* end;
  bool ok = true;
  void text(const char* value) {
    const size_t count = std::strlen(value);
    if (!ok || count > size_t(end - cursor)) { ok = false; return; }
    std::memcpy(cursor, value, count); cursor += count;
  }
  template<class Integer> void number(Integer value) {
    if (!ok) return;
    auto result = std::to_chars(cursor, end, value);
    if (result.ec != std::errc{}) { ok = false; return; }
    cursor = result.ptr;
  }
  void hex64(uint64_t value) {
    if (!ok || end - cursor < 16) { ok = false; return; }
    static constexpr char digits[] = "0123456789abcdef";
    for (int shift = 60; shift >= 0; shift -= 4) *cursor++ = digits[(value >> shift) & 15];
  }
};
bool format_observer_row(char* line, size_t capacity, const Row& r, int rank,
                         const char* elapsed, const char* reason) {
  if (!capacity) return false;
  ObserverRowWriter w{line, line + capacity - 1};
  // Compute-only L1 evidence. Omit unused stream/communication/size/scope
  // metadata while retaining the original formatter below for other rows.
  if (std::strcmp(r.kind, "gemm") == 0 || std::strcmp(r.kind, "lt_matmul") == 0) {
    w.text("{\"kind\":\""); w.text(r.kind);
    w.text("\",\"rank\":"); w.number(rank);
    w.text(",\"iteration\":"); w.number((unsigned long long)r.iteration);
    w.text(",\"sequence\":"); w.number((unsigned long long)r.sequence);
    w.text(",\"op\":"); w.number(r.op);
    w.text(",\"dtype\":"); w.number(r.dtype);
    w.text(",\"dtype_b\":"); w.number(r.dtype_b);
    w.text(",\"dtype_c\":"); w.number(r.dtype_c);
    w.text(",\"dtype_d\":"); w.number(r.dtype_d);
    w.text(",\"compute\":"); w.number(r.compute);
    w.text(",\"batch\":"); w.number(r.batch);
    w.text(",\"shape\":["); w.number((unsigned long long)r.m);
    w.text(","); w.number((unsigned long long)r.n);
    w.text(","); w.number((unsigned long long)r.k);
    w.text("],\"elapsed_us\":"); w.text(elapsed);
    w.text(",\"skip\":\""); w.text(reason ? reason : ""); w.text("\"}\n");
    if (!w.ok) return false;
    *w.cursor = 0;
    return true;
  }
  if (std::strcmp(r.kind, "kernel") == 0) {
    w.text("{\"kind\":\"kernel\",\"rank\":");
    w.number(rank);
    w.text(",\"iteration\":");
    w.number((unsigned long long)r.iteration);
    w.text(",\"sequence\":");
    w.number((unsigned long long)r.sequence);
    w.text(",\"kernel_id\":\"");
    w.hex64((unsigned long long)r.kernel_id);
    w.text("\",\"grid\":[");
    w.number(r.grid[0]);
    w.text(",");
    w.number(r.grid[1]);
    w.text(",");
    w.number(r.grid[2]);
    w.text("],\"block\":[");
    w.number(r.block[0]);
    w.text(",");
    w.number(r.block[1]);
    w.text(",");
    w.number(r.block[2]);
    w.text("],\"dynamic_shared_bytes\":");
    w.number((unsigned long long)r.dynamic_shared_bytes);
    w.text(",\"launch_api\":\"");
    w.text(r.launch_api);
    w.text("\",\"sampling_stride\":256,\"elapsed_us\":");
    w.text(elapsed);
    w.text(",\"skip\":\"");
    w.text(reason?reason:"");
    w.text("\"}\n");
  } else {
    w.text("{\"kind\":\"");
    w.text(r.kind);
    w.text("\",\"rank\":");
    w.number(rank);
    w.text(",\"iteration\":");
    w.number((unsigned long long)r.iteration);
    w.text(",\"sequence\":");
    w.number((unsigned long long)r.sequence);
    w.text(",\"stream\":");
    w.number((unsigned long long)r.stream);
    w.text(",\"comm_id\":\"");
    w.hex64((unsigned long long)r.comm);
    w.text("\",\"op\":");
    w.number(r.op);
    w.text(",\"numel\":");
    w.number((unsigned long long)r.numel);
    w.text(",\"dtype\":");
    w.number(r.dtype);
    w.text(",\"dtype_b\":");
    w.number(r.dtype_b);
    w.text(",\"dtype_c\":");
    w.number(r.dtype_c);
    w.text(",\"dtype_d\":");
    w.number(r.dtype_d);
    w.text(",\"compute\":");
    w.number(r.compute);
    w.text(",\"batch\":");
    w.number(r.batch);
    w.text(",\"shape\":[");
    w.number((unsigned long long)r.m);
    w.text(",");
    w.number((unsigned long long)r.n);
    w.text(",");
    w.number((unsigned long long)r.k);
    w.text("],\"elapsed_us\":");
    w.text(elapsed);
    w.text(",\"group\":");
    w.number((unsigned long long)r.group);
    w.text(",\"timing_scope\":\"");
    w.text(r.group ? "nccl_group_envelope" : "library_call_envelope");
    w.text("\",\"skip\":\"");
    w.text(reason ? reason : "");
    w.text("\"}\n");
  }
  if (!w.ok) return false;
  *w.cursor = 0;
  return true;
}
