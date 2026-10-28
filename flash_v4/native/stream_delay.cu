#include <cuda_runtime.h>

// One resident thread, not a full-GPU busy loop. It changes only the ordering
// of this process's stream and never reads/writes training tensors.
__global__ void flash_v4_stream_wait_kernel(unsigned long long nanoseconds) {
  unsigned long long start, now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  do {
    asm volatile("nanosleep.u32 256;" ::: "memory");
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  } while (now - start < nanoseconds);
}

extern "C" cudaError_t flash_v4_delay_launch_wait(cudaStream_t stream, unsigned long long ns) {
  void* args[] = {&ns};
  return cudaLaunchKernel(reinterpret_cast<const void*>(flash_v4_stream_wait_kernel),
                          dim3(1), dim3(1), args, 0, stream);
}
