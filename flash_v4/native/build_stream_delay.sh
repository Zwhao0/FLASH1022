#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_HOME:?set CUDA_HOME to an existing CUDA toolkit}"
src_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
out="${1:-${src_dir}/libflash_v4_stream_delay.so}"
arch="${FLASH_V4_DELAY_CUDA_ARCH:-90}"
"${CUDA_HOME}/bin/nvcc" -std=c++17 -O2 -shared -cudart shared \
  -gencode "arch=compute_${arch},code=sm_${arch}" \
  -gencode "arch=compute_${arch},code=compute_${arch}" \
  -Xcompiler=-fPIC,-pthread,-Wall,-Wextra \
  "${src_dir}/stream_delay.cpp" "${src_dir}/stream_delay.cu" \
  -L"${CUDA_HOME}/lib64" -Xlinker=-rpath -Xlinker="${CUDA_HOME}/lib64" \
  -ldl -lpthread -o "$out"
printf '%s\n' "$out"
