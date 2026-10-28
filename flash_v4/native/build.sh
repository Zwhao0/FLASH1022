#!/usr/bin/env bash
set -euo pipefail
: "${CUDA_HOME:?set CUDA_HOME to an existing CUDA toolkit}"
: "${NCCL_INCLUDE_DIR:?set NCCL_INCLUDE_DIR to the directory containing nccl.h}"
src_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
out="${1:-${src_dir}/libflash_v4_observer.so}"
"${CXX:-g++}" -std=c++17 -O2 -fPIC -shared -pthread \
  -Wall -Wextra -Werror=return-type \
  -I"${CUDA_HOME}/include" -I"${NCCL_INCLUDE_DIR}" \
  "${src_dir}/observer.cpp" -L"${CUDA_HOME}/lib64" \
  -Wl,-rpath,"${CUDA_HOME}/lib64" -lcudart -ldl -o "$out"
printf '%s\n' "$out"
