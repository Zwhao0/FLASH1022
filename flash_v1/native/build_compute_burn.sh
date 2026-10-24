#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p flash_v1/build
"${NVCC:-nvcc}" -O3 -shared -Xcompiler -fPIC flash_v1/native/compute_burn.cu \
  -o flash_v1/build/libflashcompute.so
