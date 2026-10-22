#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p flash_v1/build
# Independent CPU reader; no CUDA, PyTorch or Boost shared libraries required.
g++ -std=c++17 -O2 -shared -fPIC -I Greyhound-master/detector \
 flash_v1/native/snapshot.cpp -pthread -lrt -o flash_v1/build/libflashsnapshot.so
TORCH_ROOT=$(python -c 'import torch,pathlib;print(pathlib.Path(torch.__file__).parent)')
NCCL_ROOT=$(python -c 'import nvidia.nccl,pathlib;print(pathlib.Path(nvidia.nccl.__path__[0]))' 2>/dev/null || true)
mkdir -p flash_v1/build
g++ -std=c++17 -O2 -shared -fPIC -DBOOST_LOG_DYN_LINK \
 -I Greyhound-master/detector -I /usr/local/cuda/include -I "$TORCH_ROOT/include" -I "$NCCL_ROOT/include" \
 flash_v1/native/probe.cpp Greyhound-master/detector/shm_storage.cpp \
 -L /usr/local/cuda/lib64 -lcudart -lboost_log -lboost_log_setup -lboost_thread -lpthread -ldl -lrt \
 -o flash_v1/build/libflashprobe.so
