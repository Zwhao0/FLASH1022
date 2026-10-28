// Included by observer.cpp. Generic launch nesting only: no actor/truth imports.
// LD_PRELOAD must place this observer before any downstream launch wrappers so
// its outer guard also covers their synchronously submitted helper launches.
#include <cuda.h>
#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif

namespace {
constexpr unsigned kKernelStride = 256;
thread_local bool inside_kernel_launch = false;
struct KernelLaunchGuard {
  KernelLaunchGuard() { inside_kernel_launch = true; }
  ~KernelLaunchGuard() { inside_kernel_launch = false; }
};
void* kernel_driver_library() {
  static void* library = dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);
  return library;
}
template<class T> T kernel_real(const char* name,bool runtime) {
  void* p=dlsym(RTLD_NEXT,name);
  if(!p) {
    static std::mutex mutex;
    static void* cudart=nullptr;
    std::lock_guard<std::mutex> lock(mutex);
    if(runtime&&!cudart) {
      const char* path=std::getenv("FLASH_V4_KERNEL_CUDART_LIBRARY");
      cudart=dlopen(path?path:"libcudart.so.12",RTLD_NOW|RTLD_LOCAL);
    }
    void* library=runtime?cudart:kernel_driver_library();
    if(library) p=dlsym(library,name);
  }
  if(!p) { std::fprintf(stderr,"FLASH v4: cannot resolve real %s\n",name);std::abort(); }
  return reinterpret_cast<T>(p);
}
struct KernelIdentity { uint64_t hash=0; int category=0; }; // 0 unknown, 1 compute, 2 NCCL
#include "observer_identity_cache.inc"
uint64_t kernel_name_hash(const char* name) {
  uint64_t hash=14695981039346656037ULL;
  for(const unsigned char* p=reinterpret_cast<const unsigned char*>(name);*p;++p) { hash^=*p;hash*=1099511628211ULL; }
  return hash;
}
bool kernel_mentions_nccl(const std::string& text) {
  std::string lower=text;
  std::transform(lower.begin(),lower.end(),lower.begin(),[](unsigned char c){ return char(std::tolower(c)); });
  return lower.find("nccl")!=std::string::npos;
}
KernelIdentity kernel_identity(uintptr_t function,bool runtime) {
  static thread_local KernelIdentityFrontCache front;
  KernelIdentity cached;
  if (front.get(function,runtime,cached)) return cached;
  using NameKey=std::pair<uintptr_t,bool>;
  static std::mutex mutex;
  static std::map<NameKey,KernelIdentity> cache;
  NameKey key{function,runtime};
  { std::lock_guard<std::mutex> lock(mutex);auto it=cache.find(key);if(it!=cache.end()) {
      front.put(function,runtime,it->second);return it->second;
    } }
  KernelIdentity identity;
  std::string owner;
  if(runtime) {
    Dl_info info{};
    if(dladdr(reinterpret_cast<void*>(function),&info)) {
      if(info.dli_fname) owner+=info.dli_fname;
      if(info.dli_sname) { owner+=' ';owner+=info.dli_sname; }
    }
  }
  const char* device_name=nullptr;
  if(kernel_mentions_nccl(owner)) identity.category=2;
  else {
    bool named=false;
    if(runtime) {
      using Fn=cudaError_t (*)(const char**,const void*);
      // CUDA runtime instance shared with the launch path; this optional
      // reflection occurs only on first encounter, never on each cache hit.
      static Fn name=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"cudaFuncGetName"));
      named=name&&name(&device_name,reinterpret_cast<const void*>(function))==cudaSuccess&&device_name&&*device_name;
    } else {
      using Fn=CUresult (*)(const char**,CUfunction);
      static Fn name=kernel_driver_library()?reinterpret_cast<Fn>(dlsym(kernel_driver_library(),"cuFuncGetName")):nullptr;
      named=name&&name(&device_name,reinterpret_cast<CUfunction>(function))==CUDA_SUCCESS&&device_name&&*device_name;
    }
    if(named) {
      if(kernel_mentions_nccl(device_name)) identity.category=2;
      else { identity.category=1;identity.hash=kernel_name_hash(device_name); }
    }
  }
  std::lock_guard<std::mutex> lock(mutex);
  if(cache.size()<4096) { cache.emplace(key,identity);front.put(function,runtime,identity); }
  return identity;
}
template<class F> auto observe_kernel(uintptr_t function,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,size_t smem,bool runtime,const char* api,cudaStream_t stream,F call) -> decltype(call()) {
  auto& o=observer();
  if(!o.kernel_enabled||!o.active()) return call();
  if(inside_kernel_launch) { ++o.kernel_nested_excluded;return call(); }
  if(inside_blas) { ++o.kernel_blas_excluded;return call(); }
  KernelLaunchGuard guard;
  auto identity=kernel_identity(function,runtime);
  if(identity.category!=1) {
    if(identity.category==2) ++o.kernel_nccl_excluded;else ++o.kernel_unknown_excluded;
    return call();
  }
  uint64_t sequence=o.kernel_sequence.fetch_add(1);
  ++o.seen;++o.kernel_candidates;
  uint64_t iteration=o.iteration.load();
  if(sequence%kKernelStride!=iteration%kKernelStride) return call();
  ++o.sampled;++o.kernel_sampled;
  Row row;row.kind="kernel";row.iteration=iteration;row.sequence=sequence;row.kernel_id=identity.hash;
  row.grid[0]=gx;row.grid[1]=gy;row.grid[2]=gz;row.block[0]=bx;row.block[1]=by;row.block[2]=bz;
  row.dynamic_shared_bytes=smem;row.launch_api=api;
  Timing timing;
  const char* failure=o.event_start(timing,stream);
  auto result=call();timing.rows.push_back(row);
  o.finish(timing,stream,failure?failure:(result==0?nullptr:"kernel_launch_error"));
  return result;
}
using KernelRuntimeLaunch=cudaError_t (*)(const void*,dim3,dim3,void**,size_t,cudaStream_t);
using KernelRuntimeLaunchEx=cudaError_t (*)(const cudaLaunchConfig_t*,const void*,void**);
using KernelDriverLaunch=CUresult (*)(CUfunction,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,CUstream,void**,void**);
using KernelDriverLaunchEx=CUresult (*)(const CUlaunchConfig*,CUfunction,void**,void**);
} // namespace

extern "C" cudaError_t cudaLaunchKernel(const void* function,dim3 grid,dim3 block,void** args,size_t smem,cudaStream_t stream) {
  static auto fn=kernel_real<KernelRuntimeLaunch>("cudaLaunchKernel",true);
  return observe_kernel(reinterpret_cast<uintptr_t>(function),grid.x,grid.y,grid.z,block.x,block.y,block.z,smem,true,"runtime",stream,[&]{return fn(function,grid,block,args,smem,stream);});
}
extern "C" cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config,const void* function,void** args) {
  static auto fn=kernel_real<KernelRuntimeLaunchEx>("cudaLaunchKernelExC",true);
  if(!config)return fn(config,function,args);
  return observe_kernel(reinterpret_cast<uintptr_t>(function),config->gridDim.x,config->gridDim.y,config->gridDim.z,config->blockDim.x,config->blockDim.y,config->blockDim.z,config->dynamicSmemBytes,true,"runtime_ex",config->stream,[&]{return fn(config,function,args);});
}
extern "C" cudaError_t cudaLaunchKernel_ptsz(const void* function,dim3 grid,dim3 block,void** args,size_t smem,cudaStream_t stream) {
  static auto fn=kernel_real<KernelRuntimeLaunch>("cudaLaunchKernel_ptsz",true);
  cudaStream_t observed=stream?stream:reinterpret_cast<cudaStream_t>(CU_STREAM_PER_THREAD);
  return observe_kernel(reinterpret_cast<uintptr_t>(function),grid.x,grid.y,grid.z,block.x,block.y,block.z,smem,true,"runtime_ptsz",observed,[&]{return fn(function,grid,block,args,smem,stream);});
}
extern "C" cudaError_t cudaLaunchKernelExC_ptsz(const cudaLaunchConfig_t* config,const void* function,void** args) {
  static auto fn=kernel_real<KernelRuntimeLaunchEx>("cudaLaunchKernelExC_ptsz",true);
  if(!config)return fn(config,function,args);
  cudaStream_t observed=config->stream?config->stream:reinterpret_cast<cudaStream_t>(CU_STREAM_PER_THREAD);
  return observe_kernel(reinterpret_cast<uintptr_t>(function),config->gridDim.x,config->gridDim.y,config->gridDim.z,config->blockDim.x,config->blockDim.y,config->blockDim.z,config->dynamicSmemBytes,true,"runtime_ex_ptsz",observed,[&]{return fn(config,function,args);});
}
extern "C" CUresult cuLaunchKernel(CUfunction function,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned smem,CUstream stream,void** args,void** extra) {
  static auto fn=kernel_real<KernelDriverLaunch>("cuLaunchKernel",false);
  return observe_kernel(reinterpret_cast<uintptr_t>(function),gx,gy,gz,bx,by,bz,smem,false,"driver",reinterpret_cast<cudaStream_t>(stream),[&]{return fn(function,gx,gy,gz,bx,by,bz,smem,stream,args,extra);});
}
extern "C" CUresult cuLaunchKernelEx(const CUlaunchConfig* config,CUfunction function,void** args,void** extra) {
  static auto fn=kernel_real<KernelDriverLaunchEx>("cuLaunchKernelEx",false);
  if(!config)return fn(config,function,args,extra);
  return observe_kernel(reinterpret_cast<uintptr_t>(function),config->gridDimX,config->gridDimY,config->gridDimZ,config->blockDimX,config->blockDimY,config->blockDimZ,config->sharedMemBytes,false,"driver_ex",reinterpret_cast<cudaStream_t>(config->hStream),[&]{return fn(config,function,args,extra);});
}
extern "C" CUresult cuLaunchKernel_ptsz(CUfunction function,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned smem,CUstream stream,void** args,void** extra) {
  static auto fn=kernel_real<KernelDriverLaunch>("cuLaunchKernel_ptsz",false);
  CUstream observed=stream?stream:CU_STREAM_PER_THREAD;
  return observe_kernel(reinterpret_cast<uintptr_t>(function),gx,gy,gz,bx,by,bz,smem,false,"driver_ptsz",reinterpret_cast<cudaStream_t>(observed),[&]{return fn(function,gx,gy,gz,bx,by,bz,smem,stream,args,extra);});
}
extern "C" CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig* config,CUfunction function,void** args,void** extra) {
  static auto fn=kernel_real<KernelDriverLaunchEx>("cuLaunchKernelEx_ptsz",false);
  if(!config)return fn(config,function,args,extra);
  CUstream observed=config->hStream?config->hStream:CU_STREAM_PER_THREAD;
  return observe_kernel(reinterpret_cast<uintptr_t>(function),config->gridDimX,config->gridDimY,config->gridDimZ,config->blockDimX,config->blockDimY,config->blockDimZ,config->sharedMemBytes,false,"driver_ex_ptsz",reinterpret_cast<cudaStream_t>(observed),[&]{return fn(config,function,args,extra);});
}
namespace {
void kernel_redirect(const char* symbol,void** function,cuuint64_t flags) {
  if(!symbol||!function||!*function||!observer().kernel_enabled)return;
  bool ptsz=(flags&CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)!=0;
  if(std::strcmp(symbol,"cuLaunchKernel")==0)*function=ptsz?reinterpret_cast<void*>(&cuLaunchKernel_ptsz):reinterpret_cast<void*>(&cuLaunchKernel);
  else if(std::strcmp(symbol,"cuLaunchKernelEx")==0)*function=ptsz?reinterpret_cast<void*>(&cuLaunchKernelEx_ptsz):reinterpret_cast<void*>(&cuLaunchKernelEx);
}
} // namespace
extern "C" CUresult cuGetProcAddress(const char* symbol,void** function,int version,cuuint64_t flags) {
  using Fn=CUresult (*)(const char*,void**,int,cuuint64_t);
  static auto fn=kernel_real<Fn>("cuGetProcAddress",false);
  auto result=fn(symbol,function,version,flags);if(result==CUDA_SUCCESS)kernel_redirect(symbol,function,flags);return result;
}
extern "C" CUresult cuGetProcAddress_v2(const char* symbol,void** function,int version,cuuint64_t flags,CUdriverProcAddressQueryResult* status) {
  using Fn=CUresult (*)(const char*,void**,int,cuuint64_t,CUdriverProcAddressQueryResult*);
  static auto fn=kernel_real<Fn>("cuGetProcAddress_v2",false);
  auto result=fn(symbol,function,version,flags,status);if(result==CUDA_SUCCESS)kernel_redirect(symbol,function,flags);return result;
}
