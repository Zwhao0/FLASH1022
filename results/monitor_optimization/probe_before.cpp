// FLASH adapter: reuses GREYHOUND Record, RecordBuffer and NcclRecordStorage.
// Public NCCL APIs replace GREYHOUND's version-specific private struct parser.
#include <cuda_runtime.h>
#include <nccl.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <mutex>
#include <memory>
#include <unordered_map>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include "shm_storage.hpp"

namespace {
std::once_flag init_flag;
std::unique_ptr<NcclRecordStorage> storage;
std::unordered_map<ncclComm_t,uint64_t> ids;
thread_local std::vector<std::pair<ncclComm_t*,uint64_t>> initializations;
volatile int* state=nullptr;
volatile uint64_t* iteration_context=nullptr;
int sample_stride=1;
bool attention_backward_only=false;
thread_local std::unordered_map<ncclComm_t,std::pair<uint64_t,uint64_t>> operation_counts;
thread_local int depth=0;
thread_local uint64_t group_id=0;
struct Pending { Record record; bool timed; };
thread_local cudaEvent_t group_begin=nullptr,group_end=nullptr;
thread_local cudaStream_t group_stream=nullptr;
thread_local bool timed_group=false;
thread_local std::vector<Pending> pending;
uint64_t now_us() { return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void init() {
 std::call_once(init_flag,[]{
  std::string suffix=std::string(getenv("FLASH_RUN_ID"))+"_"+getenv("RANK");
  setenv("FLASH_RECORD_NAME", ("flash_record_"+suffix).c_str(),1);
  setenv("FLASH_LOCK_NAME", ("flash_lock_"+suffix).c_str(),1);
  storage.reset(new NcclRecordStorage(Record::numFields(),BUFFER_SIZE));
  int fd=open(getenv("FLASH_STATE_FILE"),O_RDWR);
  if(fd<0) { perror("FLASH state"); abort(); }
  state=(volatile int*)mmap(nullptr,4,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0); close(fd);
  if(const char* directory=getenv("FLASH_CONTEXT_DIR")) {
   attention_backward_only=getenv("FLASH_PROFILE_PHASE") && std::strcmp(getenv("FLASH_PROFILE_PHASE"),"attention-backward")==0;
   sample_stride=getenv("FLASH_SAMPLE_STRIDE") ? std::max(1,atoi(getenv("FLASH_SAMPLE_STRIDE"))) : 1;
   std::string path=std::string(directory)+"/"+getenv("RANK")+".bin";
   int context_fd=open(path.c_str(),O_RDONLY);
   if(context_fd<0) {perror("FLASH iteration context");abort();}
   struct stat context_stat;
   if(fstat(context_fd,&context_stat)!=0 || context_stat.st_size!=16) {
    fprintf(stderr,"FLASH context must contain iteration and phase (16 bytes)\n");abort();
   }
   iteration_context=(volatile uint64_t*)mmap(nullptr,16,PROT_READ,MAP_SHARED,context_fd,0);
   close(context_fd);
   if(iteration_context==MAP_FAILED) {perror("FLASH iteration mmap");abort();}
  }
 });
}
template<class T> T real(const char* name) {
 void* p=dlsym(RTLD_NEXT,name);
 // PyTorch loads NCCL in a local ELF scope; resolve explicitly like GREYHOUND.
 if(!p) { static void* library=dlopen(getenv("NCCL_PATH") ? getenv("NCCL_PATH") : "libnccl.so.2",RTLD_NOW|RTLD_LOCAL); if(library) p=dlsym(library,name); }
 if(!p) { fprintf(stderr,"FLASH missing %s\n",name); abort(); }
 return reinterpret_cast<T>(p);
}
uint64_t hash_id(const ncclUniqueId& id) {
 uint64_t h=1469598103934665603ULL; for(auto c:id.internal) {h^=(unsigned char)c;h*=1099511628211ULL;}
 return h & 0x7fffffffffffffffULL;
}
void register_comm(ncclComm_t* comm,ncclUniqueId id) {init(); if(depth) initializations.emplace_back(comm,hash_id(id)); else ids[*comm]=hash_id(id);}
void flush() {
 uint64_t duration=0;
 if(timed_group) {
  float ms=0;cudaEventRecord(group_end,group_stream);cudaEventSynchronize(group_end);
  cudaEventElapsedTime(&ms,group_begin,group_end);duration=(uint64_t)(ms*1000);
 }
 for(auto &p:pending) {
  if(p.timed) p.record.duration=duration;
  storage->addRecord(p.record.toVector());
 }
 pending.clear();timed_group=false;
}
}
extern "C" ncclResult_t ncclCommInitRank(ncclComm_t* c,int n,ncclUniqueId id,int r) {
 auto rc=real<decltype(&ncclCommInitRank)>("ncclCommInitRank")(c,n,id,r); if(rc==ncclSuccess)register_comm(c,id);return rc;
}
extern "C" ncclResult_t ncclCommInitRankConfig(ncclComm_t* c,int n,ncclUniqueId id,int r,ncclConfig_t* cfg) {
 auto rc=real<decltype(&ncclCommInitRankConfig)>("ncclCommInitRankConfig")(c,n,id,r,cfg);if(rc==ncclSuccess)register_comm(c,id);return rc;
}
extern "C" ncclResult_t ncclGroupStart() {if(depth==0)++group_id;depth++;return real<decltype(&ncclGroupStart)>("ncclGroupStart")();}
extern "C" ncclResult_t ncclGroupEnd() {
 auto rc=real<decltype(&ncclGroupEnd)>("ncclGroupEnd")();if(--depth==0) {for(auto &i:initializations) ids[*i.first]=i.second; initializations.clear(); if(!pending.empty())flush();}return rc;
}
extern "C" ncclResult_t ncclAllReduce(const void* in,void* out,size_t count,ncclDataType_t dtype,ncclRedOp_t op,ncclComm_t comm,cudaStream_t stream) {
 init(); int dev=0,n=0;cudaGetDevice(&dev);real<decltype(&ncclCommCount)>("ncclCommCount")(comm,&n);
 // Stable DP evidence only; EP's dynamic Send/Recv is intentionally excluded.
 // Snapshot the host launch context once; group flush may happen later.
 uint64_t iteration=iteration_context ? iteration_context[0] : 0;
 uint64_t phase=iteration_context ? iteration_context[1] : 0;
 auto &counter=operation_counts[comm];
 if(counter.first!=iteration) {counter.first=iteration;counter.second=0;}
 uint64_t operation_index=counter.second++;
 // One first call protects small DP groups; rotate the remaining TP sample.
 bool eligible=!attention_backward_only || (phase & 0xff)==2;
 bool selected=!iteration_context || operation_index==0 || (eligible && operation_index%sample_stride==iteration%sample_stride);
 bool timed=(*state==1 && count>=MIN_RECORD_OP_SIZE && selected);
 // buff1 carries Group ID; optional acquisition context uses buff2 for iteration.
 // Collection pid slot packs phase in high32, communicator-local op index
 // in low32. The 14-field record layout and default-mode pid are unchanged.
 uint64_t collection_slot=((phase & 0xffffffffULL)<<32) | (operation_index & 0xffffffffULL);
 if(depth==0)++group_id;
 Record rec((uint64_t)comm,ALL_REDUCE,count,group_id,iteration_context ? iteration : (uint64_t)out,(uint64_t)dtype,iteration_context ? collection_slot : (uint64_t)getpid(),now_us(),dev,atoi(getenv("RANK")),ids.at(comm),0,n);
 Pending p{rec,timed};
 if(timed && !timed_group) {
  if(!group_begin) {cudaEventCreate(&group_begin);cudaEventCreate(&group_end);}
  group_stream=stream;cudaEventRecord(group_begin,stream);timed_group=true;
 }
 auto rc=real<decltype(&ncclAllReduce)>("ncclAllReduce")(in,out,count,dtype,op,comm,stream);
 if(count>=MIN_RECORD_OP_SIZE) {pending.push_back(p);if(depth==0)flush();}
 return rc;
}
