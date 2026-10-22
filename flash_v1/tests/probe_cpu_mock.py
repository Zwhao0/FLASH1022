#!/usr/bin/env python3
"""Compile probe against CPU mocks; no CUDA/NCCL runtime, GPU or live library.
Optional --baseline PATH compares all eight PROFILE iterations to saved source.
"""
import argparse
import pathlib
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
CUDA = r'''
#pragma once
#include <cstddef>
#include <vector>
#include <cstdint>
extern std::vector<int> event_trace;
using cudaStream_t=void*; using cudaEvent_t=void*;
extern unsigned devices, events, syncs;
inline int cudaGetDevice(int* d){++devices;*d=3;return 0;}
inline int cudaEventCreate(cudaEvent_t* e){++events;*e=(void*)(uintptr_t)events;event_trace.push_back(1);return 0;}
inline int cudaEventRecord(cudaEvent_t e,cudaStream_t){++events;event_trace.push_back(10+(uintptr_t)e);return 0;}
inline int cudaEventSynchronize(cudaEvent_t e){++syncs;event_trace.push_back(20+(uintptr_t)e);return 0;}
inline int cudaEventElapsedTime(float* f,cudaEvent_t,cudaEvent_t){*f=1;return 0;}
'''
NCCL = r'''
#pragma once
#include "cuda_runtime.h"
using ncclComm_t=void*; using ncclResult_t=int; using ncclDataType_t=int; using ncclRedOp_t=int;
struct ncclUniqueId { char internal[128]; }; struct ncclConfig_t{};
constexpr int ncclSuccess=0;
extern "C" {
ncclResult_t ncclCommInitRank(ncclComm_t*,int,ncclUniqueId,int);
ncclResult_t ncclCommInitRankConfig(ncclComm_t*,int,ncclUniqueId,int,ncclConfig_t*);
ncclResult_t ncclGroupStart();ncclResult_t ncclGroupEnd();
ncclResult_t ncclCommCount(ncclComm_t,int*);
ncclResult_t ncclAllReduce(const void*,void*,size_t,ncclDataType_t,ncclRedOp_t,ncclComm_t,cudaStream_t);
}
'''
STORAGE = r'''
#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
constexpr int BUFFER_SIZE=100000, MIN_RECORD_OP_SIZE=16, ALL_REDUCE=4;
extern unsigned records, vectors;
extern std::vector<std::vector<uint64_t>> rows;
struct Record {
 uint64_t commAddr,callNumber,count,buff1Addr,buff2Addr,datatype,pid,callTime,deviceId,caller,aux,duration,numDevices,event_id;
 Record(uint64_t a,uint64_t b,uint64_t c,uint64_t d,uint64_t e,uint64_t f,uint64_t g,uint64_t h,uint64_t i,uint64_t j,uint64_t k,uint64_t l,uint64_t m):commAddr(a),callNumber(b),count(c),buff1Addr(d),buff2Addr(e),datatype(f),pid(g),callTime(h),deviceId(i),caller(j),aux(k),duration(l),numDevices(m),event_id(0){++records;}
 static constexpr size_t numFields(){return 14;}
 std::vector<uint64_t> toVector(){++vectors;return {commAddr,callNumber,count,buff1Addr,buff2Addr,datatype,pid,0,deviceId,caller,aux,duration,numDevices,event_id};}
};
struct NcclRecordStorage {NcclRecordStorage(size_t,size_t){} void addRecord(std::vector<uint64_t> r){rows.push_back(r);}};
'''
DRIVER = r'''
#include <iostream>
#include <cassert>
#include <chrono>
#include <string>
#include <vector>
#include <new>
#include <cstdlib>
unsigned allocations=0;
void* operator new(std::size_t n){++allocations;if(auto p=std::malloc(n))return p;throw std::bad_alloc();}
void operator delete(void* p) noexcept {std::free(p);}
unsigned devices=0,events=0,syncs=0,records=0,vectors=0,counts=0,lookups=0;
std::vector<std::vector<uint64_t>> rows;
std::vector<int> event_trace;
#include "probe.cpp"
volatile unsigned calls=0;
int fake_init(ncclComm_t* c,int,ncclUniqueId,int){*c=(void*)123;return 0;}
int fake_config(ncclComm_t* c,int n,ncclUniqueId id,int r,ncclConfig_t*){return fake_init(c,n,id,r);}
int fake_group(){return 0;}
int fake_count(ncclComm_t,int* n){++counts;*n=2;return 0;}
__attribute__((noinline)) int fake_all(const void*,void*,size_t,int,int,ncclComm_t,cudaStream_t){++calls;return 0;}
extern "C" void* mock_dlsym(void*,const char* n){
 ++lookups;
 if(!strcmp(n,"ncclCommInitRank"))return (void*)fake_init;
 if(!strcmp(n,"ncclCommInitRankConfig"))return (void*)fake_config;
 if(!strcmp(n,"ncclGroupStart")||!strcmp(n,"ncclGroupEnd"))return (void*)fake_group;
 if(!strcmp(n,"ncclCommCount"))return (void*)fake_count;
 if(!strcmp(n,"ncclAllReduce"))return (void*)fake_all;
 return nullptr;
}
void collective(ncclComm_t c,size_t n=32){ncclAllReduce(nullptr,nullptr,n,1,0,c,nullptr);}
int main(int argc,char** argv){
 std::string dir=argv[1];
 setenv("FLASH_RUN_ID","cpu-mock",1);setenv("RANK","0",1);
 setenv("FLASH_STATE_FILE",(dir+"/state").c_str(),1);setenv("FLASH_CONTEXT_DIR",dir.c_str(),1);
 setenv("FLASH_PROFILE_PHASE","attention-backward",1);setenv("FLASH_SAMPLE_STRIDE","8",1);
 ncclComm_t c=nullptr;
 // Initialization inside a group must register only after its group ends.
 ncclGroupStart();ncclCommInitRank(&c,2,ncclUniqueId{},0);assert(ids.empty());ncclGroupEnd();assert(ids.at(c)==hash_id(ncclUniqueId{}));
 auto st=const_cast<int*>(state);int fd=open((dir+"/0.bin").c_str(),O_RDWR);auto ctx=(uint64_t*)mmap(nullptr,16,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);close(fd);
 *st=0;ctx[0]=10;collective(c);ctx[0]=11;collective(c);
 bool baseline=argc>2;
 if(!baseline){assert(records==0&&vectors==0&&devices==0&&counts==0&&events==0&&syncs==0&&rows.empty());}
 rows.clear();records=vectors=devices=counts=events=syncs=0;
 // Exact fixed eight-step sequence: group, nested group, small op, two comms,
 // forward/backward phases, and enough calls to exercise stride rotation.
 ids[(void*)456]=ids.at(c);*st=1;
 for(int iteration=12;iteration<20;++iteration){
  ctx[0]=iteration;
  for(int phase:{1,2,258}){
   ctx[1]=phase;
   for(int j=0;j<12;++j){
    if(j%3==0)ncclGroupStart();
    if(j%3==0){ncclGroupStart();collective(c,8);ncclGroupEnd();}
    collective(c);collective((void*)456);
    if(j%3==2)ncclGroupEnd();
   }
  }
 }
 std::cout<<"COUNTERS "<<records<<" "<<vectors<<" "<<devices<<" "<<counts<<" "<<events<<" "<<syncs<<"\n";
 for(auto& row:rows){for(auto x:row)std::cout<<x<<",";std::cout<<"\n";}
 std::cout<<"EVENTS ";for(auto e:event_trace)std::cout<<e<<",";std::cout<<"\n";
 if(baseline)return 0;
 auto active_records=records;ctx[0]=20;*st=0;collective(c);*st=1;collective(c);
 assert(records==active_records);ctx[0]=21;ctx[1]=2;collective(c);
 assert(records==active_records+1);assert((rows.back()[6]&0xffffffffULL)==0);
 // Active -> off within one group must finish the already pending record.
 ctx[0]=22;ncclGroupStart();collective(c);auto before_flush=rows.size();
 *st=0;collective(c);ncclGroupEnd();assert(rows.size()==before_flush+1);
 // Off/on after an active call in the same iteration must not reset to op0.
 *st=1;auto before_restart=records;collective(c);assert(records==before_restart);
 ctx[0]=23;collective(c);assert((rows.back()[6]&0xffffffffULL)==0);
 *st=0;ctx[0]=24;collective(c);auto old_lookups=lookups;
 auto old_allocations=allocations;
 auto old_records=records,old_events=events,old_syncs=syncs,old_counts=counts,old_devices=devices,old_vectors=vectors;
 constexpr int repetitions=1000000;
 auto bench=[&](bool wrapper){auto begin=std::chrono::steady_clock::now();for(int i=0;i<repetitions;++i){if(wrapper)collective(c);else fake_all(nullptr,nullptr,32,1,0,c,nullptr);}return std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-begin).count()/repetitions;};
 double direct=1e99,monitor=1e99;for(int repeat=0;repeat<5;++repeat){direct=std::min(direct,bench(false));monitor=std::min(monitor,bench(true));}
 assert(allocations==old_allocations&&lookups==old_lookups&&records==old_records&&events==old_events&&syncs==old_syncs&&counts==old_counts&&devices==old_devices&&vectors==old_vectors);
 std::cerr<<"PASS monitor zero instrumentation/allocations; boundary/restart/group flush; cached symbols. CPU ns/call direct="<<direct<<" monitor="<<monitor<<" incremental="<<monitor-direct<<"\n";
}
'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', type=pathlib.Path)
    args = ap.parse_args()
    compiler = shutil.which('c++') or shutil.which('clang++')
    with tempfile.TemporaryDirectory(prefix='flash-probe-cpu-') as work:
        path = pathlib.Path(work)
        for name, text in {'cuda_runtime.h': CUDA, 'nccl.h': NCCL, 'shm_storage.hpp': STORAGE, 'driver.cpp': DRIVER,
                           'dlfcn.h': '#pragma once\n#define RTLD_NEXT ((void*)-1)\n#define RTLD_NOW 2\n#define RTLD_LOCAL 0\nextern "C" void* mock_dlsym(void*,const char*);\n#define dlsym mock_dlsym\ninline void* dlopen(const char*,int){return nullptr;}\n'}.items():
            (path/name).write_text(text)
        outputs = []
        sources = [ROOT/'flash_v1/native/probe.cpp'] + ([args.baseline] if args.baseline else [])
        for index, source in enumerate(sources):
            (path/'probe.cpp').write_text(source.read_text())
            subprocess.run([compiler, '-O2', '-std=c++17', '-pthread', '-I', work, str(path/'driver.cpp'), '-o', str(path/'test')], check=True)
            (path/'state').write_bytes(bytes(4));(path/'0.bin').write_bytes(bytes(16))
            result = subprocess.run([str(path/'test'), work] + (['baseline'] if index else []), text=True, capture_output=True, check=True)
            print(result.stderr, end='')
            if index == 0:
                print('PROFILE ' + result.stdout.splitlines()[0])
            outputs.append(result.stdout)
        if len(outputs)==2:
            assert outputs[0]==outputs[1], 'PROFILE record/order/CUDA instrumentation differs from baseline'
            print('PASS exact eight-iteration PROFILE equivalence (timestamps normalized)')
        else:
            print('PASS eight-iteration PROFILE mock')
if __name__ == '__main__':
    main()
