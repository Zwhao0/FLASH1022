"""NCCL hook smoke: real GPU AllReduce, no failure injection."""
import os
import torch
import torch.distributed as d
rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
d.init_process_group('nccl')
x=torch.full((2048,),float(rank+1),device='cuda')
for _ in range(5):
    x.fill_(rank+1);d.all_reduce(x);torch.cuda.synchronize();assert x[0].item()==10
print('hook allreduce OK',rank,flush=True)
d.destroy_process_group()
