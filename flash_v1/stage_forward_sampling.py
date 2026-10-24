"""Explicit sparse3 contract; selected positions are relative to EACH epoch."""
CONFIG=dict(schema='flash_stage_forward_sparse3_v1',window_size=8,sample_offsets=[1,3,7],expected_aux_rows=48,core_evidence_steps=8)
def contract(epoch,start):return dict(CONFIG,epoch=epoch,start=start,end=start+7)
class Sampler:
    def __init__(self):self.starts={};self.last={};self.invalid={}
    def select(self,iteration,enabled,epoch):
        if not enabled:return False
        if type(epoch) is not int or epoch<=0 or type(iteration) is not int:
            return False
        if epoch not in self.starts:self.starts[epoch]=iteration
        elif iteration!=self.last[epoch]+1:self.invalid[epoch]='noncontiguous_or_duplicate_epoch_activation'
        self.last[epoch]=iteration
        relative=iteration-self.starts[epoch]+1
        if not 1<=relative<=8:self.invalid[epoch]='epoch_outside_eight_step_window'
        return epoch not in self.invalid and relative in CONFIG['sample_offsets']
    def manifest(self,epoch):return contract(epoch,self.starts[epoch])
