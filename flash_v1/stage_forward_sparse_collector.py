"""Original57c collector restricted by an explicit caller-side sampling gate."""
from .stage_forward_evidence import StageForwardEvidence as DenseEvidence
from .stage_forward_sampling import Sampler
class StageForwardEvidence(DenseEvidence):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.sampler=Sampler()
    def selected(self,iteration,enabled,epoch):return self.sampler.select(iteration,enabled,epoch)
