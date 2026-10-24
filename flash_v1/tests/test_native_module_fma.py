import unittest
from types import SimpleNamespace
from flash_v1.native_module_fma import NativeModuleFMA


class Engine:
    def __init__(self,*args):self.units=[]
    def launch_units(self,units):self.units.append(units)


class Module:
    def register_forward_hook(self,callback,with_kwargs):
        self.callback=callback;return SimpleNamespace(remove=lambda:None)


class NativeTests(unittest.TestCase):
    def test_no_tensor_edits_second_mb_only_and_restore(self):
        module=Module();stage=SimpleNamespace(named_modules=lambda:[('norm',module)])
        cases=[dict(case_id='module',rank=0,kind='module',module_path='norm',start=3,end=4,extra_ms=20),dict(case_id='loss',rank=0,kind='second_microbatch_loss',start=3,end=4,extra_ms=30)]
        owner=NativeModuleFMA(None,stage,0,cases,10,Engine);owner.begin_iteration(3)
        x=object();fn=lambda *args,**kwargs:x;functional=SimpleNamespace(cross_entropy=fn)
        with owner.loss_wrapper(functional):
            self.assertIs(functional.cross_entropy(x),x)
            self.assertEqual(owner.current['injected_launches'],0)
            self.assertIs(functional.cross_entropy(x),x)
        self.assertIs(functional.cross_entropy,fn)
        self.assertIsNone(module.callback(module,(x,),{},x));module.callback(module,(x,),{},x)
        owner.end_iteration()
        self.assertEqual(owner.current['injected_launches'],3)
        self.assertEqual(owner.engine.units,[4096,300,100,100])
        owner.begin_iteration(5);module.callback(module,(x,),{},x)
        self.assertEqual(owner.current['injected_launches'],0);owner.close()

    def test_other_rank_never_allocates_or_launches(self):
        stage=SimpleNamespace(named_modules=lambda:[])
        owner=NativeModuleFMA(None,stage,1,[dict(case_id='loss',rank=0,kind='second_microbatch_loss',start=1,end=2,extra_ms=30)],10,Engine)
        owner.begin_iteration(1);owner._observe(owner.cases[0]);owner._observe(owner.cases[0]);self.assertIsNone(owner.engine)
