import unittest
from flash_v1.stage_forward_benchmark import validate_blocks, active_auxiliary


class BenchmarkTests(unittest.TestCase):
    def test_monitor_crossover_no_profile_or_cuda_needed(self):
        aux=object();blocks=validate_blocks([dict(id='A',start=43,end=50,aux_enabled=False),dict(id='B',start=51,end=66,aux_enabled=True)])
        self.assertIsNone(active_auxiliary(aux,blocks,43,0))
        self.assertIs(active_auxiliary(aux,blocks,51,0),aux)
        self.assertIs(active_auxiliary(aux,blocks,85,1),aux)
        with self.assertRaises(ValueError):active_auxiliary(aux,blocks,43,1)

    def test_invalid_blocks_rejected(self):
        for blocks in ([dict(id='bad',start=True,end=2,aux_enabled=False)],[dict(id='a',start=1,end=3,aux_enabled=True),dict(id='b',start=3,end=5,aux_enabled=False)],[dict(id='a',start=1,end=3,aux_enabled=1)]):
            with self.subTest(blocks=blocks),self.assertRaises(ValueError):validate_blocks(blocks)
