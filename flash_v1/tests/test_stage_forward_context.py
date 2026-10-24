import unittest
from flash_v1.stage_forward_context import observed_context


class ContextTests(unittest.TestCase):
    def test_actual_constructor_and_hooks_only(self):
        Scoped=type('ScopedComputeSlowdown',(),{})
        scoped=Scoped();scoped.handles=[object()];scoped.target={'rank':9};scoped.extra_ratio=5
        self.assertEqual(observed_context(scoped),'scoped_dispatch')
        scoped.target=None;scoped.extra_ratio=0
        self.assertEqual(observed_context(scoped),'scoped_dispatch')
        scoped.handles=[]
        with self.assertRaises(ValueError):observed_context(scoped)
        self.assertEqual(observed_context(None),'plain')
        self.assertEqual(observed_context(type('WholeComputeSlowdown',(),{})()),'whole_dispatch')
        with self.assertRaises(ValueError):observed_context(object())
