"""New-plan defaults only: frozen plans/runtime and workload remain unchanged."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from flash_v1.scripts import v2_collect
from flash_v1.scripts.v2_benchmark import benchmark_plan


class PlanDefaultsTests(unittest.TestCase):
    def check_only_flag(self, factory):
        default, off = factory(), factory(monitor_confirm_recent_three=False)
        self.assertIs(default.pop('monitor_confirm_recent_three'), True)
        self.assertIs(off.pop('monitor_confirm_recent_three'), False)
        self.assertEqual(default, off)

    def test_collection_and_online_defaults_change_only_flag(self):
        for layout in v2_collect.LAYOUTS:
            self.check_only_flag(lambda **kw: v2_collect.plan('cpu',layout,'development',7,**kw))
            self.check_only_flag(lambda **kw: v2_collect.online_validation_plan('cpu',layout,7,'model.json',**kw))
        with self.assertRaises(ValueError):
            v2_collect.plan('cpu','legacy','train',7,monitor_confirm_recent_three='false')

    def test_all_benchmark_families_allow_identical_false_control(self):
        for mode in ('A','B'):
            for family in ('profile','profile_healthy','monitor','full_online'):
                self.check_only_flag(lambda **kw: benchmark_plan('cpu','legacy',7,mode,family,'model.json',**kw))

    def test_cli_default_and_explicit_false_write_bool(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(v2_collect,'ROOT',Path(folder)):
            for name, flags, expected in [('on',[],True),('off',['--no-monitor-confirm-recent-three'],False)]:
                argv=['v2_collect','--session',name,'--layout','legacy','--role','development','--seed','7']+flags
                with patch('sys.argv',argv), contextlib.redirect_stdout(io.StringIO()):
                    v2_collect.main()
                plan=json.loads((Path(folder)/'results/v2_collection/plans'/f'{name}.json').read_text())
                self.assertIs(plan['monitor_confirm_recent_three'],expected)


if __name__=='__main__':
    unittest.main()
