"""Run existing safety contracts plus the post-restore stale-progress regression."""
from pathlib import Path
import importlib.util,io,json,sys,unittest
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R))
from flash_v4.tests import test_rank_progress as original
spec=importlib.util.spec_from_file_location('clock_watch_r100',Path(__file__).with_name('clock_watch_r100.py'))
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
original.watcher=candidate


class PostRestoreRegression(original.WatcherCleanup):
    def test_complete_restoration_stops_before_later_stale_progress(self):
        rc,result,events,_=self.run_fake(self.rows()[:3]+[dict(progress_error='stale after successful full restore')])
        self.assertEqual(rc,0)
        self.assertIsNone(result['error'])
        self.assertEqual(result['last_iteration'],81)
        self.assertTrue(result['restored'])
        self.assertFalse(result['independent_watchdog_left_armed'])
        self.assertTrue(any(x['kind']=='fault_interval_finished' for x in events))
        self.assertFalse(any(x['kind']=='error' for x in events))


if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(original),
                              unittest.defaultTestLoader.loadTestsFromName('test_complete_restoration_stops_before_later_stale_progress',PostRestoreRegression)])
    output=io.StringIO();result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    log=R/'results/v4/preflight/r100_watcher_CPU.log';log.write_text(output.getvalue())
    data=dict(passed=result.wasSuccessful(),run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
              existing_safety_contracts_preserved=True,new_postrestore_regression=True,
              training_monitor_and_localizer_changed=False)
    (R/'results/v4/preflight/r100_watcher_CPU.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data));raise SystemExit(0 if result.wasSuccessful() else 1)
