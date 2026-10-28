import tempfile
import unittest
from pathlib import Path

from flash_v4.shared_node import prepare_training_environment
from flash_v4.train_entry import PRIVATE_ACTOR_WINDOW_ENV, PRIVATE_STREAM_DELAY_ENV


class PrivateActorEnvironmentTests(unittest.TestCase):
    def test_window_is_training_only_and_not_given_to_mps(self):
        with tempfile.TemporaryDirectory() as directory:
            library=Path(directory)/'actor.so';library.touch()
            env={PRIVATE_ACTOR_WINDOW_ENV:'{"first":40,"last":71}',PRIVATE_STREAM_DELAY_ENV:'{}'}
            control,training,enabled=prepare_training_environment(env,stream_delay_library=library)
            self.assertTrue(enabled)
            self.assertNotIn(PRIVATE_ACTOR_WINDOW_ENV,control)
            self.assertIn(PRIVATE_ACTOR_WINDOW_ENV,training)

    def test_window_without_actor_is_rejected_before_gpu_start(self):
        env={PRIVATE_ACTOR_WINDOW_ENV:'{"first":40,"last":71}'}
        with self.assertRaisesRegex(ValueError,'requires'):
            prepare_training_environment(env)
        self.assertNotIn(PRIVATE_ACTOR_WINDOW_ENV,env)


if __name__=='__main__':unittest.main()
