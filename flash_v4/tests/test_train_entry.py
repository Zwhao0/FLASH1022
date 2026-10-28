"""CPU-only contracts for the v4 torchrun entry adapter."""
from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from flash_v4 import train_entry


class EntryArgumentTests(unittest.TestCase):
    def test_healthy_arguments_forward_as_fixed_pp2_ep2(self):
        args = train_entry._parse_args([
            "--layout", "node_local_pp", "--output", "results/run-a",
            "--steps", "8", "--warmup", "3", "--seq-len", "16", "--seed", "19",
        ])
        forwarded = train_entry._v1_argv(args)
        self.assertIn("--tensor-parallel-size", forwarded)
        self.assertEqual(forwarded[forwarded.index("--tensor-parallel-size") + 1], "2")
        self.assertEqual(forwarded[forwarded.index("--expert-parallel-size") + 1], "2")
        self.assertEqual(forwarded[forwarded.index("--steps") + 1], "8")
        self.assertEqual(forwarded[forwarded.index("--warmup") + 1], "3")
        self.assertEqual(forwarded[forwarded.index("--seq-len") + 1], "16")
        self.assertEqual(forwarded[forwarded.index("--seed") + 1], "19")
        self.assertNotIn("--collection-plan", forwarded)
        self.assertNotIn("--online", forwarded)

    def test_collection_and_fault_entry_flags_are_rejected(self):
        for flag in ("--collection-plan", "--collection-plan=plan.json", "--online",
                     "--monitor-benchmark", "--compute-delay-ms", "--fault-rank"):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "disabled"):
                train_entry._parse_args([
                    "--layout", "legacy", "--output", "out", flag,
                ])

    def test_unsupported_future_arguments_are_not_forwarded(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                train_entry._parse_args([
                    "--layout", "legacy", "--output", "out", "--evidence-enabled",
                ])


class StartupEnvironmentTests(unittest.TestCase):
    def launch_env(self, rank: int = 3) -> dict[str, str]:
        return {
            "WORLD_SIZE": "16",
            "LOCAL_WORLD_SIZE": "8",
            "RANK": str(rank),
            "LOCAL_RANK": str(rank % 8),
        }

    def test_private_rank_quota_is_consumed_and_missing_ranks_get_full_share(self):
        target = self.launch_env(rank=3)
        target[train_entry.PRIVATE_QUOTA_ENV] = '{"3": 50}'
        with patch.object(train_entry, "_assert_cuda_not_initialized"):
            startup = train_entry._configure_process_environment("legacy", target)
        self.assertEqual((startup.rank, startup.local_rank), (3, 3))
        self.assertEqual(startup.requested_quota_percent, 50)
        self.assertIsNone(startup.actuation_dir)
        self.assertNotIn(train_entry.PRIVATE_QUOTA_ENV, target)
        self.assertEqual(target[train_entry.MPS_QUOTA_ENV], "50")
        self.assertEqual(target[train_entry.LAYOUT_ENV], "legacy")

        other = self.launch_env(rank=4)
        other[train_entry.PRIVATE_QUOTA_ENV] = '{"3": 50}'
        other[train_entry.MPS_QUOTA_ENV] = "17"
        with patch.object(train_entry, "_assert_cuda_not_initialized"):
            startup = train_entry._configure_process_environment("node_local_pp", other)
        self.assertNotIn(train_entry.PRIVATE_QUOTA_ENV, other)
        self.assertEqual(other[train_entry.MPS_QUOTA_ENV], "100")
        self.assertEqual(other[train_entry.LAYOUT_ENV], "node_local_pp")
        self.assertEqual(startup.requested_quota_percent, 100)

    def test_private_actuation_path_is_consumed_and_separate_from_training_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = self.launch_env(rank=3)
            private_dir = root / "controller-evidence"
            training_output = root / "training-output"
            env[train_entry.PRIVATE_ACTUATION_DIR_ENV] = str(private_dir)
            env[train_entry.PRIVATE_QUOTA_ENV] = '{"3": 100}'
            with patch.object(train_entry, "_assert_cuda_not_initialized"):
                startup = train_entry._configure_process_environment(
                    "legacy", env, training_output=training_output)
            self.assertNotIn(train_entry.PRIVATE_ACTUATION_DIR_ENV, env)
            self.assertNotIn(train_entry.PRIVATE_QUOTA_ENV, env)
            self.assertEqual(startup.actuation_dir, private_dir.resolve())

            overlap_env = self.launch_env(rank=3)
            overlap_env[train_entry.PRIVATE_ACTUATION_DIR_ENV] = str(training_output / "private")
            with patch.object(train_entry, "_assert_cuda_not_initialized"):
                with self.assertRaisesRegex(ValueError, "separate from training output"):
                    train_entry._configure_process_environment(
                        "legacy", overlap_env, training_output=training_output)
            self.assertNotIn(train_entry.PRIVATE_ACTUATION_DIR_ENV, overlap_env)

    def test_private_mapping_is_removed_even_when_invalid_and_is_not_echoed(self):
        env = self.launch_env()
        secret = "controller-only-invalid-payload"
        env[train_entry.PRIVATE_QUOTA_ENV] = secret
        with patch.object(train_entry, "_assert_cuda_not_initialized"):
            with self.assertRaises(ValueError) as caught:
                train_entry._configure_process_environment("legacy", env)
        self.assertNotIn(train_entry.PRIVATE_QUOTA_ENV, env)
        self.assertNotIn(secret, str(caught.exception))

    def test_stream_delay_mapping_is_consumed_and_uses_zero_ratio_shams(self):
        for rank, ratio in ((3, 0.5), (4, 0.0)):
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as temp:
                env = self.launch_env(rank=rank)
                env[train_entry.PRIVATE_STREAM_DELAY_ENV] = '{"3": 0.5}'
                env[train_entry.PRIVATE_ACTUATION_DIR_ENV] = temp
                env[train_entry.PRIVATE_QUOTA_ENV] = '{"3": 100}'
                with patch.object(train_entry, "_assert_cuda_not_initialized"):
                    startup = train_entry._configure_process_environment("legacy", env)
                self.assertTrue(startup.stream_delay_enabled)
                self.assertEqual(startup.stream_delay_ratio, ratio)
                self.assertEqual((startup.stream_delay_first, startup.stream_delay_last), (4, 11))
                self.assertEqual(startup.requested_quota_percent, 100)
                self.assertNotIn(train_entry.PRIVATE_STREAM_DELAY_ENV, env)
                self.assertNotIn(train_entry.PRIVATE_QUOTA_ENV, env)

    def test_private_actor_window_is_consumed_and_saved_as_bounded_fields(self):
        raw = '{"first":40,"last":71}'
        env = self.launch_env(rank=3)
        env[train_entry.PRIVATE_STREAM_DELAY_ENV] = "{}"
        env[train_entry.PRIVATE_ACTOR_WINDOW_ENV] = raw
        with tempfile.TemporaryDirectory() as temp:
            env[train_entry.PRIVATE_ACTUATION_DIR_ENV] = temp
            with patch.object(train_entry, "_assert_cuda_not_initialized"):
                startup = train_entry._configure_process_environment("legacy", env)
        self.assertEqual((startup.stream_delay_first, startup.stream_delay_last), (40, 71))
        self.assertNotIn(train_entry.PRIVATE_ACTOR_WINDOW_ENV, env)
        self.assertNotIn(raw, repr(startup))

    def test_private_actor_window_requires_actor_and_rejects_invalid_windows(self):
        no_actor = self.launch_env()
        no_actor[train_entry.PRIVATE_ACTOR_WINDOW_ENV] = '{"first":40,"last":71}'
        with patch.object(train_entry, "_assert_cuda_not_initialized"):
            with self.assertRaisesRegex(ValueError, "requires a stream-delay actor mapping") as caught:
                train_entry._configure_process_environment("legacy", no_actor)
        self.assertNotIn(train_entry.PRIVATE_ACTOR_WINDOW_ENV, no_actor)
        self.assertNotIn("40", str(caught.exception))

        invalid = (
            "not-json",
            '{"first":3,"last":10}',
            '{"first":40,"last":72}',
            '{"first":100,"last":121}',
            '{"first":41,"last":40}',
            '{"first":40.0,"last":41}',
            '{"first":true,"last":41}',
            '{"first":40}',
            '{"first":40,"last":41,"extra":1}',
        )
        for raw in invalid:
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as temp:
                env = self.launch_env()
                env[train_entry.PRIVATE_STREAM_DELAY_ENV] = "{}"
                env[train_entry.PRIVATE_ACTUATION_DIR_ENV] = temp
                env[train_entry.PRIVATE_ACTOR_WINDOW_ENV] = raw
                with patch.object(train_entry, "_assert_cuda_not_initialized"):
                    with self.assertRaises(ValueError) as caught:
                        train_entry._configure_process_environment("legacy", env)
                self.assertNotIn(raw, str(caught.exception))
                self.assertNotIn(train_entry.PRIVATE_ACTOR_WINDOW_ENV, env)

    def test_empty_stream_delay_mapping_enables_all_rank_sham(self):
        env = self.launch_env(rank=7)
        env[train_entry.PRIVATE_STREAM_DELAY_ENV] = "{}"
        with tempfile.TemporaryDirectory() as temp:
            env[train_entry.PRIVATE_ACTUATION_DIR_ENV] = temp
            with patch.object(train_entry, "_assert_cuda_not_initialized"):
                startup = train_entry._configure_process_environment("legacy", env)
        self.assertTrue(startup.stream_delay_enabled)
        self.assertEqual(startup.stream_delay_ratio, 0.0)

    def test_stream_delay_requires_private_path_and_full_mps_quota(self):
        no_path = self.launch_env()
        no_path[train_entry.PRIVATE_STREAM_DELAY_ENV] = "{}"
        with patch.object(train_entry, "_assert_cuda_not_initialized"):
            with self.assertRaisesRegex(ValueError, "private actuation directory"):
                train_entry._configure_process_environment("legacy", no_path)
        self.assertNotIn(train_entry.PRIVATE_STREAM_DELAY_ENV, no_path)

        low_quota = self.launch_env()
        low_quota[train_entry.PRIVATE_STREAM_DELAY_ENV] = '{"3": 0.5}'
        low_quota[train_entry.PRIVATE_QUOTA_ENV] = '{"3": 50}'
        with tempfile.TemporaryDirectory() as temp:
            low_quota[train_entry.PRIVATE_ACTUATION_DIR_ENV] = temp
            with patch.object(train_entry, "_assert_cuda_not_initialized"):
                with self.assertRaisesRegex(ValueError, "MPS quota 100"):
                    train_entry._configure_process_environment("legacy", low_quota)

    def test_stream_delay_rejects_invalid_ratios_without_echoing_mapping(self):
        for raw in ('{"3": -0.1}', '{"3": 8.1}', '{"3": NaN}', '{"16": 0.5}'):
            env = self.launch_env()
            env[train_entry.PRIVATE_STREAM_DELAY_ENV] = raw
            with tempfile.TemporaryDirectory() as temp:
                env[train_entry.PRIVATE_ACTUATION_DIR_ENV] = temp
                with patch.object(train_entry, "_assert_cuda_not_initialized"):
                    with self.assertRaises(ValueError) as caught:
                        train_entry._configure_process_environment("legacy", env)
            self.assertNotIn(raw, str(caught.exception))
            self.assertNotIn(train_entry.PRIVATE_STREAM_DELAY_ENV, env)

    def test_strong_development_ratio_has_a_finite_hard_limit(self):
        self.assertEqual(train_entry._parse_stream_delay_mapping('{"8": 8.0}'), {8: 8.0})
        for raw in ('{"8": 8.01}', '{"8": Infinity}', '{"8": true}'):
            with self.assertRaises(ValueError):
                train_entry._parse_stream_delay_mapping(raw)

    def test_world_and_local_world_sizes_are_fixed(self):
        env = self.launch_env()
        env["WORLD_SIZE"] = "8"
        with patch.object(train_entry, "_assert_cuda_not_initialized"):
            with self.assertRaisesRegex(ValueError, "WORLD_SIZE=16"):
                train_entry._configure_process_environment("legacy", env)

    def test_legacy_probe_preload_is_rejected(self):
        env = self.launch_env()
        env["LD_PRELOAD"] = "/old/build/libflashprobe.so"
        with patch.object(train_entry, "_assert_cuda_not_initialized"):
            with self.assertRaisesRegex(ValueError, "legacy FLASH LD_PRELOAD"):
                train_entry._configure_process_environment("legacy", env)

    def test_quota_mapping_rejects_bad_rank_and_out_of_range_value(self):
        for value in ('{"16": 50}', '{"3": 0}', '{"3": 50.5}', '[50]'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                train_entry._parse_quota_mapping(value)


class RuntimeAdapterTests(unittest.TestCase):
    def test_noop_delay_does_not_inspect_model_or_register_hooks(self):
        class PoisonModel:
            @property
            def model(self):
                raise AssertionError("no-op delay must not inspect modules")

        delay = train_entry.NoOpGPUStreamDelay(None, None, None)
        model = PoisonModel()
        delay.attach_backward(model)
        delay.begin_iteration(1)
        delay.apply(1)
        delay.set_profile_enabled(False)
        delay.close(2)
        self.assertIsNone(delay.interval)
        self.assertEqual(delay.cycles, 0)

    def test_null_redis_writer_has_close_compatible_resources(self):
        writer = train_entry.LocalNullRedisWriter("unused-host", 12345)
        writer.write(1, 2.0)
        writer.reader.close()
        writer.connection.close()
        writer.close()

    def test_training_module_is_patched_before_pipeline_import(self):
        calls = []

        class FakeCuda:
            def set_device(self, device):
                calls.append(("set_device", device))

            def set_per_process_memory_fraction(self, fraction, device=None):
                calls.append(("memory_fraction", fraction, device))

        training_module = SimpleNamespace(
            RedisWriter=object,
            GPUStreamDelay=object,
            torch=SimpleNamespace(cuda=FakeCuda()),
        )
        startup = train_entry.ProcessStartup(3, 3, 100, None)
        train_entry._patch_v1_training_module(training_module, startup)
        self.assertIs(training_module.RedisWriter, train_entry.LocalNullRedisWriter)
        self.assertIs(training_module.GPUStreamDelay, train_entry.NoOpGPUStreamDelay)
        training_module.torch.cuda.set_device(5)
        self.assertEqual(calls, [
            ("set_device", 5),
            ("memory_fraction", 0.03, 5),
        ])

    def test_private_evidence_is_written_once_after_device_and_allocator_setup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence_dir = root / "controller-evidence"
            training_output = root / "training-output"
            training_output.mkdir()
            events = []

            class FakeCuda:
                def set_device(self, device):
                    events.append(("set_device", device))

                def set_per_process_memory_fraction(self, fraction, device=None):
                    events.append(("memory_fraction", fraction, device))

                def get_device_properties(self, device):
                    events.append(("device_properties", device))
                    return SimpleNamespace(
                        multi_processor_count=80,
                        uuid="GPU-test-uuid",
                    )

                def is_initialized(self):
                    return True

            torch_module = SimpleNamespace(cuda=FakeCuda())
            startup = train_entry.ProcessStartup(3, 3, 50, evidence_dir)
            train_entry._install_allocator_budget(torch_module, startup)
            torch_module.cuda.set_device(3)
            torch_module.cuda.set_device(3)

            record = json.loads((evidence_dir / "rank_3.json").read_text())
            self.assertEqual(record["rank"], 3)
            self.assertEqual(record["local_device"], 3)
            self.assertEqual(record["multi_processor_count"], 80)
            self.assertEqual(record["device_uuid"], "GPU-test-uuid")
            self.assertEqual(record["requested_mps_quota_percent"], 50)
            self.assertIsNone(record["effective_mps_quota_percent"])
            self.assertFalse(record["effective_quota_verified"])
            self.assertIn("captured_at_unix", record)
            self.assertFalse(any(training_output.iterdir()))
            self.assertEqual(events, [
                ("set_device", 3),
                ("memory_fraction", 0.03, 3),
                ("device_properties", 3),
                ("set_device", 3),
                ("memory_fraction", 0.03, 3),
            ])

    def test_missing_uuid_is_allowed_but_missing_sm_count_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            startup = train_entry.ProcessStartup(0, 0, 100, Path(temp) / "private")
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(
                get_device_properties=lambda device: SimpleNamespace(multi_processor_count=16),
                is_initialized=lambda: True,
            ))
            train_entry._write_actuation_evidence(startup, fake_torch, 0)
            record = json.loads((startup.actuation_dir / "rank_0.json").read_text())
            self.assertIsNone(record["device_uuid"])

            bad = train_entry.ProcessStartup(1, 1, 100, Path(temp) / "private-bad")
            bad_torch = SimpleNamespace(cuda=SimpleNamespace(
                get_device_properties=lambda device: SimpleNamespace(uuid="GPU-no-sm"),
            ))
            with self.assertRaisesRegex(RuntimeError, "device properties"):
                train_entry._write_actuation_evidence(bad, bad_torch, 1)

    def test_allocator_budget_is_process_local_and_recorded_without_quota(self):
        with tempfile.TemporaryDirectory() as temp:
            train_entry._write_runtime_metadata(temp, "legacy")
            record = json.loads((Path(temp) / train_entry.RUNTIME_METADATA).read_text())
        budget = record["shared_compatibility_resource_budget"]
        self.assertEqual(budget["pytorch_allocator_fraction"], 0.03)
        self.assertEqual(budget["scope"], "this_process_only")
        self.assertNotIn("quota", json.dumps(budget))

    def test_preimported_pipeline_is_rejected(self):
        startup = train_entry.ProcessStartup(0, 0, 100, None)
        with patch.dict(sys.modules, {"flash_v1.train_pipeline": object()}):
            with self.assertRaisesRegex(RuntimeError, "imported before"):
                train_entry._load_v1_train_pipeline(startup)

    def test_metadata_records_adapter_types_without_quota_or_fault_values(self):
        with tempfile.TemporaryDirectory() as temp:
            train_entry._write_runtime_metadata(temp, "legacy")
            record = json.loads((Path(temp) / train_entry.RUNTIME_METADATA).read_text())
        self.assertEqual(record["layout"], "legacy")
        self.assertIn("local_null_redis_writer", record["runtime_adaptations"])
        self.assertIn("no_op_gpu_stream_delay", record["runtime_adaptations"])
        serialized = json.dumps(record)
        self.assertNotIn("quota_percent", serialized)
        self.assertNotIn("fault_rank", serialized)
        self.assertNotIn("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", serialized)

    def test_best_effort_exit_synchronizes_only_an_initialized_cuda_context(self):
        calls = []
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            is_initialized=lambda: True,
            synchronize=lambda: calls.append("sync"),
        ))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            train_entry._best_effort_cuda_synchronize()
        self.assertEqual(calls, ["sync"])


if __name__ == "__main__":
    unittest.main()
