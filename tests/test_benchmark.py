"""Smoke tests for the chapter 6 benchmark pipeline."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.config import (
    AUCTION_CONFIG,
    ENV_CONFIG,
    NODE_CONFIG,
    TASK_CONFIG,
    TRAINING_CONFIG,
)
from src.learning.benchmark import BenchmarkRunner, MethodSpec, ScenarioSpec


class TestBenchmarkRunner(unittest.TestCase):
    """Check that the benchmark suite produces the expected artifact skeleton."""

    def test_run_suite_exports_seed_and_aggregate_summaries(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = BenchmarkRunner(results_root=tmp_dir)
            scenario = ScenarioSpec(
                name="smoke",
                label="Smoke",
                env_config=replace(ENV_CONFIG, num_nodes=2, num_devices=3, task_lambda_arrival=1.5),
                node_config=NODE_CONFIG,
                task_config=TASK_CONFIG,
                training_config=replace(
                    TRAINING_CONFIG,
                    num_episodes=1,
                    max_steps_per_episode=4,
                    batch_size=1,
                    buffer_size=8,
                    target_update_freq=1,
                ),
                evaluation_episodes=1,
                description="Минимальный smoke-сценарий.",
            )
            methods = [
                MethodSpec(
                    name="vcg",
                    label="MA-VCG",
                    auction_config=replace(AUCTION_CONFIG, vcg_weight=0.0, global_reward_weight=0.0),
                    learning_enabled=False,
                    fixed_policy="always_accept",
                ),
                MethodSpec(
                    name="hybrid",
                    label="MA-VCG-QMIX",
                    auction_config=replace(AUCTION_CONFIG, vcg_weight=0.5, global_reward_weight=0.2),
                    learning_enabled=True,
                ),
            ]

            summary = runner.run_suite(
                scenarios=[scenario],
                methods=methods,
                seed=7,
                num_seeds=1,
            )

            self.assertEqual(set(summary["method"]), {"vcg", "hybrid"})
            self.assertIn("mean_deadline_success_rate", summary.columns)
            self.assertIn("mean_social_welfare_ci95", summary.columns)
            self.assertTrue((Path(tmp_dir) / "summary.csv").exists())
            self.assertTrue((Path(tmp_dir) / "summary_by_seed.csv").exists())


if __name__ == "__main__":
    unittest.main()
