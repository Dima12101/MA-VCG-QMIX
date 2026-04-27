"""Smoke tests for the chapter 6 benchmark pipeline."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import pandas as pd

from src.config import (
    AUCTION_CONFIG,
    ENV_CONFIG,
    NODE_CONFIG,
    TASK_CONFIG,
    TRAINING_CONFIG,
)
from src.learning.benchmark import BenchmarkRunner, MethodSpec, ScenarioSpec
from visualization.plot_results import ResultsVisualizer


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

            summary_by_seed = pd.read_csv(Path(tmp_dir) / "summary_by_seed.csv")
            seeds_by_method = summary_by_seed.groupby("method")["seed"].unique().to_dict()
            self.assertEqual(
                {tuple(seeds) for seeds in seeds_by_method.values()},
                {(7,)},
            )

    def test_visualizer_exports_all_tied_winners(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dir = Path(tmp_dir)
            summary = pd.DataFrame(
                [
                    {
                        "scenario": "baseline",
                        "scenario_label": "Сценарий 1",
                        "method": "vcg",
                        "method_label": "MA-VCG",
                        "mean_social_welfare": 3.5,
                        "mean_social_welfare_std": 0.2,
                        "mean_acceptance_rate": 60.0,
                        "mean_acceptance_rate_std": 1.0,
                        "mean_deadline_success_rate": 0.8,
                        "mean_deadline_success_rate_std": 0.01,
                        "mean_avg_latency": 150.0,
                        "mean_avg_latency_std": 5.0,
                        "mean_gini_payment": 0.0,
                        "mean_gini_payment_std": 0.0,
                        "mean_fairness_index": 0.25,
                        "mean_fairness_index_std": 0.01,
                    },
                    {
                        "scenario": "baseline",
                        "scenario_label": "Сценарий 1",
                        "method": "heuristic",
                        "method_label": "Heuristic-LoadAware",
                        "mean_social_welfare": 3.5,
                        "mean_social_welfare_std": 0.2,
                        "mean_acceptance_rate": 60.0,
                        "mean_acceptance_rate_std": 1.0,
                        "mean_deadline_success_rate": 0.8,
                        "mean_deadline_success_rate_std": 0.01,
                        "mean_avg_latency": 150.0,
                        "mean_avg_latency_std": 5.0,
                        "mean_gini_payment": 0.0,
                        "mean_gini_payment_std": 0.0,
                        "mean_fairness_index": 0.25,
                        "mean_fairness_index_std": 0.01,
                    },
                    {
                        "scenario": "baseline",
                        "scenario_label": "Сценарий 1",
                        "method": "qmix",
                        "method_label": "QMIX",
                        "mean_social_welfare": 3.1,
                        "mean_social_welfare_std": 0.4,
                        "mean_acceptance_rate": 58.0,
                        "mean_acceptance_rate_std": 2.0,
                        "mean_deadline_success_rate": 0.76,
                        "mean_deadline_success_rate_std": 0.03,
                        "mean_avg_latency": 148.0,
                        "mean_avg_latency_std": 7.0,
                        "mean_gini_payment": 0.0,
                        "mean_gini_payment_std": 0.0,
                        "mean_fairness_index": 0.28,
                        "mean_fairness_index_std": 0.02,
                    },
                ]
            )
            summary.to_csv(results_dir / "summary.csv", index=False)

            visualizer = ResultsVisualizer(results_dir=results_dir)
            visualizer.export_tables()

            winners = pd.read_csv(results_dir / "tables" / "scenario_winners.csv")
            self.assertEqual(
                winners.loc[0, "Лидирующий метод(ы) по SW"],
                "Heuristic-LoadAware / MA-VCG",
            )


if __name__ == "__main__":
    unittest.main()
