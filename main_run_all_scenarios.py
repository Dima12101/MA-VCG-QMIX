"""Reproducible pipeline for the chapter 6 benchmark suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.learning.benchmark import BenchmarkRunner
from visualization.plot_results import ResultsVisualizer


def _default_dissertation_root() -> Path | None:
    repo_root = Path(__file__).resolve().parent
    candidate = repo_root.parent.parent / "диссертация" / "SPbU-Phd-LaTeX-Dissertation"
    return candidate if candidate.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full chapter 6 benchmark suite and build dissertation artifacts."
    )
    parser.add_argument(
        "--results-dir",
        default="experiments/results/chapter6",
        help="Directory for raw benchmark outputs and aggregated artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for the benchmark suite.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=5,
        help="Number of independent seed runs per scenario and method.",
    )
    parser.add_argument(
        "--dissertation-root",
        default=str(_default_dissertation_root()) if _default_dissertation_root() else None,
        help="Optional path to the dissertation repository root for automatic artifact sync.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dissertation_root = Path(args.dissertation_root) if args.dissertation_root else None

    print("\n" + "=" * 88)
    print("CHAPTER 6 BENCHMARK PIPELINE")
    print("=" * 88)
    print(f"Results directory : {args.results_dir}")
    print(f"Base seed         : {args.seed}")
    print(f"Seed runs         : {args.num_seeds}")
    print(
        "Dissertation sync : "
        + (str(dissertation_root) if dissertation_root is not None else "disabled")
    )
    print("=" * 88 + "\n")

    runner = BenchmarkRunner(results_root=args.results_dir)
    summary = runner.run_suite(seed=args.seed, num_seeds=args.num_seeds)

    visualizer = ResultsVisualizer(
        results_dir=args.results_dir,
        dissertation_root=dissertation_root,
    )
    visualizer.build_all()

    print("Итоговая сводка по сценариям и методам:")
    print(
        summary[
            [
                "scenario_label",
                "method_label",
                "mean_social_welfare",
                "mean_avg_latency",
                "mean_deadline_success_rate",
            ]
        ].to_string(index=False)
    )
    print(f"\nАртефакты сохранены в: {Path(args.results_dir).resolve()}")
    if dissertation_root is not None:
        print(f"Таблицы и изображения синхронизированы в: {dissertation_root}")


if __name__ == "__main__":
    main()
