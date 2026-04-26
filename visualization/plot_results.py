"""Chapter-ready plotting and export utilities for the validation benchmark."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.learning.benchmark import BenchmarkRunner


sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["legend.fontsize"] = 11


class ResultsVisualizer:
    """Build plots and tables for the experimental validation chapter."""

    def __init__(
        self,
        results_dir: str = "experiments/results/validation",
        dissertation_root: str | Path | None = None,
    ):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.tables_dir = self.results_dir / "tables"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        self.dissertation_root = Path(dissertation_root) if dissertation_root else None
        self.dissertation_images_dir = None
        self.dissertation_tables_dir = None
        if self.dissertation_root is not None:
            self.dissertation_images_dir = self.dissertation_root / "Dissertation" / "images"
            self.dissertation_tables_dir = (
                self.dissertation_root / "Dissertation" / "tables" / "validation"
            )
            self.dissertation_images_dir.mkdir(parents=True, exist_ok=True)
            self.dissertation_tables_dir.mkdir(parents=True, exist_ok=True)

        self.palette = {
            "MA-VCG": "#2563eb",
            "Heuristic-LoadAware": "#dc2626",
            "Random": "#6b7280",
            "QMIX": "#f59e0b",
            "MA-VCG-QMIX": "#059669",
        }
        self.scenario_specs = {spec.name: spec for spec in BenchmarkRunner.default_scenarios()}

    def _save_figure(self, stem: str):
        saved_paths = []
        for suffix in ("png", "pdf"):
            output_path = self.plots_dir / f"{stem}.{suffix}"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            saved_paths.append(output_path)
        if self.dissertation_images_dir is not None:
            for path in saved_paths:
                shutil.copy2(path, self.dissertation_images_dir / path.name)

    def _write_latex_table(self, dataframe: pd.DataFrame, filename: str):
        output_path = self.tables_dir / filename
        dataframe.to_latex(output_path, index=False, escape=False)
        if self.dissertation_tables_dir is not None:
            shutil.copy2(output_path, self.dissertation_tables_dir / filename)

    def load_summary(self) -> pd.DataFrame:
        """Load the aggregated benchmark summary table."""
        summary_path = self.results_dir / "summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(
                "Файл summary.csv не найден. Сначала запустите benchmark-пайплайн."
            )
        return pd.read_csv(summary_path)

    def load_summary_by_seed(self) -> pd.DataFrame:
        """Load the seed-level experiment summaries."""
        summary_path = self.results_dir / "summary_by_seed.csv"
        if not summary_path.exists():
            raise FileNotFoundError(
                "Файл summary_by_seed.csv не найден. Сначала запустите benchmark-пайплайн."
            )
        return pd.read_csv(summary_path)

    def load_episode_summary(self, phase: str = "train") -> pd.DataFrame:
        """Load episode-level aggregates for the selected phase."""
        files = sorted(self.results_dir.glob(f"*/*/seed_*/{phase}/episode_summary.csv"))
        if not files:
            raise FileNotFoundError(f"Файлы {phase}/episode_summary.csv не найдены.")
        return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)

    def load_step_records(
        self,
        scenario: str | None = None,
        method: str | None = None,
        phase: str = "eval",
    ) -> pd.DataFrame:
        """Load step-level records for the selected scope."""
        scenario_glob = scenario if scenario is not None else "*"
        method_glob = method if method is not None else "*"
        files = sorted(
            self.results_dir.glob(
                f"{scenario_glob}/{method_glob}/seed_*/{phase}/episodes/episode_*.csv"
            )
        )
        if not files:
            raise FileNotFoundError(
                f"Не найдены step-level результаты для {scenario_glob}/{method_glob}/{phase}."
            )
        return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)

    @staticmethod
    def _format_mean_pm_std(mean_value: float, std_value: float, precision: int = 3) -> str:
        return f"{mean_value:.{precision}f} $\\pm$ {std_value:.{precision}f}"

    def export_tables(self):
        """Export summary tables in CSV and LaTeX formats."""
        summary = self.load_summary().sort_values(["scenario", "method"])

        chapter_table = pd.DataFrame(
            {
                "Сценарий": summary["scenario_label"],
                "Метод": summary["method_label"],
                "SW": [
                    self._format_mean_pm_std(mean_value, std_value)
                    for mean_value, std_value in zip(
                        summary["mean_social_welfare"],
                        summary["mean_social_welfare_std"],
                    )
                ],
                "Принятие задач, \\%": [
                    self._format_mean_pm_std(mean_value, std_value, precision=2)
                    for mean_value, std_value in zip(
                        summary["mean_acceptance_rate"],
                        summary["mean_acceptance_rate_std"],
                    )
                ],
                "До дедлайна": [
                    self._format_mean_pm_std(mean_value, std_value, precision=3)
                    for mean_value, std_value in zip(
                        summary["mean_deadline_success_rate"],
                        summary["mean_deadline_success_rate_std"],
                    )
                ],
                "Задержка, мс": [
                    self._format_mean_pm_std(mean_value, std_value, precision=2)
                    for mean_value, std_value in zip(
                        summary["mean_avg_latency"],
                        summary["mean_avg_latency_std"],
                    )
                ],
                "Джини": [
                    self._format_mean_pm_std(mean_value, std_value)
                    for mean_value, std_value in zip(
                        summary["mean_gini_payment"],
                        summary["mean_gini_payment_std"],
                    )
                ],
                "Fairness": [
                    self._format_mean_pm_std(mean_value, std_value)
                    for mean_value, std_value in zip(
                        summary["mean_fairness_index"],
                        summary["mean_fairness_index_std"],
                    )
                ],
            }
        )
        chapter_table.to_csv(self.tables_dir / "summary_table.csv", index=False)
        self._write_latex_table(chapter_table, "summary_table.tex")

        winners = (
            summary.sort_values("mean_social_welfare", ascending=False)
            .groupby("scenario_label", as_index=False)
            .first()[["scenario_label", "method_label", "mean_social_welfare", "mean_social_welfare_std"]]
        )
        winners = winners.rename(
            columns={
                "scenario_label": "Сценарий",
                "method_label": "Лучший метод по SW",
            }
        )
        winners["SW"] = [
            self._format_mean_pm_std(mean_value, std_value)
            for mean_value, std_value in zip(
                winners["mean_social_welfare"],
                winners["mean_social_welfare_std"],
            )
        ]
        winners = winners[["Сценарий", "Лучший метод по SW", "SW"]]
        winners.to_csv(self.tables_dir / "scenario_winners.csv", index=False)
        self._write_latex_table(winners, "scenario_winners.tex")

        for scenario_name, scenario_df in summary.groupby("scenario", sort=False):
            scenario_table = chapter_table.loc[summary["scenario"] == scenario_name].reset_index(drop=True)
            scenario_table.to_csv(self.tables_dir / f"{scenario_name}_table.csv", index=False)
            self._write_latex_table(scenario_table, f"{scenario_name}_table.tex")

        load_spikes = self._build_load_spike_window_table()
        load_spikes.to_csv(self.tables_dir / "load_spikes_windows.csv", index=False)
        self._write_latex_table(load_spikes, "load_spikes_windows.tex")

    def _annotate_event_windows(self, axes, scenario_name: str):
        scenario = self.scenario_specs[scenario_name]
        windows = list(scenario.env_config.load_spike_windows)
        if windows:
            for ax in axes:
                for start_step, end_step in windows:
                    ax.axvspan(start_step, end_step, color="#fef3c7", alpha=0.4)
        if scenario.env_config.failure_start_step > 0:
            failure_end = (
                scenario.env_config.failure_start_step
                + scenario.env_config.failure_recovery_steps
            )
            for ax in axes:
                ax.axvspan(
                    scenario.env_config.failure_start_step,
                    failure_end,
                    color="#fee2e2",
                    alpha=0.35,
                )

    def plot_method_overview(self):
        """Grouped bar charts with confidence intervals for key evaluation metrics."""
        summary_by_seed = self.load_summary_by_seed()
        metrics = [
            ("mean_social_welfare", "Социальное благосостояние"),
            ("mean_deadline_success_rate", "Доля завершений до дедлайна"),
            ("mean_avg_latency", "Средняя задержка, мс"),
            ("mean_gini_payment", "Джини платежей"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(20, 13))
        axes = axes.flatten()
        for ax, (metric, title) in zip(axes, metrics):
            sns.barplot(
                data=summary_by_seed,
                x="scenario_label",
                y=metric,
                hue="method_label",
                palette=self.palette,
                errorbar=("ci", 95),
                capsize=0.08,
                ax=ax,
            )
            ax.set_title(title)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=15)
            if metric == "mean_gini_payment":
                ax.axhline(0.3, color="#7c3aed", linestyle="--", linewidth=1.5)
        handles, labels = axes[0].get_legend_handles_labels()
        for ax in axes:
            if ax.legend_:
                ax.legend_.remove()
        fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
        # fig.suptitle("Сравнение методов по ключевым метрикам (95\\% CI)", y=1.02)
        plt.tight_layout()
        self._save_figure("method_overview")
        plt.close()

    def plot_social_welfare_boxplot(self):
        """Boxplot over the seed-level welfare distribution."""
        summary_by_seed = self.load_summary_by_seed()
        plt.figure(figsize=(18, 8))
        sns.boxplot(
            data=summary_by_seed,
            x="scenario_label",
            y="mean_social_welfare",
            hue="method_label",
            palette=self.palette,
        )
        plt.xlabel("")
        plt.ylabel("Социальное благосостояние")
        # plt.title("Распределение SW по независимым запускам")
        plt.xticks(rotation=15)
        plt.tight_layout()
        self._save_figure("social_welfare_boxplot")
        plt.close()

    def plot_learning_curves(self):
        """Episode-level learning curves for the trainable methods."""
        episode_summary = self.load_episode_summary(phase="train")
        episode_summary = episode_summary[episode_summary["method_label"].isin(["QMIX", "MA-VCG-QMIX"])]
        metrics = [
            ("mean_social_welfare", "Социальное благосостояние"),
            ("mean_acceptance_rate", "Принятие задач, %"),
            ("mean_reward", "Среднее вознаграждение"),
            ("mean_td_error", "TD-ошибка"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(20, 13))
        axes = axes.flatten()
        for ax, (metric, title) in zip(axes, metrics):
            sns.lineplot(
                data=episode_summary,
                x="episode",
                y=metric,
                hue="method_label",
                style="scenario_label",
                palette=self.palette,
                errorbar=("ci", 95),
                linewidth=2.2,
                ax=ax,
            )
            ax.set_title(title)
            ax.set_xlabel("Эпизод обучения")
        handles, labels = axes[0].get_legend_handles_labels()
        for ax in axes:
            if ax.legend_:
                ax.legend_.remove()
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
        fig.suptitle("Кривые обучения QMIX и MA-VCG-QMIX", y=1.02)
        plt.tight_layout()
        self._save_figure("learning_curves")
        plt.close()

    def plot_step_dynamics(self):
        """Step-level evaluation dynamics for each scenario."""
        summary_by_seed = self.load_summary_by_seed()
        plots = [
            ("social_welfare", "Социальное благосостояние"),
            ("acceptance_rate", "Принятие задач, %"),
            ("avg_latency", "Средняя задержка, мс"),
            ("deadline_success_rate", "Доля завершений до дедлайна"),
        ]
        for scenario_name in summary_by_seed["scenario"].unique():
            step_df = self.load_step_records(scenario=scenario_name, phase="eval")
            fig, axes = plt.subplots(4, 1, figsize=(18, 18), sharex=True)
            for ax, (metric, title) in zip(axes, plots):
                sns.lineplot(
                    data=step_df,
                    x="step",
                    y=metric,
                    hue="method_label",
                    palette=self.palette,
                    errorbar=("ci", 95),
                    linewidth=2.2,
                    ax=ax,
                )
                ax.set_title(title)
                ax.set_xlabel("Шаг симуляции")
                if metric == "avg_latency":
                    ax.set_ylim(bottom=0)
            self._annotate_event_windows(axes, scenario_name)
            handles, labels = axes[0].get_legend_handles_labels()
            for ax in axes:
                if ax.legend_:
                    ax.legend_.remove()
            fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
            scenario_label = self.scenario_specs[scenario_name].label
            # fig.suptitle(f"Пошаговая динамика метрик: {scenario_label}", y=1.01)
            plt.tight_layout()
            self._save_figure(f"{scenario_name}_dynamics")
            plt.close()

    def plot_load_heatmaps(self, preferred_method: str = "hybrid"):
        """Heatmaps of node load for the preferred method in each scenario."""
        summary_by_seed = self.load_summary_by_seed()
        for scenario_name in summary_by_seed["scenario"].unique():
            step_df = self.load_step_records(scenario=scenario_name, method=preferred_method, phase="eval")
            load_cols = [column for column in step_df.columns if column.startswith("load_node_")]
            if not load_cols:
                continue
            heatmap_data = step_df.groupby("step", as_index=True)[load_cols].mean().T
            heatmap_data.index = [index.replace("load_node_", "Узел ") for index in heatmap_data.index]
            plt.figure(figsize=(18, 6))
            sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws={"label": "Нагрузка"})
            scenario_label = self.scenario_specs[scenario_name].label
            # plt.title(f"Распределение нагрузки по узлам: {scenario_label} (MA-VCG-QMIX)")
            plt.xlabel("Шаг симуляции")
            plt.ylabel("")
            plt.tight_layout()
            self._save_figure(f"{scenario_name}_{preferred_method}_load_heatmap")
            plt.close()

    def plot_fairness_welfare_scatter(self):
        """Scatter plot showing the welfare/fairness trade-off."""
        summary_by_seed = self.load_summary_by_seed()
        plt.figure(figsize=(13, 9))
        sns.scatterplot(
            data=summary_by_seed,
            x="mean_gini_payment",
            y="mean_social_welfare",
            hue="method_label",
            style="scenario_label",
            palette=self.palette,
            s=180,
        )
        plt.axvline(0.3, color="#7c3aed", linestyle="--", linewidth=1.5)
        plt.xlabel("Коэффициент Джини платежей")
        plt.ylabel("Социальное благосостояние")
        # plt.title("Компромисс между справедливостью и эффективностью")
        plt.tight_layout()
        self._save_figure("fairness_welfare_scatter")
        plt.close()

    def _build_load_spike_window_table(self) -> pd.DataFrame:
        """Aggregate the load spike scenario into before/peak/after windows."""
        scenario = self.scenario_specs["load_spikes"]
        step_df = self.load_step_records(scenario="load_spikes", phase="eval")
        first_start = scenario.env_config.load_spike_windows[0][0]
        last_end = scenario.env_config.load_spike_windows[-1][1]

        def classify_period(step_value: int) -> str:
            if step_value < first_start:
                return "До пика"
            if any(start <= step_value <= end for start, end in scenario.env_config.load_spike_windows):
                return "Пик"
            if step_value > last_end:
                return "После пика"
            return "Между пиками"

        step_df["period"] = step_df["step"].map(classify_period)
        period_df = (
            step_df.groupby(["method_label", "period"], as_index=False)[
                [
                    "social_welfare",
                    "acceptance_rate",
                    "avg_latency",
                    "deadline_success_rate",
                ]
            ]
            .mean()
        )
        period_df = period_df.rename(
            columns={
                "method_label": "Метод",
                "period": "Интервал",
                "social_welfare": "SW",
                "acceptance_rate": "Принятие задач, \\%",
                "avg_latency": "Задержка, мс",
                "deadline_success_rate": "До дедлайна",
            }
        )
        for column in ["SW", "Принятие задач, \\%", "Задержка, мс", "До дедлайна"]:
            period_df[column] = period_df[column].map(lambda value: f"{value:.3f}")
        return period_df

    def build_all(self):
        """Build the complete chapter artifact set."""
        self.export_tables()
        self.plot_method_overview()
        self.plot_social_welfare_boxplot()
        self.plot_learning_curves()
        self.plot_step_dynamics()
        self.plot_load_heatmaps()
        self.plot_fairness_welfare_scatter()


if __name__ == "__main__":
    ResultsVisualizer().build_all()
