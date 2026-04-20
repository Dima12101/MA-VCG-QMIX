"""Chapter-ready plotting utilities for the validation benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["legend.fontsize"] = 11


class ResultsVisualizer:
    """Build plots and tables for the experimental validation chapter."""

    def __init__(self, results_dir: str = "experiments/results/validation"):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.tables_dir = self.results_dir / "tables"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        self.palette = {
            "MA-VCG": "#2563eb",
            "QMIX": "#f59e0b",
            "MA-VCG-QMIX": "#059669",
        }

    def _save_figure(self, stem: str):
        for suffix in ("png", "pdf"):
            plt.savefig(
                self.plots_dir / f"{stem}.{suffix}",
                dpi=300,
                bbox_inches="tight",
            )

    def load_summary(self) -> pd.DataFrame:
        """Load the benchmark summary table."""
        summary_path = self.results_dir / "summary.csv"
        if summary_path.exists():
            return pd.read_csv(summary_path)

        summary_files = sorted(self.results_dir.glob("*/*/summary.csv"))
        if not summary_files:
            raise FileNotFoundError(
                "Файлы summary.csv не найдены. Сначала запустите benchmark в notebooks/analysis.ipynb."
            )
        frames = [pd.read_csv(path) for path in summary_files]
        summary = pd.concat(frames, ignore_index=True)
        summary.to_csv(summary_path, index=False)
        return summary

    def load_episode_summary(self) -> pd.DataFrame:
        """Load episode-level aggregates."""
        files = sorted(self.results_dir.glob("*/*/episode_summary.csv"))
        if not files:
            raise FileNotFoundError("Файлы episode_summary.csv не найдены.")
        return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)

    def load_step_records(self, scenario: str, method: str) -> pd.DataFrame:
        """Load step-level records for one scenario and method."""
        episode_files = sorted((self.results_dir / scenario / method / "episodes").glob("episode_*.csv"))
        if not episode_files:
            raise FileNotFoundError(f"Не найдены step-level результаты для {scenario}/{method}.")
        return pd.concat([pd.read_csv(path) for path in episode_files], ignore_index=True)

    @staticmethod
    def _format_summary_table(df: pd.DataFrame) -> pd.DataFrame:
        formatted = df.copy()
        rename_map = {
            "scenario_label": "Сценарий",
            "method_label": "Метод",
            "mean_acceptance_rate": "Принятие задач, %",
            "mean_social_welfare": "Социальное благосостояние",
            "mean_avg_latency": "Средняя задержка, мс",
            "mean_gini_payment": "Джини платежей",
            "mean_fairness_index": "Индекс справедливости",
            "mean_completed_tasks": "Выполнено задач",
            "mean_reward": "Среднее вознаграждение",
        }
        formatted = formatted[list(rename_map.keys())].rename(columns=rename_map)
        return formatted

    def export_tables(self):
        """Export summary tables in CSV and LaTeX formats."""
        summary = self.load_summary()
        summary = summary.sort_values(["scenario", "method"])
        chapter_table = self._format_summary_table(summary)
        chapter_table.to_csv(self.tables_dir / "summary_table.csv", index=False)
        chapter_table.to_latex(
            self.tables_dir / "summary_table.tex",
            index=False,
            escape=False,
            float_format=lambda value: f"{value:.3f}" if isinstance(value, float) else str(value),
        )

        winners = (
            summary.sort_values("mean_social_welfare", ascending=False)
            .groupby("scenario_label", as_index=False)
            .first()[["scenario_label", "method_label", "mean_social_welfare"]]
            .rename(
                columns={
                    "scenario_label": "Сценарий",
                    "method_label": "Лучший метод по SW",
                    "mean_social_welfare": "Макс. SW",
                }
            )
        )
        winners.to_csv(self.tables_dir / "scenario_winners.csv", index=False)
        winners.to_latex(
            self.tables_dir / "scenario_winners.tex",
            index=False,
            escape=False,
            float_format=lambda value: f"{value:.3f}" if isinstance(value, float) else str(value),
        )

        for scenario_name, scenario_df in summary.groupby("scenario"):
            scenario_table = self._format_summary_table(scenario_df)
            scenario_table.to_csv(self.tables_dir / f"{scenario_name}_table.csv", index=False)
            scenario_table.to_latex(
                self.tables_dir / f"{scenario_name}_table.tex",
                index=False,
                escape=False,
                float_format=lambda value: f"{value:.3f}" if isinstance(value, float) else str(value),
            )

    def plot_method_overview(self):
        """Grouped bar charts for the key summary metrics."""
        summary = self.load_summary()
        metrics = [
            ("mean_social_welfare", "Социальное благосостояние"),
            ("mean_acceptance_rate", "Принятие задач, %"),
            ("mean_avg_latency", "Средняя задержка, мс"),
            ("mean_gini_payment", "Джини платежей"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        for ax, (metric, title) in zip(axes, metrics):
            sns.barplot(
                data=summary,
                x="scenario_label",
                y=metric,
                hue="method_label",
                palette=self.palette,
                ax=ax,
            )
            ax.set_title(title)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=15)
            if metric == "mean_gini_payment":
                ax.axhline(0.3, color="#7c3aed", linestyle="--", linewidth=1.5, label="Целевой порог")
        handles, labels = axes[0].get_legend_handles_labels()
        for ax in axes:
            if ax.legend_:
                ax.legend_.remove()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.suptitle("Сравнение методов по ключевым метрикам", y=1.02)
        plt.tight_layout()
        self._save_figure("method_overview")
        plt.close()

    def plot_learning_curves(self):
        """Episode-level learning curves for the trainable methods."""
        episode_summary = self.load_episode_summary()
        metrics = [
            ("mean_social_welfare", "Социальное благосостояние"),
            ("mean_acceptance_rate", "Принятие задач, %"),
            ("mean_reward", "Среднее вознаграждение"),
            ("mean_td_error", "TD-ошибка"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        for ax, (metric, title) in zip(axes, metrics):
            sns.lineplot(
                data=episode_summary,
                x="episode",
                y=metric,
                hue="method_label",
                style="scenario_label",
                palette=self.palette,
                linewidth=2.2,
                ax=ax,
            )
            ax.set_title(title)
            ax.set_xlabel("Эпизод")
        handles, labels = axes[0].get_legend_handles_labels()
        for ax in axes:
            if ax.legend_:
                ax.legend_.remove()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.suptitle("Кривые обучения и динамика качества по эпизодам", y=1.02)
        plt.tight_layout()
        self._save_figure("learning_curves")
        plt.close()

    def plot_step_dynamics(self, scenarios: Iterable[str] | None = None):
        """Step-level dynamics averaged over episodes for each scenario."""
        summary = self.load_summary()
        scenario_names = list(scenarios or summary["scenario"].unique())
        for scenario_name in scenario_names:
            scenario_rows = summary[summary["scenario"] == scenario_name]
            frames = []
            for _, row in scenario_rows.iterrows():
                step_df = self.load_step_records(row["scenario"], row["method"])
                grouped = (
                    step_df.groupby("step", as_index=False)[
                        ["social_welfare", "acceptance_rate", "avg_latency"]
                    ]
                    .mean()
                    .assign(method_label=row["method_label"], scenario_label=row["scenario_label"])
                )
                frames.append(grouped)
            dynamics = pd.concat(frames, ignore_index=True)

            fig, axes = plt.subplots(3, 1, figsize=(16, 15), sharex=True)
            plots = [
                ("social_welfare", "Социальное благосостояние"),
                ("acceptance_rate", "Принятие задач, %"),
                ("avg_latency", "Средняя задержка, мс"),
            ]
            for ax, (metric, title) in zip(axes, plots):
                sns.lineplot(
                    data=dynamics,
                    x="step",
                    y=metric,
                    hue="method_label",
                    palette=self.palette,
                    linewidth=2.4,
                    ax=ax,
                )
                ax.set_title(title)
                ax.set_xlabel("Шаг симуляции")
                if metric == "avg_latency":
                    ax.set_ylim(bottom=0)
            handles, labels = axes[0].get_legend_handles_labels()
            for ax in axes:
                if ax.legend_:
                    ax.legend_.remove()
            fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
            fig.suptitle(
                f"Пошаговая динамика метрик: {scenario_rows['scenario_label'].iloc[0]}",
                y=1.01,
            )
            plt.tight_layout()
            self._save_figure(f"{scenario_name}_dynamics")
            plt.close()

    def plot_load_heatmaps(self, preferred_method: str = "hybrid"):
        """Heatmaps of node load for the preferred method in each scenario."""
        summary = self.load_summary()
        for scenario_name in summary["scenario"].unique():
            if not (self.results_dir / scenario_name / preferred_method).exists():
                continue
            step_df = self.load_step_records(scenario_name, preferred_method)
            load_cols = [column for column in step_df.columns if column.startswith("load_node_")]
            if not load_cols:
                continue
            heatmap_data = (
                step_df.groupby("step", as_index=True)[load_cols].mean().T
            )
            heatmap_data.index = [index.replace("load_node_", "Узел ") for index in heatmap_data.index]
            plt.figure(figsize=(18, 6))
            sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws={"label": "Нагрузка"})
            scenario_label = summary.loc[summary["scenario"] == scenario_name, "scenario_label"].iloc[0]
            plt.title(f"Распределение нагрузки по узлам: {scenario_label} ({preferred_method})")
            plt.xlabel("Шаг симуляции")
            plt.ylabel("")
            plt.tight_layout()
            self._save_figure(f"{scenario_name}_{preferred_method}_load_heatmap")
            plt.close()

    def plot_fairness_welfare_scatter(self):
        """Scatter plot showing the welfare/fairness trade-off."""
        summary = self.load_summary()
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=summary,
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
        plt.title("Компромисс между справедливостью и эффективностью")
        plt.tight_layout()
        self._save_figure("fairness_welfare_scatter")
        plt.close()

    def build_all(self):
        """Build the complete chapter artifact set."""
        self.export_tables()
        self.plot_method_overview()
        self.plot_learning_curves()
        self.plot_step_dynamics()
        self.plot_load_heatmaps()
        self.plot_fairness_welfare_scatter()


if __name__ == "__main__":
    ResultsVisualizer().build_all()
