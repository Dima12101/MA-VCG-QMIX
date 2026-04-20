"""Benchmark runner for the experimental validation chapter."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Optional
import random

import numpy as np
import pandas as pd
import torch

from src.config import (
    AUCTION_CONFIG,
    ENV_CONFIG,
    NETWORK_CONFIG,
    NODE_CONFIG,
    TASK_CONFIG,
    TRAINING_CONFIG,
    AuctionConfig,
    EnvironmentConfig,
    NetworkConfig,
    NodeConfig,
    TaskConfig,
    TrainingConfig,
)
from src.environment.environment import EdgeComputingSystem
from src.learning.trainer import QMIXTrainer


PolicyFn = Callable[[np.ndarray, int, int], np.ndarray]


@dataclass(frozen=True)
class ScenarioSpec:
    """Description of one simulation scenario."""

    name: str
    label: str
    env_config: EnvironmentConfig
    node_config: NodeConfig
    task_config: TaskConfig
    training_config: TrainingConfig
    description: str


@dataclass(frozen=True)
class MethodSpec:
    """Description of one compared method."""

    name: str
    label: str
    auction_config: AuctionConfig
    learning_enabled: bool = True
    fixed_policy: Optional[str] = None
    color: str = "#1f77b4"
    description: str = ""


def _always_accept_policy(observations: np.ndarray, step: int, episode: int) -> np.ndarray:
    del observations, step, episode
    return np.array([], dtype=int)


class BenchmarkRunner:
    """Run comparable experiments for multiple methods and scenarios."""

    def __init__(self, results_root: str | Path = "experiments/results/validation"):
        self.results_root = Path(results_root)
        self.results_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def default_scenarios() -> list[ScenarioSpec]:
        """Provide the canonical validation scenarios."""
        base_training = replace(
            TRAINING_CONFIG,
            num_episodes=12,
            max_steps_per_episode=80,
            batch_size=16,
            epsilon_decay=0.997,
        )
        return [
            ScenarioSpec(
                name="baseline",
                label="Сценарий 1: Стабильная нагрузка",
                env_config=replace(ENV_CONFIG, task_lambda_arrival=2.5),
                node_config=NODE_CONFIG,
                task_config=TASK_CONFIG,
                training_config=base_training,
                description=(
                    "Стационарный поток задач без сбоев и без выраженных всплесков "
                    "нагрузки."
                ),
            ),
            ScenarioSpec(
                name="load_spikes",
                label="Сценарий 2: Пиковая нагрузка",
                env_config=replace(
                    ENV_CONFIG,
                    task_lambda_arrival=4.0,
                    load_spike_probability=0.15,
                    load_spike_multiplier=2.0,
                ),
                node_config=NODE_CONFIG,
                task_config=TASK_CONFIG,
                training_config=replace(base_training, max_steps_per_episode=100),
                description=(
                    "Синусоидальная интенсивность потока с добавлением кратковременных "
                    "всплесков нагрузки."
                ),
            ),
            ScenarioSpec(
                name="heterogeneous",
                label="Сценарий 3: Разнородные ресурсы",
                env_config=replace(ENV_CONFIG, task_lambda_arrival=3.0),
                node_config=replace(NODE_CONFIG, heterogeneous_resources=True),
                task_config=TASK_CONFIG,
                training_config=base_training,
                description=(
                    "Гетерогенные edge-узлы с различной CPU- и memory-ёмкостью."
                ),
            ),
            ScenarioSpec(
                name="failures",
                label="Сценарий 4: Отказы узлов",
                env_config=replace(
                    ENV_CONFIG,
                    task_lambda_arrival=3.0,
                    failure_fraction=0.2,
                    failure_start_step=40,
                    failure_recovery_steps=25,
                ),
                node_config=NODE_CONFIG,
                task_config=TASK_CONFIG,
                training_config=replace(base_training, max_steps_per_episode=90),
                description=(
                    "Часть узлов выходит из строя в середине эпизода и затем "
                    "восстанавливается."
                ),
            ),
        ]

    @staticmethod
    def default_methods() -> list[MethodSpec]:
        """Provide the compared methods used in the chapter."""
        return [
            MethodSpec(
                name="vcg",
                label="MA-VCG",
                auction_config=replace(AUCTION_CONFIG, vcg_weight=0.0, global_reward_weight=0.0),
                learning_enabled=False,
                fixed_policy="always_accept",
                color="#2563eb",
                description=(
                    "Аукционный механизм без обучения: все выделенные аукционом задачи "
                    "принимаются к исполнению."
                ),
            ),
            MethodSpec(
                name="qmix",
                label="QMIX",
                auction_config=replace(AUCTION_CONFIG, vcg_weight=0.0, global_reward_weight=0.0),
                learning_enabled=True,
                color="#f59e0b",
                description=(
                    "Обучаемый controller без экономического слоя: VCG-платежи не "
                    "включаются в reward."
                ),
            ),
            MethodSpec(
                name="hybrid",
                label="MA-VCG-QMIX",
                auction_config=replace(AUCTION_CONFIG, vcg_weight=0.5, global_reward_weight=0.2),
                learning_enabled=True,
                color="#059669",
                description=(
                    "Гибридный метод: reward сочетает RL-сигнал, глобальную компоненту "
                    "и VCG-платежи."
                ),
            ),
        ]

    @staticmethod
    def _policy_from_name(name: str) -> PolicyFn:
        if name == "always_accept":
            return lambda observations, step, episode: np.zeros(len(observations), dtype=int)
        raise ValueError(f"Неизвестная фиксированная политика: {name}")

    @staticmethod
    def _seed_everything(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def run_suite(
        self,
        scenarios: Optional[Iterable[ScenarioSpec]] = None,
        methods: Optional[Iterable[MethodSpec]] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Run the full validation benchmark suite."""
        scenarios = list(scenarios or self.default_scenarios())
        methods = list(methods or self.default_methods())
        summaries = []
        for scenario_idx, scenario in enumerate(scenarios):
            for method_idx, method in enumerate(methods):
                run_seed = seed + scenario_idx * 100 + method_idx * 7
                summaries.append(self.run_experiment(scenario, method, run_seed))

        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(self.results_root / "summary.csv", index=False)
        return summary_df

    def run_experiment(self, scenario: ScenarioSpec, method: MethodSpec, seed: int) -> dict:
        """Run one method inside one scenario and persist the artifacts."""
        self._seed_everything(seed)
        scenario_dir = self.results_root / scenario.name / method.name
        episode_dir = scenario_dir / "episodes"
        episode_dir.mkdir(parents=True, exist_ok=True)

        env = EdgeComputingSystem(
            env_config=scenario.env_config,
            node_config=scenario.node_config,
            task_config=scenario.task_config,
            auction_config=method.auction_config,
        )

        trainer = None
        if method.learning_enabled:
            network_config = replace(
                NETWORK_CONFIG,
                state_size=scenario.env_config.num_nodes * NETWORK_CONFIG.obs_size,
            )
            trainer = QMIXTrainer(
                num_agents=scenario.env_config.num_nodes,
                network_config=network_config,
                training_config=scenario.training_config,
            )

        policy = None
        if not method.learning_enabled:
            policy = self._policy_from_name(method.fixed_policy or "always_accept")

        episode_summaries = []

        for episode in range(scenario.training_config.num_episodes):
            env.reset()
            step_records = []
            for step in range(scenario.training_config.max_steps_per_episode):
                state = env.get_observations()
                if trainer is not None:
                    actions = trainer.select_actions(state)
                else:
                    actions = policy(state, step, episode)

                rewards, info, metrics = env.step(actions)
                next_state = env.get_observations()

                td_error = np.nan
                epsilon = np.nan
                if trainer is not None:
                    trainer.add_experience(
                        state=state,
                        actions=actions,
                        rewards=rewards,
                        next_state=next_state,
                        done=step == scenario.training_config.max_steps_per_episode - 1,
                    )
                    td_error = trainer.train_step()
                    epsilon = trainer.epsilon

                record = {
                    "scenario": scenario.name,
                    "scenario_label": scenario.label,
                    "method": method.name,
                    "method_label": method.label,
                    "episode": episode,
                    "step": step,
                    "accepted_tasks": info["accepted"],
                    "rejected_tasks": info["rejected"],
                    "completed_tasks": info["completed"],
                    "avg_latency": metrics["avg_latency"],
                    "acceptance_rate": metrics["acceptance_rate"],
                    "gini_payment": metrics["gini_payment"],
                    "fairness_index": metrics["fairness_index"],
                    "social_welfare": metrics["social_welfare"],
                    "reward_mean": float(np.mean(rewards)),
                    "reward_std": float(np.std(rewards)),
                    "td_error": np.nan if td_error is None else float(td_error),
                    "epsilon": epsilon,
                }
                for node_idx, load in enumerate(metrics["resource_utilization"]):
                    record[f"load_node_{node_idx}"] = load
                step_records.append(record)

            episode_df = pd.DataFrame(step_records)
            episode_df.to_csv(episode_dir / f"episode_{episode:02d}.csv", index=False)
            episode_summary = {
                "scenario": scenario.name,
                "scenario_label": scenario.label,
                "method": method.name,
                "method_label": method.label,
                "episode": episode,
                "mean_acceptance_rate": episode_df["acceptance_rate"].mean(),
                "mean_social_welfare": episode_df["social_welfare"].mean(),
                "mean_avg_latency": episode_df["avg_latency"].mean(),
                "mean_gini_payment": episode_df["gini_payment"].mean(),
                "mean_fairness_index": episode_df["fairness_index"].mean(),
                "mean_completed_tasks": episode_df["completed_tasks"].mean(),
                "mean_reward": episode_df["reward_mean"].mean(),
                "mean_td_error": episode_df["td_error"].dropna().mean(),
                "final_epsilon": episode_df["epsilon"].dropna().iloc[-1]
                if episode_df["epsilon"].dropna().size
                else np.nan,
            }
            episode_summaries.append(episode_summary)

        episode_summary_df = pd.DataFrame(episode_summaries)
        episode_summary_df.to_csv(scenario_dir / "episode_summary.csv", index=False)

        summary = {
            "scenario": scenario.name,
            "scenario_label": scenario.label,
            "scenario_description": scenario.description,
            "method": method.name,
            "method_label": method.label,
            "method_description": method.description,
            "seed": seed,
            "num_nodes": scenario.env_config.num_nodes,
            "num_devices": scenario.env_config.num_devices,
            "arrival_rate": scenario.env_config.task_lambda_arrival,
            "episodes": scenario.training_config.num_episodes,
            "episode_length": scenario.training_config.max_steps_per_episode,
            "mean_acceptance_rate": episode_summary_df["mean_acceptance_rate"].mean(),
            "std_acceptance_rate": episode_summary_df["mean_acceptance_rate"].std(ddof=0),
            "mean_social_welfare": episode_summary_df["mean_social_welfare"].mean(),
            "std_social_welfare": episode_summary_df["mean_social_welfare"].std(ddof=0),
            "mean_avg_latency": episode_summary_df["mean_avg_latency"].mean(),
            "std_avg_latency": episode_summary_df["mean_avg_latency"].std(ddof=0),
            "mean_gini_payment": episode_summary_df["mean_gini_payment"].mean(),
            "mean_fairness_index": episode_summary_df["mean_fairness_index"].mean(),
            "mean_completed_tasks": episode_summary_df["mean_completed_tasks"].mean(),
            "mean_reward": episode_summary_df["mean_reward"].mean(),
            "final_td_error": episode_summary_df["mean_td_error"].dropna().iloc[-1]
            if episode_summary_df["mean_td_error"].dropna().size
            else np.nan,
            "final_epsilon": episode_summary_df["final_epsilon"].dropna().iloc[-1]
            if episode_summary_df["final_epsilon"].dropna().size
            else np.nan,
            "learning_enabled": method.learning_enabled,
        }

        pd.DataFrame([summary]).to_csv(scenario_dir / "summary.csv", index=False)
        return summary
