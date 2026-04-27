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
    evaluation_episodes: int = 6


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


def _valid_actions_from_observation(obs: np.ndarray) -> np.ndarray:
    """Extract the valid action ids from the observation tail mask."""
    mask = np.asarray(obs[-4:], dtype=np.float32) > 0.5
    valid_actions = np.flatnonzero(mask)
    if valid_actions.size == 0:
        return np.array([0], dtype=int)
    return valid_actions.astype(int)


def _always_accept_policy(observations: np.ndarray, step: int, episode: int) -> np.ndarray:
    """Accept the auction recommendation whenever the mask allows it."""
    del step, episode
    actions = []
    for obs in observations:
        valid_actions = _valid_actions_from_observation(obs)
        actions.append(0 if 0 in valid_actions else int(valid_actions[0]))
    return np.array(actions, dtype=int)


def _random_policy(observations: np.ndarray, step: int, episode: int) -> np.ndarray:
    """Sample uniformly from the currently valid local actions."""
    del step, episode
    return np.array(
        [int(np.random.choice(_valid_actions_from_observation(obs))) for obs in observations],
        dtype=int,
    )


def _load_aware_policy(observations: np.ndarray, step: int, episode: int) -> np.ndarray:
    """Interpretable heuristic that reacts to deadline pressure and local load."""
    del step, episode
    actions = []
    for obs in observations:
        valid_actions = _valid_actions_from_observation(obs)
        if valid_actions.size == 1:
            actions.append(int(valid_actions[0]))
            continue

        cpu_available = float(obs[0])
        memory_available = float(obs[1])
        queue_level = float(obs[2])
        payment_signal = float(obs[6])
        cpu_pressure = float(obs[7])
        service_ratio = float(obs[8])

        if service_ratio > 1.0 or cpu_available < 0.18 or memory_available < 0.18:
            preferred_action = 1
        elif service_ratio > 0.8 or payment_signal > 0.55:
            preferred_action = 2
        elif queue_level > 0.6 and cpu_pressure < 0.55 and service_ratio < 0.5:
            preferred_action = 3
        else:
            preferred_action = 0

        if preferred_action in valid_actions:
            actions.append(preferred_action)
        else:
            actions.append(int(valid_actions[0]))

    return np.array(actions, dtype=int)


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
                evaluation_episodes=6,
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
                    load_spike_probability=0.0,
                    load_spike_multiplier=2.2,
                    load_spike_windows=((28, 35), (60, 68)),
                ),
                node_config=NODE_CONFIG,
                task_config=TASK_CONFIG,
                training_config=replace(base_training, max_steps_per_episode=100),
                evaluation_episodes=6,
                description=(
                    "Нестационарная нагрузка с детерминированными интервалами резкого "
                    "роста входного потока."
                ),
            ),
            ScenarioSpec(
                name="heterogeneous",
                label="Сценарий 3: Разнородные ресурсы",
                env_config=replace(ENV_CONFIG, task_lambda_arrival=3.0),
                node_config=replace(NODE_CONFIG, heterogeneous_resources=True),
                task_config=TASK_CONFIG,
                training_config=base_training,
                evaluation_episodes=6,
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
                evaluation_episodes=6,
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
                name="heuristic",
                label="Heuristic-LoadAware",
                auction_config=replace(AUCTION_CONFIG, vcg_weight=0.0, global_reward_weight=0.0),
                learning_enabled=False,
                fixed_policy="load_aware",
                color="#dc2626",
                description=(
                    "Интерпретируемая эвристика, учитывающая локальную загрузку, "
                    "давление по дедлайнам и аукционный сигнал."
                ),
            ),
            MethodSpec(
                name="random",
                label="Random",
                auction_config=replace(AUCTION_CONFIG, vcg_weight=0.0, global_reward_weight=0.0),
                learning_enabled=False,
                fixed_policy="random",
                color="#6b7280",
                description=(
                    "Случайная политика, выбирающая одно из допустимых локальных "
                    "действий."
                ),
            ),
            MethodSpec(
                name="qmix",
                label="QMIX",
                auction_config=replace(AUCTION_CONFIG, vcg_weight=0.0, global_reward_weight=0.0),
                learning_enabled=True,
                color="#f59e0b",
                description=(
                    "Обучаемый controller без экономического слоя: VCG-платежи и "
                    "аукционная reward-компонента отключены."
                ),
            ),
            MethodSpec(
                name="hybrid",
                label="MA-VCG-QMIX",
                auction_config=replace(AUCTION_CONFIG, vcg_weight=0.65, global_reward_weight=0.15),
                learning_enabled=True,
                color="#059669",
                description=(
                    "Гибридный метод: аукционная reward-компонента включается "
                    "только в стрессовых режимах среды."
                ),
            ),
        ]

    @staticmethod
    def _policy_from_name(name: str) -> PolicyFn:
        if name == "always_accept":
            return _always_accept_policy
        if name == "random":
            return _random_policy
        if name == "load_aware":
            return _load_aware_policy
        raise ValueError(f"Неизвестная фиксированная политика: {name}")

    @staticmethod
    def _seed_everything(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def _build_trainer(
        env: EdgeComputingSystem,
        scenario: ScenarioSpec,
    ) -> QMIXTrainer:
        network_config = replace(
            NETWORK_CONFIG,
            obs_size=env.observation_size,
            state_size=scenario.env_config.num_nodes * env.observation_size,
        )
        return QMIXTrainer(
            num_agents=scenario.env_config.num_nodes,
            network_config=network_config,
            training_config=scenario.training_config,
        )

    def _run_episode(
        self,
        env: EdgeComputingSystem,
        scenario: ScenarioSpec,
        method: MethodSpec,
        seed: int,
        episode: int,
        phase: str,
        trainer: Optional[QMIXTrainer] = None,
        policy: Optional[PolicyFn] = None,
        train: bool = False,
    ) -> tuple[pd.DataFrame, dict]:
        """Run one train or evaluation episode and return step + episode aggregates."""
        step_records = []
        max_steps = scenario.training_config.max_steps_per_episode
        state = env.get_observations()
        for step in range(max_steps):
            if trainer is not None:
                actions = trainer.select_actions(state)
            else:
                actions = policy(state, step, episode)

            rewards, info, metrics = env.step(actions)
            next_state = None
            if train or step < max_steps - 1:
                next_state = env.get_observations()

            td_error = np.nan
            epsilon = np.nan
            if trainer is not None:
                epsilon = trainer.epsilon
            if train and trainer is not None:
                trainer.add_experience(
                    state=state,
                    actions=actions,
                    rewards=rewards,
                    next_state=next_state if next_state is not None else state,
                    done=step == max_steps - 1,
                )
                td_error = trainer.train_step()
                epsilon = trainer.epsilon

            record = {
                "scenario": scenario.name,
                "scenario_label": scenario.label,
                "method": method.name,
                "method_label": method.label,
                "seed": seed,
                "phase": phase,
                "episode": episode,
                "step": step,
                "generated_tasks": info["generated"],
                "accepted_tasks": info["accepted"],
                "rejected_tasks": info["rejected"],
                "completed_tasks": info["completed"],
                "avg_latency": metrics["avg_latency"],
                "acceptance_rate": metrics["acceptance_rate"],
                "drop_rate": metrics["drop_rate"],
                "deadline_success_rate": metrics["deadline_success_rate"],
                "completed_before_deadline": metrics["completed_before_deadline"],
                "deadline_violations": metrics["deadline_violations"],
                "gini_payment": metrics["gini_payment"],
                "fairness_index": metrics["fairness_index"],
                "social_welfare": metrics["social_welfare"],
                "load_imbalance": metrics["load_imbalance"],
                "backlog_pressure": metrics["backlog_pressure"],
                "arrival_rate": metrics["arrival_rate"],
                "load_spike_active": metrics["load_spike_active"],
                "failed_nodes": metrics["failed_nodes"],
                "stress_context": metrics["stress_context"],
                "reward_mean": float(np.mean(rewards)),
                "reward_std": float(np.std(rewards)),
                "td_error": np.nan if td_error is None else float(td_error),
                "epsilon": epsilon,
            }
            for node_idx, load in enumerate(metrics["resource_utilization"]):
                record[f"load_node_{node_idx}"] = load
            step_records.append(record)

            if next_state is not None:
                state = next_state

        step_df = pd.DataFrame(step_records)
        episode_summary = {
            "scenario": scenario.name,
            "scenario_label": scenario.label,
            "method": method.name,
            "method_label": method.label,
            "seed": seed,
            "phase": phase,
            "episode": episode,
            "mean_acceptance_rate": step_df["acceptance_rate"].mean(),
            "mean_social_welfare": step_df["social_welfare"].mean(),
            "mean_avg_latency": step_df["avg_latency"].mean(),
            "mean_drop_rate": step_df["drop_rate"].mean(),
            "mean_deadline_success_rate": step_df["deadline_success_rate"].mean(),
            "mean_gini_payment": step_df["gini_payment"].mean(),
            "mean_fairness_index": step_df["fairness_index"].mean(),
            "mean_load_imbalance": step_df["load_imbalance"].mean(),
            "mean_backlog_pressure": step_df["backlog_pressure"].mean(),
            "mean_completed_tasks": step_df["completed_tasks"].mean(),
            "mean_completed_before_deadline": step_df["completed_before_deadline"].mean(),
            "mean_reward": step_df["reward_mean"].mean(),
            "mean_td_error": step_df["td_error"].dropna().mean(),
            "final_epsilon": step_df["epsilon"].dropna().iloc[-1]
            if step_df["epsilon"].dropna().size
            else np.nan,
        }
        return step_df, episode_summary

    @staticmethod
    def _aggregate_seed_summaries(summary_by_seed: pd.DataFrame) -> pd.DataFrame:
        """Aggregate seed-level experiment summaries into chapter-level statistics."""
        group_cols = [
            "scenario",
            "scenario_label",
            "scenario_description",
            "method",
            "method_label",
            "method_description",
            "learning_enabled",
            "num_nodes",
            "num_devices",
            "arrival_rate",
            "training_episodes",
            "evaluation_episodes",
            "episode_length",
        ]
        metric_cols = [
            column
            for column in summary_by_seed.columns
            if column not in group_cols + ["seed"]
            and pd.api.types.is_numeric_dtype(summary_by_seed[column])
        ]
        aggregated = summary_by_seed.groupby(group_cols, as_index=False).agg(
            {metric: ["mean", "std", "count"] for metric in metric_cols}
        )
        aggregated.columns = [
            "_".join(filter(None, column)).rstrip("_")
            for column in aggregated.columns.to_flat_index()
        ]
        rename_map = {f"{column}_mean": column for column in metric_cols}
        aggregated = aggregated.rename(columns=rename_map)
        for metric in metric_cols:
            std_column = f"{metric}_std"
            count_column = f"{metric}_count"
            if std_column in aggregated.columns:
                aggregated[std_column] = aggregated[std_column].fillna(0.0)
            if count_column in aggregated.columns:
                aggregated[f"{metric}_ci95"] = (
                    1.96
                    * aggregated[std_column]
                    / np.sqrt(np.maximum(aggregated[count_column], 1))
                )
        aggregated = aggregated.rename(columns={"mean_social_welfare_count": "num_seeds"})
        if "num_seeds" not in aggregated.columns:
            aggregated["num_seeds"] = 1
        return aggregated

    def run_suite(
        self,
        scenarios: Optional[Iterable[ScenarioSpec]] = None,
        methods: Optional[Iterable[MethodSpec]] = None,
        seed: int = 42,
        num_seeds: int = 5,
    ) -> pd.DataFrame:
        """Run the full validation benchmark suite."""
        scenarios = list(scenarios or self.default_scenarios())
        methods = list(methods or self.default_methods())
        summaries = []
        total_runs = num_seeds * len(scenarios) * len(methods)
        run_index = 0
        for seed_idx in range(num_seeds):
            run_seed_base = seed + seed_idx * 1000
            for scenario_idx, scenario in enumerate(scenarios):
                for method in methods:
                    run_seed = run_seed_base + scenario_idx * 100
                    run_index += 1
                    print(
                        f"[{run_index:03d}/{total_runs:03d}] "
                        f"{scenario.name}/{method.name} seed={run_seed}"
                    )
                    summaries.append(self.run_experiment(scenario, method, run_seed))

        summary_by_seed = pd.DataFrame(summaries).sort_values(["scenario", "method", "seed"])
        summary_by_seed.to_csv(self.results_root / "summary_by_seed.csv", index=False)
        summary = self._aggregate_seed_summaries(summary_by_seed)
        summary.to_csv(self.results_root / "summary.csv", index=False)
        return summary

    def run_experiment(self, scenario: ScenarioSpec, method: MethodSpec, seed: int) -> dict:
        """Run one method inside one scenario and persist the artifacts."""
        self._seed_everything(seed)
        seed_dir = self.results_root / scenario.name / method.name / f"seed_{seed:05d}"
        train_dir = seed_dir / "train"
        eval_dir = seed_dir / "eval"
        eval_episode_dir = eval_dir / "episodes"
        train_dir.mkdir(parents=True, exist_ok=True)
        eval_episode_dir.mkdir(parents=True, exist_ok=True)

        env = EdgeComputingSystem(
            env_config=scenario.env_config,
            node_config=scenario.node_config,
            task_config=scenario.task_config,
            auction_config=method.auction_config,
        )

        trainer = self._build_trainer(env, scenario) if method.learning_enabled else None
        policy = None if method.learning_enabled else self._policy_from_name(
            method.fixed_policy or "always_accept"
        )

        training_episode_summaries = []
        if trainer is not None:
            for episode in range(scenario.training_config.num_episodes):
                env.reset()
                _, episode_summary = self._run_episode(
                    env=env,
                    scenario=scenario,
                    method=method,
                    seed=seed,
                    episode=episode,
                    phase="train",
                    trainer=trainer,
                    train=True,
                )
                training_episode_summaries.append(episode_summary)

        training_episode_df = pd.DataFrame(training_episode_summaries)
        if not training_episode_df.empty:
            training_episode_df.to_csv(train_dir / "episode_summary.csv", index=False)

        evaluation_episode_summaries = []
        if trainer is not None:
            trainer.epsilon = 0.0
        for episode in range(scenario.evaluation_episodes):
            env.reset()
            step_df, episode_summary = self._run_episode(
                env=env,
                scenario=scenario,
                method=method,
                seed=seed,
                episode=episode,
                phase="eval",
                trainer=trainer,
                policy=policy,
                train=False,
            )
            step_df.to_csv(eval_episode_dir / f"episode_{episode:02d}.csv", index=False)
            evaluation_episode_summaries.append(episode_summary)

        evaluation_episode_df = pd.DataFrame(evaluation_episode_summaries)
        evaluation_episode_df.to_csv(eval_dir / "episode_summary.csv", index=False)

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
            "training_episodes": scenario.training_config.num_episodes,
            "evaluation_episodes": scenario.evaluation_episodes,
            "episode_length": scenario.training_config.max_steps_per_episode,
            "mean_acceptance_rate": evaluation_episode_df["mean_acceptance_rate"].mean(),
            "mean_social_welfare": evaluation_episode_df["mean_social_welfare"].mean(),
            "mean_avg_latency": evaluation_episode_df["mean_avg_latency"].mean(),
            "mean_drop_rate": evaluation_episode_df["mean_drop_rate"].mean(),
            "mean_deadline_success_rate": evaluation_episode_df["mean_deadline_success_rate"].mean(),
            "mean_gini_payment": evaluation_episode_df["mean_gini_payment"].mean(),
            "mean_fairness_index": evaluation_episode_df["mean_fairness_index"].mean(),
            "mean_load_imbalance": evaluation_episode_df["mean_load_imbalance"].mean(),
            "mean_completed_tasks": evaluation_episode_df["mean_completed_tasks"].mean(),
            "mean_completed_before_deadline": evaluation_episode_df["mean_completed_before_deadline"].mean(),
            "mean_reward": evaluation_episode_df["mean_reward"].mean(),
            "final_training_social_welfare": training_episode_df["mean_social_welfare"].iloc[-1]
            if not training_episode_df.empty
            else np.nan,
            "final_training_reward": training_episode_df["mean_reward"].iloc[-1]
            if not training_episode_df.empty
            else np.nan,
            "final_td_error": training_episode_df["mean_td_error"].dropna().iloc[-1]
            if not training_episode_df.empty and training_episode_df["mean_td_error"].dropna().size
            else np.nan,
            "final_epsilon": training_episode_df["final_epsilon"].dropna().iloc[-1]
            if not training_episode_df.empty and training_episode_df["final_epsilon"].dropna().size
            else np.nan,
            "learning_enabled": method.learning_enabled,
        }

        pd.DataFrame([summary]).to_csv(seed_dir / "summary.csv", index=False)
        return summary
