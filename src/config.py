"""Глобальная конфигурация проекта."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class EnvironmentConfig:
    """Параметры симулируемого edge-окружения."""

    num_nodes: int = 10
    num_devices: int = 20
    task_lambda_arrival: float = 2.5
    arrival_cycle: int = 100
    arrival_amplitude: float = 0.3
    load_spike_probability: float = 0.0
    load_spike_multiplier: float = 2.5
    failure_fraction: float = 0.0
    failure_start_step: int = 0
    failure_recovery_steps: int = 0
    step_duration_ms: int = 50
    device_importance: Dict[str, float] = field(
        default_factory=lambda: {"alpha": 2.0, "beta": 5.0}
    )
    node_to_node_probability: float = 0.4
    node_to_node_weight: Dict[str, float] = field(
        default_factory=lambda: {"min": 0.5, "max": 5.0}
    )
    device_to_node_max_edges: int = 3
    device_to_node_weight: Dict[str, float] = field(
        default_factory=lambda: {"min": 0.5, "max": 2.0}
    )


@dataclass
class TaskConfig:
    """Параметры генерируемых вычислительных задач."""

    cpu: Dict[str, int] = field(default_factory=lambda: {"min": 1, "max": 8})
    memory: Dict[str, int] = field(default_factory=lambda: {"min": 1, "max": 16})
    data_size: Dict[str, int] = field(default_factory=lambda: {"min": 1, "max": 30})
    deadline: Dict[str, int] = field(default_factory=lambda: {"min": 500, "max": 10000})


@dataclass
class NodeConfig:
    """Параметры вычислительного узла."""

    cpu_capacity: int = 32
    memory_capacity: int = 64
    heterogeneous_resources: bool = False
    cpu_capacity_range: Tuple[int, int] = (20, 40)
    memory_capacity_range: Tuple[int, int] = (32, 64)
    cpu_unit_price: float = 0.2
    memory_unit_price: float = 0.05
    network_unit_price: float = 0.01
    transmission_energy_coeff: float = 0.001
    indirect_link_penalty: float = 1.25
    delay_sensitivity: float = 1.0


@dataclass
class TrainingConfig:
    """Параметры обучения QMIX."""

    num_episodes: int = 10
    max_steps_per_episode: int = 100
    learning_rate: float = 0.001
    gamma: float = 0.99
    batch_size: int = 32
    buffer_size: int = 10000
    target_update_freq: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    grad_norm_clip: float = 10.0


@dataclass
class NetworkConfig:
    """Параметры нейросетей QMIX."""

    hidden_size: int = 64
    obs_size: int = 5
    action_size: int = 4
    state_size: Optional[int] = None


@dataclass
class AuctionConfig:
    """Параметры интеграции аукциона и QMIX."""

    vcg_weight: float = 0.5
    global_reward_weight: float = 0.1
    payment_scaling: float = 1.0
    gini_target: float = 0.3
    fairness_target: float = 0.85


ENV_CONFIG = EnvironmentConfig()
TASK_CONFIG = TaskConfig()
NODE_CONFIG = NodeConfig()
TRAINING_CONFIG = TrainingConfig()
NETWORK_CONFIG = NetworkConfig()
AUCTION_CONFIG = AuctionConfig()
