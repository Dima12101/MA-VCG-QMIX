"""
Конфигурация глобальная для всех компонентов
"""

from dataclasses import dataclass
from typing import Dict

@dataclass
class EnvironmentConfig:
    """Конфигурация окружения"""
    num_edges: int = 4  # Edge-узлы
    num_devices: int = 100  # IoT устройства
    lambda_arrival: float = 2.5  # Интенсивность трафика
    max_task_value: float = 1.0  # Максимальная ценность задачи
    min_task_value: float = 0.1
    max_processing_time: int = 100  # максимум ms
    min_processing_time: int = 10

@dataclass
class TrainingConfig:
    """Конфигурация обучения QMIX"""
    num_episodes: int = 1000
    max_steps_per_episode: int = 500
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor
    batch_size: int = 32
    buffer_size: int = 10000
    target_update_freq: int = 100  # Обновлять целевые сети каждые N шагов
    epsilon_start: float = 1.0  # Epsilon-greedy exploration
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995

@dataclass
class NetworkConfig:
    """Конфигурация нейросетей"""
    hidden_size: int = 64  # GRU hidden size
    obs_size: int = 16  # Размер наблюдения
    action_size: int = 4  # Размер действия

@dataclass
class AuctionConfig:
    """Конфигурация VCG аукциона"""
    vcg_enabled: bool = True
    vcg_weight: float = 0.5  # Вес VCG в функции вознаграждения
    payment_scaling: float = 1.0
    gini_target: float = 0.3
    fairness_target: float = 0.85

class ENV_CONFIG(EnvironmentConfig, TrainingConfig, NetworkConfig, AuctionConfig):
    """Объединённая конфигурация"""
    
    def copy(self):
        """Создать копию конфигурации"""
        return ENV_CONFIG(
            num_edges=self.num_edges,
            num_devices=self.num_devices,
            lambda_arrival=self.lambda_arrival,
            max_task_value=self.max_task_value,
            min_task_value=self.min_task_value,
            max_processing_time=self.max_processing_time,
            min_processing_time=self.min_processing_time,
            num_episodes=self.num_episodes,
            max_steps_per_episode=self.max_steps_per_episode,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            batch_size=self.batch_size,
            buffer_size=self.buffer_size,
            target_update_freq=self.target_update_freq,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay=self.epsilon_decay,
            hidden_size=self.hidden_size,
            obs_size=self.obs_size,
            action_size=self.action_size,
            vcg_enabled=self.vcg_enabled,
            vcg_weight=self.vcg_weight,
            payment_scaling=self.payment_scaling,
            gini_target=self.gini_target,
            fairness_target=self.fairness_target
        )

# Создать глобальный объект конфигурации
ENV_CONFIG = ENV_CONFIG()
