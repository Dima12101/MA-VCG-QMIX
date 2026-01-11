"""
Глобальная конфигурация для всех компонентов
"""
from dataclasses import dataclass

@dataclass
class EnvironmentConfig:
    """Конфигурация окружения"""
    num_nodes: int = 10                 # Количество Edge-узлов
    num_devices: int = 20               # Количество мобильных устройств
    task_lambda_arrival: float = 2.5    # λ: интенсивность прихода задач
    device_importance = {
        "min": 2, "max": 5}             # Диапазон распределения важности устройств
    node_to_node_probability = 0.4      # Вероятность связи между узлами (%)
    node_to_node_weight = {
        "min": 1.0, "max": 5.0}         # Задержка передачи между узлами / latency (ms)
    device_to_node_max_edges = 3        # Максимальное кол-во связей устройства с узлами
    device_to_node_weight = {
        "min": 0.5, "max": 2.0 }        # Задержка передачи между устройствами и узлами / latency (ms)

@dataclass
class TaskConfig:
    """Конфигурация задач"""
    cpu = {
        "min": 1, "max": 4} 
    memory = {
        "min": 100, "max": 500}         # MB
    bandwidth = {
        "min": 10, "max": 100}          # Mbps
    data_size = {
        "min": 1, "max": 30}            # MB
    deadline = {
        "min": 1000, "max": 5000}       # ms
    importance = {
        "min": 0.5, "max": 1.0}

@dataclass
class NodeConfig:
    """Конфигурация edge-узла"""
    cpu_capacity: int = 4               
    memory_capacity: int = 1024          # MB
    bandwidth: float = 100.0             # Mbps

@dataclass
class TrainingConfig:
    """Конфигурация обучения QMIX"""
    num_episodes: int = 10              # Кол-во эпизодов обучения
    max_steps_per_episode: int = 50     # Максимальное кол-во шагов обучения на одном эпизоде
    learning_rate: float = 0.001
    gamma: float = 0.99                 # Discount factor
    batch_size: int = 32
    buffer_size: int = 10000
    target_update_freq: int = 100       # Обновлять целевые сети каждые N шагов
    epsilon_start: float = 1.0          # Epsilon-greedy exploration
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995

@dataclass
class NetworkConfig:
    """Конфигурация нейросетей"""
    hidden_size: int = 64               # GRU hidden size
    obs_size: int = 5                   # Размер наблюдения
    action_size: int = 4                # Размер действия (Accept, Reject, Priority_High, Priority_Low)

@dataclass
class AuctionConfig:
    """Конфигурация VCG аукциона"""
    vcg_weight: float = 0.5             # Вес VCG в функции вознаграждения
    payment_scaling: float = 1.0
    gini_target: float = 0.3
    fairness_target: float = 0.85


# Глобальная конфигурация
ENV_CONFIG = EnvironmentConfig()
TASK_CONFIG = TaskConfig()
NODE_CONFIG = NodeConfig()
TRAINING_CONFIG = TrainingConfig()
NETWORK_CONFIG = NetworkConfig()
AUCTION_CONFIG = AuctionConfig()