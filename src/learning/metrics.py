import numpy as np
from typing import List

def calculate_gini_coefficient(payments: List[float]) -> float:
    """Коэффициент Джини для платежей"""
    payments = np.array(sorted(payments))
    n = len(payments)
    cumsum = np.cumsum(payments)
    return (2 * np.sum((np.arange(1, n+1)) * payments)) / (n * np.sum(payments)) - (n + 1) / n

def calculate_fairness_index(allocations: np.ndarray) -> float:
    """Индекс справедливости Джини для распределений"""
    x = allocations.sum(axis=1)  # По устройствам
    return (x.sum() ** 2) / (len(x) * np.sum(x ** 2))

def calculate_social_welfare(
    utility_matrix: np.ndarray,
    cost_matrix: np.ndarray,
    allocation: np.ndarray
) -> float:
    """Социальное благосустояние"""
    utility_part = np.sum(allocation * utility_matrix)
    cost_part = np.sum(allocation * cost_matrix)
    return utility_part - cost_part

def calculate_td_error(
    q_current: float,
    reward: float,
    q_next: float,
    gamma: float = 0.99,
    done: bool = False
) -> float:
    """TD-ошибка для обучения QMIX"""
    target = reward + (gamma * q_next) * (1 - done)
    error = abs(target - q_current)
    return error

def calculate_acceptance_rate(
    accepted_count: int,
    total_count: int
) -> float:
    """Процент принятых задач"""
    if total_count == 0:
        return 0.0
    return accepted_count / total_count * 100

def calculate_avg_latency(latencies: List[float]) -> float:
    """Средняя задержка"""
    if len(latencies) == 0:
        return 0.0
    return np.mean(latencies)

def calculate_resource_utilization(
    used: np.ndarray,
    capacity: np.ndarray
) -> float:
    """Утилизация ресурсов"""
    return np.mean(used / capacity) * 100
