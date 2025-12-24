import numpy as np
from typing import Dict, List, Tuple

def calculate_vcg_payments(
    allocation: np.ndarray,  # Binary matrix [m x n]: allocation[i][j] = 1 if device i uses edge j
    utility_matrix: np.ndarray,  # [m x n]: U[i][j]
    cost_matrix: np.ndarray,  # [m x n]: C[i][j]
) -> Tuple[np.ndarray, float]:
    """
    Вычислить платежи VCG
    
    Args:
        allocation: матрица размещения [m x n]
        utility_matrix: матрица полезности [m x n]
        cost_matrix: матрица стоимости [m x n]
    
    Returns:
        payments: вектор платежей [m]
        total_sw: итоговое социальное благосустояние
    """
    m, n = allocation.shape  # m - устройства, n - узлы
    
    # Социальное благосустояние с текущим распределением
    current_sw = np.sum(allocation * utility_matrix) - np.sum(allocation * cost_matrix)
    
    # Платежи VCG
    payments = np.zeros(m)
    
    for i in range(m):
        # Создать поддельное распределение без устройства i
        allocation_without_i = allocation.copy()
        allocation_without_i[i] = 0
        
        # SW без устройства i
        sw_without_i = (np.sum(allocation_without_i * utility_matrix) - 
                       np.sum(allocation_without_i * cost_matrix))
        
        # Платёж = внешний эффект
        payments[i] = sw_without_i - (current_sw - np.sum(allocation[i] * utility_matrix))
    
    return payments, current_sw

def update_utility_function(
    utility_matrix: np.ndarray,
    success_rate: np.ndarray,
    learning_rate: float = 0.01
) -> np.ndarray:
    """
    Адаптировать функцию полезности на основе успешных выполнений
    
    Args:
        utility_matrix: текущая матрица полезности [m x n]
        success_rate: процент успешных выполнений [m x n]
        learning_rate: скорость адаптации
    
    Returns:
        updated_utility_matrix: обновлённая матрица
    """
    # Экспоненциальное скользящее среднее
    updated = utility_matrix * (1 - learning_rate) + success_rate * learning_rate
    return np.clip(updated, 0, 1)

def update_cost_function(
    cost_matrix: np.ndarray,
    actual_costs: np.ndarray,
    overload_penalty: np.ndarray,
    learning_rate: float = 0.01
) -> np.ndarray:
    """
    Адаптировать функцию стоимости на основе реальных затрат
    """
    combined_cost = actual_costs + overload_penalty
    updated = cost_matrix * (1 - learning_rate) + combined_cost * learning_rate
    return np.clip(updated, 0, 10)
