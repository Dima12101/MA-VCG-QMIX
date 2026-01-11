"""
Механизм MA-VCG (Multi-Agent VCG Auction)
"""

import numpy as np
from typing import List
from dataclasses import dataclass

@dataclass
class AuctionResult:
    """Результат одного раунда аукциона"""
    allocation: np.ndarray  # [m x n]: устройства × узлы
    payments: np.ndarray    # [m]: платежи для каждого устройства

class VCGAuctioneer:
    """Класс для проведения MA-VCG аукционера"""
    
    def __init__(self, num_devices: int, num_edges: int):
        self.num_devices = num_devices
        self.num_edges = num_edges

    def reset(self):
        self.history = []
    
    def run_auction(
        self,
        utilities: np.ndarray,      # [m x n]: полезность
        costs: np.ndarray,          # [m x n]: стоимость
    ) -> AuctionResult:
        """
        Провести один раунд аукциона
        
        Args:
            utilities: матрица полезности [m x n]
            costs: матрица стоимости [m x n]
        
        Returns:
            AuctionResult: результат аукциона
        """
        # Фаза оптимизации (вычислить оптимальное распределение)
        allocation = self._compute_optimal_allocation(utilities, costs)
        
        # Фаза платежей (вычислить VCG платежи)
        payments = self._compute_vcg_payments(allocation, utilities, costs)
        
        result = AuctionResult(
            allocation=allocation,
            payments=payments,
        )
        
        return result
    
    def _compute_optimal_allocation(
        self,
        utilities: np.ndarray,
        costs: np.ndarray
    ) -> np.ndarray:
        """
        Вычислить оптимальное распределение задач
        Используется жадный алгоритм для каждого устройства
        """
        m, n = utilities.shape
        allocation = np.zeros((m, n), dtype=int)
        
        # Для каждого устройства выбрать лучший узел
        for i in range(m):
            # Вычислить "выгоду" для каждого узла
            profit = utilities[i] - costs[i]
            
            # Выбрать узел с максимальной выгодой
            best_node = np.argmax(profit)
            
            if profit[best_node] > 0:  # Только если выгодно
                allocation[i, best_node] = 1
            # Иначе устройство отклоняет все предложения
        
        return allocation
    
    def _compute_vcg_payments(
        self,
        allocation: np.ndarray,         # Binary matrix [m x n]: allocation[i][j] = 1 if device i uses edge j
        utility_matrix: np.ndarray,     # [m x n]: U[i][j]
        cost_matrix: np.ndarray,        # [m x n]: C[i][j]
    ) -> np.ndarray:
        """
        Вычислить платежи VCG
        
        Args:
            allocation: матрица размещения [m x n]
            utility_matrix: матрица полезности [m x n]
            cost_matrix: матрица стоимости [m x n]
        
        Returns:
            payments: вектор платежей [m]
        """
        m, _ = allocation.shape  # m - устройства, n - узлы
        
        # Социальное благосустояние с текущим распределением
        current_sw = self.compute_social_welfare(utility_matrix, cost_matrix, allocation)
                
        # Платежи VCG
        payments = np.zeros(m)

        for i in range(m):
            # SW без устройства i
            allocation_without_i = allocation.copy()
            allocation_without_i[i] = 0
            sw_without_i = self.compute_social_welfare(utility_matrix, cost_matrix, allocation_without_i)
            
            # Платёж = внешний эффект
            current_contribution = np.sum(allocation[i] * utility_matrix[i])
            payments[i] = sw_without_i - (current_sw - current_contribution)
        
        return payments
    
    def compute_social_welfare(
        self,
        utility_matrix: np.ndarray,
        cost_matrix: np.ndarray,
        allocation: np.ndarray
    ) -> float:
        """Социальное благосостояние"""
        return np.sum(allocation * utility_matrix) - np.sum(allocation * cost_matrix)
    
    def compute_gini_coefficient(self, payments: List[float]) -> float:
        """Коэффициент Джини для платежей"""
        payments = np.array(sorted(payments))
        n = len(payments)
        return (2 * np.sum((np.arange(1, n+1)) * payments)) / (n * np.sum(payments)) - (n + 1) / n

    def compute_fairness_index(self, allocation: np.ndarray) -> float:
        """Индекс справедливости Джини для распределений"""
        x = allocation.sum(axis=1)  # По устройствам
        return (x.sum() ** 2) / (len(x) * np.sum(x ** 2))
