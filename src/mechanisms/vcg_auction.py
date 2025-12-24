"""
Механизм MA-VCG (Multi-Agent VCG Auction)
Реализация повторяющегося аукциона Викри-Кларка-Гровса
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class AuctionResult:
    """Результат одного раунда аукциона"""
    allocation: np.ndarray  # [m x n]: устройства × узлы
    payments: np.ndarray    # [m]: платежи для каждого устройства
    social_welfare: float   # Суммарное благосустояние
    timestamp: int

class VCGAuction:
    """Класс для проведения MA-VCG аукциона"""
    
    def __init__(self, num_devices: int, num_edges: int):
        self.num_devices = num_devices
        self.num_edges = num_edges
        self.history: List[AuctionResult] = []
    
    def run_auction(
        self,
        valuations: np.ndarray,  # [m x n]: полезность
        costs: np.ndarray,       # [m x n]: стоимость
        timestamp: int
    ) -> AuctionResult:
        """
        Провести один раунд аукциона
        
        Args:
            valuations: матрица полезности [m x n]
            costs: матрица стоимости [m x n]
            timestamp: текущее время
        
        Returns:
            AuctionResult: результат аукциона
        """
        # Шаг 1: Фаза сбора (устройства "торгуются")
        # В реальности здесь была бы коммуникация с устройствами
        # Здесь используем полученные valuations
        
        # Шаг 2: Фаза оптимизации (вычислить оптимальное распределение)
        allocation = self._compute_optimal_allocation(valuations, costs)
        
        # Шаг 3: Фаза платежей (вычислить VCG платежи)
        payments, sw = self._compute_vcg_payments(allocation, valuations, costs)
        
        result = AuctionResult(
            allocation=allocation,
            payments=payments,
            social_welfare=sw,
            timestamp=timestamp
        )
        
        self.history.append(result)
        return result
    
    def _compute_optimal_allocation(
        self,
        valuations: np.ndarray,
        costs: np.ndarray
    ) -> np.ndarray:
        """
        Вычислить оптимальное распределение задач
        Используется жадный алгоритм для каждого устройства
        """
        m, n = valuations.shape
        allocation = np.zeros((m, n), dtype=int)
        
        # Для каждого устройства выбрать лучший узел
        for i in range(m):
            # Вычислить "выгоду" для каждого узла
            utility = valuations[i] - costs[i]
            
            # Выбрать узел с максимальной выгодой
            best_edge = np.argmax(utility)
            
            if utility[best_edge] > 0:  # Только если выгодно
                allocation[i, best_edge] = 1
            # Иначе устройство отклоняет все предложения
        
        return allocation
    
    def _compute_vcg_payments(
        self,
        allocation: np.ndarray,
        valuations: np.ndarray,
        costs: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Вычислить VCG платежи
        
        Returns:
            payments: вектор платежей [m]
            social_welfare: итоговое социальное благосустояние
        """
        m = allocation.shape
        
        # Социальное благосустояние с текущим распределением
        current_sw = np.sum(allocation * valuations) - np.sum(allocation * costs)
        
        payments = np.zeros(m)
        
        for i in range(m):
            # SW без устройства i
            allocation_without_i = allocation.copy()
            allocation_without_i[i] = 0
            
            sw_without_i = (np.sum(allocation_without_i * valuations) - 
                           np.sum(allocation_without_i * costs))
            
            # Платёж = внешний эффект
            current_contribution = np.sum(allocation[i] * valuations[i])
            payments[i] = sw_without_i - (current_sw - current_contribution)
        
        return payments, current_sw
    
    def get_average_gini(self) -> float:
        """Получить среднее значение Джини платежей"""
        if not self.history:
            return 0.0
        
        ginis = []
        for result in self.history:
            payments = result.payments[result.payments > 0]  # Только положительные
            if len(payments) > 1:
                gini = self._compute_gini(payments)
                ginis.append(gini)
        
        return np.mean(ginis) if ginis else 0.0
    
    @staticmethod
    def _compute_gini(values: np.ndarray) -> float:
        """Вычислить коэффициент Джини"""
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        cumsum = np.cumsum(sorted_vals)
        return (2 * np.sum((np.arange(1, n+1)) * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n
