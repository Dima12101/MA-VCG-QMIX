"""
Механизм MA-VCG (Multi-Agent VCG Auction)
"""

import numpy as np
from typing import List, Optional, Sequence, Tuple
from dataclasses import dataclass
from ..environment.edge_node import EdgeNode
from ..environment.task import Task

@dataclass
class AuctionResult:
    """Результат одного раунда аукциона"""
    allocation: np.ndarray  # [m x n]: устройства × узлы
    payments: np.ndarray    # [m]: платежи для каждого устройства
    social_welfare: float

class VCGAuctioneer:
    """Класс для проведения MA-VCG аукционера"""
    
    def __init__(
        self,
        num_devices: int,
        num_nodes: Optional[int] = None,
        num_edges: Optional[int] = None,
    ):
        self.num_devices = num_devices
        self.num_nodes = num_nodes if num_nodes is not None else num_edges
        self.num_edges = self.num_nodes

    def reset(self):
        self.history = []
    
    def run_auction(
        self,
        utilities: np.ndarray,      # [m x n]: полезность
        costs: np.ndarray,          # [m x n]: стоимость
        tasks: Optional[Sequence[Optional[Task]]] = None,
        nodes: Optional[Sequence[EdgeNode]] = None,
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
        allocation = self._compute_optimal_allocation(utilities, costs, tasks, nodes)
        social_welfare = self.compute_social_welfare(utilities, costs, allocation)
        
        # Фаза платежей (вычислить VCG платежи)
        payments = self._compute_vcg_payments(allocation, utilities, costs, tasks, nodes, social_welfare)
        
        result = AuctionResult(
            allocation=allocation,
            payments=payments,
            social_welfare=social_welfare,
        )
        
        return result
    
    def _compute_optimal_allocation(
        self,
        utilities: np.ndarray,
        costs: np.ndarray,
        tasks: Optional[Sequence[Optional[Task]]] = None,
        nodes: Optional[Sequence[EdgeNode]] = None,
        excluded_device: Optional[int] = None,
    ) -> np.ndarray:
        """
        Вычислить оптимальное распределение задач
        Используется жадная максимизация социального благосостояния
        с учетом ограничений ресурсов узлов.
        """
        m, n = utilities.shape
        allocation = np.zeros((m, n), dtype=int)

        remaining_cpu = [None] * n
        remaining_memory = [None] * n
        if nodes is not None:
            remaining_cpu = [node.cpu_available for node in nodes]
            remaining_memory = [node.memory_available for node in nodes]

        candidates = []
        for device_id in range(m):
            if excluded_device is not None and device_id == excluded_device:
                continue
            if tasks is not None and tasks[device_id] is None:
                continue
            for node_id in range(n):
                profit = float(utilities[device_id, node_id] - costs[device_id, node_id])
                if np.isfinite(profit) and profit > 0:
                    candidates.append((profit, float(utilities[device_id, node_id]), device_id, node_id))

        candidates.sort(reverse=True)
        assigned_devices = set()
        for _, _, device_id, node_id in candidates:
            if device_id in assigned_devices:
                continue
            if tasks is not None and nodes is not None:
                task = tasks[device_id]
                if task is None:
                    continue
                if remaining_cpu[node_id] < task.cpu_required or remaining_memory[node_id] < task.memory_required:
                    continue
                remaining_cpu[node_id] -= task.cpu_required
                remaining_memory[node_id] -= task.memory_required
            allocation[device_id, node_id] = 1
            assigned_devices.add(device_id)
        
        return allocation
    
    def _compute_vcg_payments(
        self,
        allocation: np.ndarray,         # Binary matrix [m x n]: allocation[i][j] = 1 if device i uses edge j
        utility_matrix: np.ndarray,     # [m x n]: U[i][j]
        cost_matrix: np.ndarray,        # [m x n]: C[i][j]
        tasks: Optional[Sequence[Optional[Task]]] = None,
        nodes: Optional[Sequence[EdgeNode]] = None,
        current_sw: Optional[float] = None,
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
        if current_sw is None:
            current_sw = self.compute_social_welfare(utility_matrix, cost_matrix, allocation)
                
        # Платежи VCG
        payments = np.zeros(m)

        for i in range(m):
            if allocation[i].sum() == 0:
                continue

            allocation_without_i = self._compute_optimal_allocation(
                utility_matrix,
                cost_matrix,
                tasks=tasks,
                nodes=nodes,
                excluded_device=i,
            )
            sw_without_i = self.compute_social_welfare(utility_matrix, cost_matrix, allocation_without_i)
            
            # Платёж = внешний эффект
            current_contribution = np.sum(allocation[i] * utility_matrix[i])
            payments[i] = max(0.0, sw_without_i - (current_sw - current_contribution))
        
        return payments
    
    def compute_social_welfare(
        self,
        utility_matrix: np.ndarray,
        cost_matrix: np.ndarray,
        allocation: np.ndarray
    ) -> float:
        """Социальное благосостояние"""
        chosen = allocation.astype(bool)
        if not np.any(chosen):
            return 0.0
        chosen_utilities = np.asarray(utility_matrix, dtype=float)[chosen]
        chosen_costs = np.asarray(cost_matrix, dtype=float)[chosen]
        return float(np.sum(chosen_utilities) - np.sum(chosen_costs))
    
    def compute_gini_coefficient(self, payments: List[float]) -> float:
        """Коэффициент Джини для платежей"""
        payments = np.clip(np.array(sorted(payments), dtype=float), 0.0, None)
        n = len(payments)
        total = payments.sum()
        if n == 0 or np.isclose(total, 0.0):
            return 0.0
        return (2 * np.sum((np.arange(1, n+1)) * payments)) / (n * np.sum(payments)) - (n + 1) / n

    def compute_fairness_index(self, allocation: np.ndarray) -> float:
        """Индекс справедливости Джини для распределений"""
        x = allocation.sum(axis=1).astype(float)  # По устройствам
        denominator = len(x) * np.sum(x ** 2)
        if np.isclose(denominator, 0.0):
            return 1.0
        return (x.sum() ** 2) / (len(x) * np.sum(x ** 2))
