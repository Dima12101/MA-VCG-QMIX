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
        с учетом ограничений ресурсов узлов.

        Для сценариев валидации из диссертации используется точный поиск
        branch-and-bound по устройствам. Это сохраняет корректность свойств
        статического раунда (эффективность и VCG-платежи) при умеренных
        размерах экземпляра.
        """
        m, n = utilities.shape
        allocation = np.zeros((m, n), dtype=int)
        net_values = np.asarray(utilities, dtype=float) - np.asarray(costs, dtype=float)

        if tasks is None or nodes is None:
            best_nodes = np.argmax(net_values, axis=1)
            for device_id in range(m):
                if excluded_device is not None and device_id == excluded_device:
                    continue
                best_node = int(best_nodes[device_id])
                best_value = float(net_values[device_id, best_node])
                if np.isfinite(best_value) and best_value > 0:
                    allocation[device_id, best_node] = 1
            return allocation

        remaining_cpu = tuple(max(0, node.cpu_available) for node in nodes)
        remaining_memory = tuple(max(0, node.memory_available) for node in nodes)

        device_options = {}
        ordered_devices: List[int] = []
        for device_id in range(m):
            if excluded_device is not None and device_id == excluded_device:
                continue
            task = tasks[device_id]
            if task is None:
                continue

            options = []
            for node_id in range(n):
                net_value = float(net_values[device_id, node_id])
                if not np.isfinite(net_value) or net_value <= 0:
                    continue
                if (
                    task.cpu_required > nodes[node_id].cpu_available
                    or task.memory_required > nodes[node_id].memory_available
                ):
                    continue
                options.append((node_id, net_value, task.cpu_required, task.memory_required))

            if options:
                options.sort(key=lambda option: option[1], reverse=True)
                device_options[device_id] = tuple(options)
                ordered_devices.append(device_id)

        if not ordered_devices:
            return allocation

        ordered_devices.sort(
            key=lambda device_id: device_options[device_id][0][1],
            reverse=True,
        )
        best_future_gain = [0.0] * (len(ordered_devices) + 1)
        for idx in range(len(ordered_devices) - 1, -1, -1):
            best_future_gain[idx] = (
                best_future_gain[idx + 1] + device_options[ordered_devices[idx]][0][1]
            )

        best_score = 0.0
        best_choices = np.full(m, -1, dtype=int)
        current_choices = np.full(m, -1, dtype=int)
        cpu_state = list(remaining_cpu)
        memory_state = list(remaining_memory)

        def search(idx: int, current_score: float) -> None:
            nonlocal best_score, best_choices
            if idx >= len(ordered_devices):
                if current_score > best_score + 1e-9:
                    best_score = current_score
                    best_choices = current_choices.copy()
                return

            # Admissible upper bound: each remaining device contributes at most
            # its best feasible net value in an unconstrained future.
            if current_score + best_future_gain[idx] <= best_score + 1e-9:
                return

            device_id = ordered_devices[idx]

            # Try allocations first to obtain a strong incumbent early.
            for node_id, net_value, cpu_req, mem_req in device_options[device_id]:
                if cpu_state[node_id] < cpu_req or memory_state[node_id] < mem_req:
                    continue
                current_choices[device_id] = node_id
                cpu_state[node_id] -= cpu_req
                memory_state[node_id] -= mem_req
                search(idx + 1, current_score + net_value)
                memory_state[node_id] += mem_req
                cpu_state[node_id] += cpu_req
                current_choices[device_id] = -1

            # Skipping the device remains a valid exact branch.
            search(idx + 1, current_score)

        search(0, 0.0)

        for device_id, node_id in enumerate(best_choices):
            if node_id >= 0:
                allocation[device_id, node_id] = 1
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
            chosen_nodes = allocation[i].astype(bool)
            current_contribution = float(
                np.sum(
                    np.asarray(utility_matrix[i], dtype=float)[chosen_nodes]
                    - np.asarray(cost_matrix[i], dtype=float)[chosen_nodes]
                )
            )
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
