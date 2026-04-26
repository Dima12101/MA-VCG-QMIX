import random
import numpy as np
import networkx as nx
from dataclasses import replace
from typing import Dict, List, Optional, Tuple
from .task import Task, TaskPriority
from .edge_node import EdgeNode
from .device import Device
from ..config import EnvironmentConfig, NodeConfig, TaskConfig, AuctionConfig
from ..mechanisms.auction import VCGAuctioneer
from ..mechanisms.evaluation import evaluate_utility, evaluate_cost
from ..learning.reward_manager import RewardManager

# --------------------------------------------------------------------------- #
#                        EDGE-COMPUTING SYSTEM SIMULATOR                      #
# --------------------------------------------------------------------------- #

class EdgeComputingSystem:
    """Класс для представления моделируемой среды"""

    ACTION_ACCEPT = 0
    ACTION_REJECT = 1
    ACTION_PRIORITY_HIGH = 2
    ACTION_PRIORITY_LOW = 3
    
    def __init__(self, 
                 env_config: EnvironmentConfig,
                 node_config: NodeConfig,
                 task_config: TaskConfig,
                 auction_config: AuctionConfig):
        self.env_config = env_config
        self.node_config = node_config
        self.task_config = task_config
        self.auction_config = auction_config

        self.current_time = 0 # step
        self.task_counter = 0
        self.tasks = {}
        self.failures_triggered = False
        
        # Инициализировать узлы и устройства
        self._initialize_network()

        # Создать VCG аукционера
        self.auctioneer = VCGAuctioneer(self.env_config.num_devices, self.env_config.num_nodes)

        # Создать менеджера наград
        self.reward_manager = RewardManager(
            self.env_config.num_nodes,
            self.env_config.num_devices,
            self.auction_config.vcg_weight,
            global_weight=self.auction_config.global_reward_weight,
            fairness_target=self.auction_config.fairness_target,
            gini_target=self.auction_config.gini_target,
        )

    @staticmethod
    def _graph_node_id(node_id: int) -> Tuple[str, int]:
        return ("node", node_id)

    @staticmethod
    def _graph_device_id(device_id: int) -> Tuple[str, int]:
        return ("device", device_id)

    def _sample_node_config(self) -> NodeConfig:
        """Получить конфигурацию узла с учетом возможной неоднородности."""
        if not self.node_config.heterogeneous_resources:
            return replace(self.node_config)
        return replace(
            self.node_config,
            cpu_capacity=random.randint(*self.node_config.cpu_capacity_range),
            memory_capacity=random.randint(*self.node_config.memory_capacity_range),
        )
    
    def _initialize_network(self):
        """Инициализировать сеть"""
        
        # Создать устройства с распределением важности
        alpha = self.env_config.device_importance.get("alpha", 2.0)
        beta = self.env_config.device_importance.get("beta", 5.0)
        device_importance_dist = np.random.beta(alpha, beta, self.env_config.num_devices)
        self.devices: List[Device] = [
            Device(i, importance=float(device_importance_dist[i]))
            for i in range(self.env_config.num_devices)
        ]

        # Создать edge-узлы
        self.nodes: List[EdgeNode] = [
            EdgeNode(j, self._sample_node_config())
            for j in range(self.env_config.num_nodes)
        ]

        # Создать сеть
        self.network = nx.Graph()
        for node in self.nodes:
            self.network.add_node(self._graph_node_id(node.id), type="node")
        for device in self.devices:
            self.network.add_node(self._graph_device_id(device.id), type="device")

        # node-to-node links
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                if random.random() < self.env_config.node_to_node_probability:               
                    w = random.uniform(self.env_config.node_to_node_weight['min'], self.env_config.node_to_node_weight['max'])
                    self.network.add_edge(
                        self._graph_node_id(self.nodes[i].id),
                        self._graph_node_id(self.nodes[j].id),
                        weight=w,
                    )
        # device-to-node links
        for d in self.devices:
            connected = random.sample(self.nodes, k=random.randint(1, min(self.env_config.device_to_node_max_edges, len(self.nodes))))
            for n in connected:
                w = random.uniform(self.env_config.device_to_node_weight['min'], self.env_config.device_to_node_weight['max'])
                self.network.add_edge(
                    self._graph_device_id(d.id),
                    self._graph_node_id(n.id),
                    weight=w,
                )
    
    def reset(self):
        """Initialize environment for new episode."""
        self.current_time = 0
        self.task_counter = 0
        self.tasks = {}
        self.failures_triggered = False
        [device.reset() for device in self.devices]
        [node.reset() for node in self.nodes]
    
    def step(self, actions: List[int]) -> Tuple[np.ndarray, Dict, Dict]:
        """
        Execute one simulation step.
        
        Args:
            actions: List of actions for each node
            
        Returns:
            (observations, reward, info)
        """
        if len(actions) != len(self.nodes):
            raise ValueError("Число действий должно совпадать с числом edge-узлов.")

        self.current_time += 1
        self._update_node_failures()
        self.tasks = {}

        # Фаза 1: Устройства отправляют заявку на выполнение задачи
        self._generate_tasks()

        # Фаза 2: Оценка полезности и стоимости выполнения задач
        utility_matrix, cost_matrix = self._evaluate_tasks()

        # Фаза 3.1: Распределение задач
        auction_result = self.auctioneer.run_auction(
            utility_matrix,
            cost_matrix,
            tasks=self._build_task_vector(),
            nodes=self.nodes,
        )
        accepted_tasks, rejected_tasks, assignments, realized_allocation, realized_payments = self._distribute_tasks(
            auction_result.allocation,
            auction_result.payments,
            actions,
        )
        
        # Фаза 3.2: Определение наград
        rewards = self._compute_rewards(
            assignments,
            realized_payments,
            utility_matrix,
            cost_matrix,
            realized_allocation,
        )
        
        # Выполнение задач
        completed_tasks, latencies = self._execute_tasks()
        info = {
            'accepted': len(accepted_tasks),
            'rejected': len(rejected_tasks),
            'completed': len(completed_tasks),
        }
        
        # Определение метрик
        metrics = {
            # Метрики производительности
            'avg_latency': float(np.mean(latencies)) if latencies else 0.0,
            'acceptance_rate': info['accepted'] / max(len(self.tasks), 1) * 100,
            'resource_utilization': [node.load for node in self.nodes],
            # Метрики справедливости
            'gini_payment': self.auctioneer.compute_gini_coefficient(realized_payments),
            'fairness_index': self.auctioneer.compute_fairness_index(realized_allocation),
            # Метрики оптимальности
            'social_welfare': self.auctioneer.compute_social_welfare(
                utility_matrix,
                cost_matrix,
                realized_allocation,
            ),
        }

        return rewards, info, metrics

    def _update_node_failures(self):
        """Активировать и завершать отказы узлов согласно сценарию."""
        for node in self.nodes:
            node.update_failure_state(self.current_time)

        if (
            self.failures_triggered
            or self.env_config.failure_fraction <= 0
            or self.env_config.failure_start_step <= 0
            or self.current_time < self.env_config.failure_start_step
        ):
            return

        num_failed = max(1, int(round(self.env_config.num_nodes * self.env_config.failure_fraction)))
        candidates = [node for node in self.nodes if not node.is_failed]
        for node in random.sample(candidates, k=min(num_failed, len(candidates))):
            dropped_tasks = node.fail(self.current_time + self.env_config.failure_recovery_steps)
            for task in dropped_tasks:
                self.devices[task.device_id].task_rejected(task)
        self.failures_triggered = True

    def _current_arrival_rate(self) -> float:
        """Текущая интенсивность прибытия задач."""
        phase = 2 * np.pi * self.current_time / max(self.env_config.arrival_cycle, 1)
        base_rate = self.env_config.task_lambda_arrival * (
            1 + self.env_config.arrival_amplitude * np.sin(phase)
        )
        if random.random() < self.env_config.load_spike_probability:
            base_rate *= self.env_config.load_spike_multiplier
        return max(0.0, base_rate)

    def _generate_tasks(self):
        """Сгенерировать новые задачи по Пуассоновскому распределению."""
        arrival_rate = self._current_arrival_rate()
        per_device_rate = arrival_rate / max(self.env_config.num_devices, 1)
        for device_id in range(self.env_config.num_devices):
            if np.random.poisson(per_device_rate) > 0:
                task = Task(
                    id=self.task_counter,
                    device_id=device_id,
                    cpu_required=random.randint(self.task_config.cpu['min'], self.task_config.cpu['max']),
                    memory_required=random.randint(self.task_config.memory['min'], self.task_config.memory['max']),
                    data_size=random.randint(self.task_config.data_size['min'], self.task_config.data_size['max']),
                    deadline=random.randint(self.task_config.deadline['min'], self.task_config.deadline['max']),
                    arrival_time=self.current_time,
                    importance=self.devices[device_id].importance,
                )
                self.devices[device_id].submit_task(task)
                self.tasks[device_id] = task
                self.task_counter += 1

    def _build_task_vector(self) -> List[Optional[Task]]:
        """Преобразовать пул задач в индексируемый по устройствам список."""
        return [self.tasks.get(device.id) for device in self.devices]

    def _device_node_latency(self, device_id: int, node_id: int) -> Tuple[float, bool]:
        """Оценить сетевую задержку между устройством и узлом."""
        source = self._graph_device_id(device_id)
        target = self._graph_node_id(node_id)
        if not nx.has_path(self.network, source, target):
            return float("inf"), False
        latency = nx.shortest_path_length(self.network, source, target, weight="weight")
        return float(latency), self.network.has_edge(source, target)
    
    def _evaluate_tasks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Составить матрицу полезностей и стоимостей относительной текущего пула задач."""
        utility_matrix = []
        cost_matrix = []
        for device in self.devices:
            if device.id not in self.tasks:
                utility_matrix.append([0.0 for _ in self.nodes])
                cost_matrix.append([0.0 for _ in self.nodes])
                continue

            task = self.tasks[device.id]
            utility_row = []
            cost_row = []
            for node in self.nodes:
                if node.is_failed:
                    utility_row.append(0.0)
                    cost_row.append(float("inf"))
                    continue
                latency, direct_connection = self._device_node_latency(device.id, node.id)
                if not np.isfinite(latency):
                    utility_row.append(0.0)
                    cost_row.append(float("inf"))
                    continue
                utility_row.append(evaluate_utility(task, node, latency))
                cost_row.append(evaluate_cost(task, node, latency, direct_connection))
            utility_matrix.append(utility_row)
            cost_matrix.append(cost_row)

        return np.array(utility_matrix, dtype=float), np.array(cost_matrix, dtype=float)
    
    def _distribute_tasks(
        self,
        allocation: np.ndarray,
        payments: np.ndarray,
        actions: List[int],
    ) -> Tuple[List[Task], List[Task], Dict[int, List[Task]], np.ndarray, np.ndarray]:
        """Назначение задач согласно распределению и стратегий узлов."""
        accepted_tasks = []
        rejected_tasks = []
        assignments = {node.id: [] for node in self.nodes}
        realized_allocation = np.zeros_like(allocation)
        realized_payments = np.zeros_like(payments)

        for device_id, device_allocation in enumerate(allocation):
            if device_id not in self.tasks:
                continue

            task = self.tasks[device_id]
            chosen_nodes = np.flatnonzero(device_allocation)
            if chosen_nodes.size == 0:
                self.devices[device_id].task_rejected(task)
                rejected_tasks.append(task)
                continue

            node_id = int(chosen_nodes[0])
            action = int(actions[node_id])
            if action == self.ACTION_REJECT or self.nodes[node_id].is_failed:
                self.devices[device_id].task_rejected(task)
                rejected_tasks.append(task)
                continue

            if action == self.ACTION_PRIORITY_HIGH:
                task.priority = TaskPriority.HIGH
            elif action == self.ACTION_PRIORITY_LOW:
                task.priority = TaskPriority.LOW

            self.nodes[node_id].accept_task(task)
            self.devices[device_id].record_payment(payments[device_id])
            accepted_tasks.append(task)
            assignments[node_id].append(task)
            realized_allocation[device_id, node_id] = 1
            realized_payments[device_id] = payments[device_id]

        return accepted_tasks, rejected_tasks, assignments, realized_allocation, realized_payments

    def _execute_tasks(self) -> Tuple[List[Task], List[float]]:
        completed_tasks = []
        latencies = []
        for node in self.nodes:
            node_completed_tasks, node_latencies = node.step(
                self.current_time,
                self.env_config.step_duration_ms,
            )
            for task in node_completed_tasks:
                self.devices[task.device_id].task_completed(task)
            completed_tasks.extend(node_completed_tasks)
            latencies.extend(node_latencies)

        return completed_tasks, latencies
    
    def _compute_rewards(
        self,
        assignments: Dict[int, List[Task]],
        realized_payments: np.ndarray,
        utility_matrix: np.ndarray,
        cost_matrix: np.ndarray,
        realized_allocation: np.ndarray,
    ) -> np.ndarray:
        node_values = np.zeros(len(self.nodes), dtype=float)
        node_revenues = np.zeros(len(self.nodes), dtype=float)
        node_costs = np.zeros(len(self.nodes), dtype=float)
        node_penalties = np.zeros(len(self.nodes), dtype=float)

        for node_id, assigned_tasks in assignments.items():
            node = self.nodes[node_id]
            for task in assigned_tasks:
                node_values[node_id] += utility_matrix[task.device_id, node_id]
                node_revenues[node_id] += realized_payments[task.device_id]
                node_costs[node_id] += cost_matrix[task.device_id, node_id]
                estimated_time_ms = task.processing_time_ms(node.cpu_capacity)
                if estimated_time_ms > task.deadline:
                    node_penalties[node_id] += (estimated_time_ms - task.deadline) / max(task.deadline, 1.0)

        local_rewards = np.array([
            self.reward_manager.compute_local_reward(
                local_value=node_values[node.id],
                operational_cost=node_costs[node.id],
                sla_penalty=node_penalties[node.id],
            )
            for node in self.nodes
        ], dtype=float)

        social_welfare = self.auctioneer.compute_social_welfare(
            utility_matrix,
            cost_matrix,
            realized_allocation,
        )
        fairness_index = self.auctioneer.compute_fairness_index(realized_allocation)
        gini_coefficient = self.auctioneer.compute_gini_coefficient(realized_payments)
        return self.reward_manager.combine_rewards(
            local_rewards=local_rewards,
            node_revenues=node_revenues,
            social_welfare=social_welfare,
            fairness_index=fairness_index,
            gini_coefficient=gini_coefficient,
        )
    
    def get_observations(self) -> np.ndarray:
        """Get local observations for each node."""
        observations = []
        for node in self.nodes:
            neighbors = [
                neighbor_id
                for neighbor_type, neighbor_id in self.network.neighbors(self._graph_node_id(node.id))
                if neighbor_type == "node"
            ]
            neighbor_load = 0.0
            if neighbors:
                neighbor_load = float(np.mean([self.nodes[n_id].load for n_id in neighbors]))

            obs = [
                node.cpu_available / node.cpu_capacity,
                node.memory_available / node.memory_capacity,
                min(len(node.task_queue) / 10.0, 1.0),
                neighbor_load,
                1.0 if node.is_failed else 0.0,
            ]
            observations.append(obs)
        
        return np.array(observations, dtype=np.float32)
