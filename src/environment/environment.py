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

    # Возможные действия агента
    ACTION_ACCEPT = 0
    ACTION_REJECT = 1
    ACTION_PRIORITY_HIGH = 2
    ACTION_PRIORITY_LOW = 3

    # Размерность локального наблюдения агента
    ACTION_SIZE = 4
    BASE_OBSERVATION_SIZE = 5
    AUCTION_FEATURE_SIZE = 4
    
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
        self._pending_round: Optional[Dict] = None
        self.last_arrival_rate = 0.0
        self.last_load_spike_active = False
        
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
        self._pending_round = None
        self.last_arrival_rate = 0.0
        self.last_load_spike_active = False
        [device.reset() for device in self.devices]
        [node.reset() for node in self.nodes]

    @property
    def observation_size(self) -> int:
        return self.BASE_OBSERVATION_SIZE + self.AUCTION_FEATURE_SIZE + self.ACTION_SIZE

    def _ensure_round_prepared(self):
        if self._pending_round is not None:
            return

        self.current_time += 1
        self._update_node_failures()
        self.tasks = {}

        # Фаза 1: Устройства отправляют заявки на выполнение задач.
        self._generate_tasks()

        # Фаза 2: Оценка полезности и стоимости для текущего пула.
        utility_matrix, cost_matrix = self._evaluate_tasks()
        task_vector = self._build_task_vector()

        # Фаза 3.1: Аукционный контур формирует allocation, payments и action masks
        auction_result = self.auctioneer.run_auction(
            utility_matrix,
            cost_matrix,
            tasks=task_vector,
            nodes=self.nodes,
        )
        action_masks = self._build_action_masks(auction_result.allocation)
        auction_features = self._build_auction_features(
            allocation=auction_result.allocation,
            payments=auction_result.payments,
        )
        observations = self._compose_observations(action_masks, auction_features)

        self._pending_round = {
            "tasks": task_vector,
            "utility_matrix": utility_matrix,
            "cost_matrix": cost_matrix,
            "auction_result": auction_result,
            "action_masks": action_masks,
            "auction_features": auction_features,
            "observations": observations,
        }
    
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
        self._ensure_round_prepared()
        pending_round = self._pending_round
        utility_matrix = pending_round["utility_matrix"]
        cost_matrix = pending_round["cost_matrix"]
        auction_result = pending_round["auction_result"]

        accepted_tasks, rejected_tasks, assignments, realized_allocation, realized_payments = self._distribute_tasks(
            auction_result.allocation,
            auction_result.payments,
            utility_matrix,
            cost_matrix,
            actions,
        )

        # Фаза 3.2: Исполнение очередей и сбор фактических метрик.
        completed_tasks, latencies = self._execute_tasks()
        rewards = self._compute_rewards(
            accepted_tasks=accepted_tasks,
            rejected_tasks=rejected_tasks,
            completed_tasks=completed_tasks,
            realized_payments=realized_payments,
            utility_matrix=utility_matrix,
            cost_matrix=cost_matrix,
            realized_allocation=realized_allocation,
        )
        info = {
            'accepted': len(accepted_tasks),
            'rejected': len(rejected_tasks),
            'completed': len(completed_tasks),
            'generated': len(self.tasks),
        }

        completed_before_deadline = sum(
            1
            for task, latency in zip(completed_tasks, latencies)
            if latency <= task.deadline
        )
        load_imbalance = float(np.std([node.load for node in self.nodes]))
        backlog_pressure = self._mean_backlog_pressure()
        failed_nodes = int(sum(node.is_failed for node in self.nodes))
        stress_context = float(self.last_load_spike_active)
        
        # Определение метрик
        metrics = {
            # Метрики производительности
            'avg_latency': float(np.mean(latencies)) if latencies else 0.0,
            'acceptance_rate': info['accepted'] / max(len(self.tasks), 1) * 100,
            'drop_rate': info['rejected'] / max(len(self.tasks), 1),
            'resource_utilization': [node.load for node in self.nodes],
            'load_imbalance': load_imbalance,
            'backlog_pressure': backlog_pressure,
            'deadline_success_rate': (
                completed_before_deadline / max(len(completed_tasks), 1)
                if completed_tasks
                else 0.0
            ),
            'completed_before_deadline': completed_before_deadline,
            'deadline_violations': max(len(completed_tasks) - completed_before_deadline, 0),
            # Метрики справедливости
            'gini_payment': self.auctioneer.compute_gini_coefficient(realized_payments),
            'fairness_index': self.auctioneer.compute_fairness_index(realized_allocation),
            # Метрики оптимальности
            'social_welfare': self.auctioneer.compute_social_welfare(
                utility_matrix,
                cost_matrix,
                realized_allocation,
            ),
            'arrival_rate': self.last_arrival_rate,
            'load_spike_active': float(self.last_load_spike_active),
            'failed_nodes': failed_nodes,
            'stress_context': stress_context,
        }
        self._pending_round = None

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
        spike_active = any(
            start_step <= self.current_time <= end_step
            for start_step, end_step in self.env_config.load_spike_windows
        )
        if spike_active or random.random() < self.env_config.load_spike_probability:
            base_rate *= self.env_config.load_spike_multiplier
            spike_active = True
        self.last_arrival_rate = max(0.0, base_rate)
        self.last_load_spike_active = spike_active
        return self.last_arrival_rate

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
                    utility_scale=self.task_config.utility_scale,
                    utility_cpu_weight=self.task_config.utility_cpu_weight,
                    utility_memory_weight=self.task_config.utility_memory_weight,
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

    def _build_action_masks(self, allocation: np.ndarray) -> np.ndarray:
        """Построить маски допустимых действий для каждого узла."""
        masks = np.zeros((len(self.nodes), self.ACTION_SIZE), dtype=np.float32)
        for node in self.nodes:
            assigned_count = int(np.sum(allocation[:, node.id]))
            if node.is_failed:
                masks[node.id, self.ACTION_REJECT] = 1.0
                continue
            if assigned_count == 0:
                masks[node.id, self.ACTION_ACCEPT] = 1.0
                continue
            masks[node.id, :] = 1.0
        return masks

    def _estimated_service_time_ms(self, task: Task, node_id: int) -> float:
        """Оценить полное время обслуживания задачи на конкретном узле."""
        latency, _ = self._device_node_latency(task.device_id, node_id)
        if not np.isfinite(latency):
            return float("inf")
        return task.processing_time_ms(self.nodes[node_id].cpu_capacity) + latency

    def _node_processing_backlog_ms(self, node_id: int) -> float:
        """Оценить накопленный backlog на узле по текущим очередям и выполнению."""
        node = self.nodes[node_id]
        if node.is_failed:
            return float("inf")

        running_ms = sum(
            max(0, remaining_steps) * max(self.env_config.step_duration_ms, 1)
            for remaining_steps in node.task_executed_time.values()
        )
        queued_ms = sum(
            task.processing_time_ms(node.cpu_capacity)
            for task in node.task_queue.values()
        )
        return float(running_ms + queued_ms)

    def _projected_completion_ms(
        self,
        task: Task,
        node_id: int,
        backlog_ms: float,
    ) -> float:
        """Оценить полное время завершения новой задачи с учетом накопленного backlog."""
        service_time_ms = self._estimated_service_time_ms(task, node_id)
        if not np.isfinite(service_time_ms) or not np.isfinite(backlog_ms):
            return float("inf")
        return float(backlog_ms + service_time_ms)

    def _mean_backlog_pressure(self) -> float:
        """Усредненная backlog-нагрузка через projected completion ratio."""
        per_node_pressure = []
        for node in self.nodes:
            if node.is_failed:
                per_node_pressure.append(1.0)
                continue

            backlog_ms = sum(
                max(0, remaining_steps) * max(self.env_config.step_duration_ms, 1)
                for remaining_steps in node.task_executed_time.values()
            )
            queued_tasks = sorted(
                node.task_queue.values(),
                key=lambda task: (-task.priority.value, task.deadline, task.id),
            )
            if not queued_tasks:
                per_node_pressure.append(min(node.load, 1.0))
                continue

            projected_ratios = []
            for task in queued_tasks:
                projected_completion_ms = backlog_ms + task.processing_time_ms(node.cpu_capacity)
                projected_ratios.append(
                    projected_completion_ms / max(task.deadline, 1.0)
                )
                backlog_ms = projected_completion_ms
            per_node_pressure.append(float(min(np.mean(projected_ratios), 2.0)))

        return float(np.mean(per_node_pressure)) if per_node_pressure else 0.0

    def _build_auction_features(
        self,
        allocation: np.ndarray,
        payments: np.ndarray,
    ) -> np.ndarray:
        """Сформировать локальный аукционный контекст для наблюдений агентов."""
        features = np.zeros((len(self.nodes), self.AUCTION_FEATURE_SIZE), dtype=np.float32)
        payment_scale = max(float(np.max(payments)) if payments.size else 0.0, 1.0)
        device_scale = max(self.env_config.num_devices, 1)

        for node in self.nodes:
            assigned_device_ids = np.flatnonzero(allocation[:, node.id])
            if assigned_device_ids.size == 0:
                continue

            assigned_tasks = [self.tasks[int(device_id)] for device_id in assigned_device_ids]
            assigned_tasks.sort(key=lambda task: (task.deadline, task.id))
            total_cpu = sum(task.cpu_required for task in assigned_tasks)
            backlog_ms = self._node_processing_backlog_ms(node.id)
            projected_ratios = []
            for task in assigned_tasks:
                projected_completion_ms = self._projected_completion_ms(
                    task,
                    node.id,
                    backlog_ms,
                )
                projected_ratios.append(
                    min(2.5, projected_completion_ms / max(task.deadline, 1.0))
                )
                backlog_ms += task.processing_time_ms(node.cpu_capacity)
            total_service_ratio = float(np.mean(projected_ratios)) if projected_ratios else 0.0

            features[node.id] = np.array(
                [
                    len(assigned_tasks) / device_scale,
                    float(np.mean(payments[assigned_device_ids])) / payment_scale,
                    min(total_cpu / max(node.cpu_capacity, 1), 1.5),
                    float(total_service_ratio),
                ],
                dtype=np.float32,
            )

        return features

    def _compose_observations(
        self,
        action_masks: np.ndarray,
        auction_features: np.ndarray,
    ) -> np.ndarray:
        """Собрать расширенные наблюдения из локального состояния и аукционных сигналов."""
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

            base_obs = [
                node.cpu_available / max(node.cpu_capacity, 1),
                node.memory_available / max(node.memory_capacity, 1),
                min((len(node.task_queue) + len(node.task_executed)) / 10.0, 1.0),
                neighbor_load,
                1.0 if node.is_failed else 0.0,
            ]
            obs = np.concatenate(
                [
                    np.array(base_obs, dtype=np.float32),
                    auction_features[node.id],
                    action_masks[node.id],
                ]
            )
            observations.append(obs)

        return np.array(observations, dtype=np.float32)
    
    def _distribute_tasks(
        self,
        allocation: np.ndarray,
        payments: np.ndarray,
        utility_matrix: np.ndarray,
        cost_matrix: np.ndarray,
        actions: List[int],
    ) -> Tuple[List[Task], List[Task], Dict[int, List[Task]], np.ndarray, np.ndarray]:
        """Назначение задач согласно распределению и стратегий узлов."""
        accepted_tasks = []
        rejected_tasks = []
        assignments = {node.id: [] for node in self.nodes}
        realized_allocation = np.zeros_like(allocation)
        realized_payments = np.zeros_like(payments)

        device_ids_by_node = {node.id: [] for node in self.nodes}
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
            device_ids_by_node[node_id].append(device_id)

        node_backlog_ms = {
            node.id: self._node_processing_backlog_ms(node.id)
            for node in self.nodes
        }
        for node in self.nodes:
            action = int(actions[node.id])
            device_ids = sorted(
                device_ids_by_node[node.id],
                key=lambda device_id: (
                    self.tasks[device_id].deadline,
                    self.tasks[device_id].id,
                ),
            )
            if not device_ids:
                continue

            if action == self.ACTION_REJECT or node.is_failed:
                for device_id in device_ids:
                    task = self.tasks[device_id]
                    self.devices[device_id].task_rejected(task)
                    rejected_tasks.append(task)
                continue

            for device_id in device_ids:
                task = self.tasks[device_id]
                projected_completion_ms = self._projected_completion_ms(
                    task,
                    node.id,
                    node_backlog_ms[node.id],
                )
                projected_service_ratio = projected_completion_ms / max(task.deadline, 1.0)

                # LOW-priority mode becomes a selective safety valve: accept only
                # the tasks that still look feasible under the current backlog.
                if (
                    action == self.ACTION_PRIORITY_LOW
                    and projected_service_ratio > 1.20
                ):
                    self.devices[device_id].task_rejected(task)
                    rejected_tasks.append(task)
                    continue

                if action == self.ACTION_PRIORITY_HIGH:
                    task.priority = TaskPriority.HIGH
                elif action == self.ACTION_PRIORITY_LOW:
                    task.priority = TaskPriority.LOW

                latency, _ = self._device_node_latency(device_id, node.id)
                task.assigned_node_id = node.id
                task.allocated_payment = float(payments[device_id])
                task.welfare_contribution = float(
                    utility_matrix[device_id, node.id] - cost_matrix[device_id, node.id]
                )
                task.allocation_latency_ms = 0.0 if not np.isfinite(latency) else float(latency)
                task.projected_completion_ms = projected_completion_ms
                task.projected_service_ratio = float(projected_service_ratio)
                self.nodes[node.id].accept_task(task)
                self.devices[device_id].record_payment(payments[device_id])
                accepted_tasks.append(task)
                assignments[node.id].append(task)
                realized_allocation[device_id, node.id] = 1
                realized_payments[device_id] = payments[device_id]
                node_backlog_ms[node.id] += task.processing_time_ms(node.cpu_capacity)

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
        accepted_tasks: List[Task],
        rejected_tasks: List[Task],
        completed_tasks: List[Task],
        realized_payments: np.ndarray,
        utility_matrix: np.ndarray,
        cost_matrix: np.ndarray,
        realized_allocation: np.ndarray,
    ) -> np.ndarray:
        social_welfare = self.auctioneer.compute_social_welfare(
            utility_matrix,
            cost_matrix,
            realized_allocation,
        )
        fairness_index = self.auctioneer.compute_fairness_index(realized_allocation)
        gini_coefficient = self.auctioneer.compute_gini_coefficient(realized_payments)
        deadline_violations = sum(
            1
            for task in accepted_tasks
            if task.projected_service_ratio > 1.0
        )

        drop_rate = len(rejected_tasks) / max(len(self.tasks), 1)
        violation_rate = deadline_violations / max(len(accepted_tasks), 1) if accepted_tasks else 0.0
        load_imbalance = float(np.std([node.load for node in self.nodes]))
        backlog_pressure = self._mean_backlog_pressure()
        stress_context = float(self.last_load_spike_active)
        completed_payments = np.array(
            [task.allocated_payment for task in completed_tasks],
            dtype=float,
        )
        completed_welfare = np.array(
            [task.welfare_contribution for task in completed_tasks],
            dtype=float,
        )
        return self.reward_manager.combine_rewards(
            social_welfare=social_welfare,
            fairness_index=fairness_index,
            gini_coefficient=gini_coefficient,
            deadline_violation_rate=violation_rate,
            drop_rate=drop_rate,
            load_imbalance=load_imbalance,
            completed_payments=completed_payments,
            completed_welfare=completed_welfare,
            backlog_pressure=backlog_pressure,
            stress_context=stress_context,
        )
    
    def get_observations(self) -> np.ndarray:
        """Вернуть расширенные наблюдения для следующего решения."""
        self._ensure_round_prepared()
        return np.array(self._pending_round["observations"], copy=True)
