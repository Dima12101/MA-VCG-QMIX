import random
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
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
        
        # Инициализировать узлы и устройства
        self._initialize_network()

        # Создать VCG аукционера
        self.auctioneer = VCGAuctioneer(self.env_config.num_devices, self.env_config.num_nodes)

        # Создать менеджера наград
        self.reward_manager = RewardManager(self.env_config.num_nodes, self.env_config.num_devices, self.auction_config.vcg_weight)

    
    def _initialize_network(self):
        """Инициализировать сеть"""
        
        # Создать устройства с распределением важности
        device_importance_dist = np.random.beta(self.env_config.device_importance['min'],
                                         self.env_config.device_importance['max'],
                                         self.env_config.num_devices)
        self.devices: List[Device] = [
            Device(i, importance=float(device_importance_dist[i]))
            for i in range(self.env_config.num_devices)
        ]

        # Создать edge-узлы
        self.nodes: List[EdgeNode] = [
            EdgeNode(j, self.node_config) 
            for j in range(self.env_config.num_nodes)
        ]

        # Создать сеть
        self.network = nx.Graph()
        [self.network.add_node(node.id, type="node") for node in self.nodes] # Добавление edge-узлов в сеть
        [self.network.add_node(device.id, type="device") for device in self.devices] # Добавление устройств в сеть
        # node-to-node links
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                if random.random() < self.env_config.node_to_node_probability:               
                    w = random.uniform(self.env_config.node_to_node_weight['min'], self.env_config.node_to_node_weight['max'])
                    self.network.add_edge(self.nodes[i].id, self.nodes[j].id, weight=w)
        # device-to-node links
        for d in self.devices:
            connected = random.sample(self.nodes, k=random.randint(1, min(self.env_config.device_to_node_max_edges, len(self.nodes))))
            for n in connected:
                w = random.uniform(self.env_config.device_to_node_weight['min'], self.env_config.device_to_node_weight['max'])
                self.network.add_edge(d.id, n.id, weight=w)          
    
    def reset(self):
        """Initialize environment for new episode."""
        self.current_time = 0
        self.task_counter = 0
        self.tasks = {}
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
        self.current_time += 1

        # Фаза 1: Устройства отправляют заявку на выполнение задачи
        self._generate_tasks()

        # Фаза 2: Оценка полезности и стоимости выполнения задач
        utility_matrix, cost_matrix = self._evaluate_tasks()

        # Фаза 3.1: Распределение задач
        auction_result = self.auctioneer.run_auction(utility_matrix, cost_matrix)
        accepted_tasks, rejected_tasks = self._distribute_tasks(auction_result.allocation, actions)
        
        # Фаза 3.2: Определение наград TODO
        rewards = self._compute_rewards(accepted_tasks, auction_result.payments)
        
        # Выполнение задач
        completed, latencies = self._execute_tasks()
        info = {
            'accepted': len(accepted_tasks),
            'rejected': len(rejected_tasks),
            'completed': completed,
        }
        
        # Определение метрик
        metrics = {
            # Метрики производительности
            'avg_latency': np.mean(latencies),
            'acceptance_rate': info['accepted'] /  len(self.tasks) * 100,
            'resource_utilization': [node.load for node in self.nodes],
            # Метрики справедливости
            'gini_payment': self.auctioneer.compute_gini_coefficient(auction_result.payments),
            'fairness_index': self.auctioneer.compute_fairness_index(auction_result.allocation),
            # Метрики оптимальности
            'social_welfare': self.auctioneer.compute_social_welfare(utility_matrix, cost_matrix, auction_result.allocation) # TODO
        }

        return rewards, info, metrics

    def _generate_tasks(self):
        """Сгенерировать новые задачи по Пуассоновскому распределению."""
        arrival_rate = 0.3 + 0.2 * np.sin(2 * np.pi * self.current_time / 100)
        for device_id in range(self.env_config.num_devices):
            if np.random.rand() < arrival_rate:    
                task = Task(
                    id=self.task_counter,
                    device_id=device_id,
                    cpu_required=random.randint(self.task_config.cpu['min'], self.task_config.cpu['max']),
                    memory_required=random.randint(self.task_config.memory['min'], self.task_config.memory['max']),
                    bandwidth_required =random.randint(self.task_config.bandwidth['min'], self.task_config.bandwidth['max']),
                    data_size=random.randint(self.task_config.data_size['min'], self.task_config.data_size['max']),
                    deadline=random.randint(self.task_config.deadline['min'], self.task_config.deadline['max']),
                    arrival_time=self.current_time,
                    importance=random.uniform(self.task_config.importance['min'], self.task_config.importance['max'])
                )
                self.devices[device_id].submit_task(task)
                self.tasks[device_id] = task
                self.task_counter += 1
    
    def _evaluate_tasks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Составить матрицу полезностей и стоимостей относительной текущего пула задач."""
        utility_matrix = np.array([
            [evaluate_utility(self.tasks[device.id], node) if device.id in self.tasks else 0.0 for node in self.nodes]
            if device.id in self.tasks
            else [0.0 for _ in self.nodes]
            for device in self.devices
        ])
        cost_matrix = np.array([
            [evaluate_cost(self.tasks[device.id], node) if device.id in self.tasks else 0.0 for node in self.nodes]
            if device.id in self.tasks
            else [0.0 for _ in self.nodes]
            for device in self.devices
        ])
        return utility_matrix, cost_matrix
    
    def _distribute_tasks(self, allocation: np.ndarray, actions: List[int]) -> Tuple[int, int]:
        """Назначение задач согласно распределению и стратегий узлов."""
        accepted_tasks = []
        rejected_tasks = []
        for device_id, device_allocation in enumerate(allocation):
            for node_id, node_allocation in enumerate(device_allocation):
                if node_allocation == 1:
                    task: Task = self.tasks[device_id]
                    action: int = actions[node_id]
                    if action < 3: # ACCEPT
                        if action == 1: # PRIORITY_HIGH
                            task.priority = TaskPriority.HIGH
                        if action == 2: # PRIORITY_LOW
                            task.priority = TaskPriority.LOW

                        self.nodes[node_id].accept_task(task)
                        accepted_tasks.append(task)
                    else:  # REJECT
                        self.devices[device_id].task_rejected(task)
                        rejected_tasks.append(task)
        for device_id, task in self.tasks.items():
            if task not in accepted_tasks and task not in rejected_tasks:
                self.devices[device_id].task_rejected(task)
                rejected_tasks.append(task)
        return accepted_tasks, rejected_tasks

    def _execute_tasks(self) -> Tuple[int, List[int]]:
        completed = 0
        latencies = []
        for node in self.nodes:
            completed_tasks, latency = node.step(self.current_time)
            [self.devices[task.device_id].task_completed(task) for task in completed_tasks]
            completed += len(completed_tasks)
            latencies.append(latency)

        return completed, latencies
    
    def _compute_rewards(self, accepted_tasks: List[Task], vcg_payments: np.ndarray) -> np.ndarray:
        # local_rewards = np.array([self.reward_manager.compute_local_reward(node.id) for node in self.nodes]) 
        # return self.reward_manager.integrate_vcg_payments(local_rewards, vcg_payments)
        return [1.0 for _ in self.nodes] # TODO
    
    def get_observations(self) -> np.ndarray:
        """Get local observations for each node."""
        observations = []
        for idx, node in enumerate(self.nodes):
            obs = [
                node.cpu_available / node.cpu_capacity,
                node.memory_available / node.memory_capacity,
                min(len(node.task_queue) / 10.0, 1.0),
                0.0,  # Placeholder for neighbor state TODO
                idx / len(self.nodes)
            ]
            observations.append(obs)
        
        return np.array(observations, dtype=np.float32)
