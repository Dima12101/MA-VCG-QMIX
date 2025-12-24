import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from .task import Task, TaskPriority
from .device import Device
from ..config import ENV_CONFIG, TASK_CONFIG, EDGE_CONFIG

class EdgeNode:
    """Класс для представления edge-узла"""
    
    def __init__(self, node_id: int, config: EDGE_CONFIG):
        self.node_id = node_id
        self.cpu_capacity = config.cpu_capacity
        self.memory_capacity = config.memory_capacity
        self.bandwidth = config.bandwidth
        
        self.cpu_used = 0
        self.memory_used = 0
        self.task_queue: List[Task] = []
        self.executing_tasks: Dict[int, float] = {}  # task_id -> remaining_time
    
    @property
    def cpu_available(self) -> int:
        return self.cpu_capacity - self.cpu_used
    
    @property
    def memory_available(self) -> int:
        return self.memory_capacity - self.memory_used
    
    @property
    def load(self) -> float:
        """Нормализованная нагрузка узла (0..1)"""
        return (self.cpu_used / self.cpu_capacity + 
                self.memory_used / self.memory_capacity) / 2
    
    def can_accept_task(self, task: Task) -> bool:
        """Может ли узел принять задачу?"""
        return (self.cpu_available >= task.cpu_required and
                self.memory_available >= task.memory_required)
    
    def allocate_task(self, task: Task):
        """Выделить ресурсы для задачи"""
        self.cpu_used += task.cpu_required
        self.memory_used += task.memory_required
        # Вычислить время обработки
        processing_time = task.get_processing_time(self.cpu_capacity)
        self.executing_tasks[task.id] = processing_time
    
    def step(self) -> Tuple[List[Task], float]:
        """Выполнить один шаг моделирования"""
        completed_tasks = []
        total_latency = 0.0
        
        # Обновить время выполнения задач
        for task_id in list(self.executing_tasks.keys()):
            self.executing_tasks[task_id] -= 1
            if self.executing_tasks[task_id] <= 0:
                # Задача завершена
                completed_tasks.append(task_id)
                del self.executing_tasks[task_id]
                total_latency += 0.0  # TODO: добавить расчёт latency
        
        # Освободить ресурсы
        for task_id in completed_tasks:
            # Найти задачу и освободить ресурсы
            # (требуется дополнительная логика)
            pass
        
        return completed_tasks, total_latency

class EdgeNetwork:
    """Класс для представления edge-сети"""
    
    def __init__(self, config: ENV_CONFIG = None):
        self.config = config or ENV_CONFIG
        self.edges: List[EdgeNode] = []
        self.devices: List[Device] = []
        self.current_time = 0
        self.task_counter = 0
        
        # Инициализировать узлы и устройства
        self._initialize_network()
        
        # История для анализа
        self.history = {
            'time': [],
            'accepted_tasks': [],
            'rejected_tasks': [],
            'avg_latency': [],
            'social_welfare': [],
            'load_per_node': [[] for _ in range(self.config.num_edges)],
        }
    
    def _initialize_network(self):
        """Инициализировать сеть"""
        # Создать edge-узлы
        for i in range(self.config.num_edges):
            self.edges.append(EdgeNode(i, EDGE_CONFIG))
        
        # Создать устройства с распределением важности
        importance_dist = np.random.beta(2, 5, self.config.num_devices)
        for i in range(self.config.num_devices):
            self.devices.append(Device(i, importance=float(importance_dist[i])))
    
    def generate_tasks(self):
        """Сгенерировать новые задачи"""
        # Пуассоновский процесс прихода задач
        num_new_tasks = np.random.poisson(self.config.arrival_rate)
        
        for _ in range(num_new_tasks):
            device_id = random.randint(0, self.config.num_devices - 1)
            cpu = random.randint(TASK_CONFIG.cpu_min, TASK_CONFIG.cpu_max)
            memory = random.randint(TASK_CONFIG.memory_min, TASK_CONFIG.memory_max)
            priority = random.choice(list(TaskPriority))
            importance = random.uniform(0.5, 1.0)
            
            task = Task(
                id=self.task_counter,
                device_id=device_id,
                cpu_required=cpu,
                memory_required=memory,
                priority=priority,
                arrival_time=self.current_time,
                importance=importance
            )
            
            self.devices[device_id].submit_task(task)
            self.task_counter += 1
    
    def step(self) -> Dict:
        """Выполнить один шаг моделирования"""
        self.current_time += 1
        
        # Сгенерировать новые задачи
        self.generate_tasks()
        
        # Выполнить шаг на каждом узле
        metrics = {
            'accepted': 0,
            'rejected': 0,
            'completed': 0,
            'avg_latency': 0,
        }
        
        for edge in self.edges:
            completed, latency = edge.step()
            metrics['completed'] += len(completed)
            metrics['avg_latency'] += latency
        
        # Записать в историю
        self.history['time'].append(self.current_time)
        self.history['accepted_tasks'].append(metrics['accepted'])
        self.history['rejected_tasks'].append(metrics['rejected'])
        self.history['avg_latency'].append(metrics['avg_latency'])
        
        # Нагрузка на узлы
        for i, edge in enumerate(self.edges):
            self.history['load_per_node'][i].append(edge.load)
        
        return metrics
    
    def get_state(self) -> Dict:
        """Получить текущее состояние сети"""
        state = {
            'time': self.current_time,
            'pending_tasks': sum(len(edge.task_queue) for edge in self.edges),
            'node_loads': [edge.load for edge in self.edges],
            'available_resources': [
                (edge.cpu_available, edge.memory_available) 
                for edge in self.edges
            ],
        }
        return state
