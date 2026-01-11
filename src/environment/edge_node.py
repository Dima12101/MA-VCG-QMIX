
from typing import List, Tuple, Dict
from .task import Task
from ..config import NodeConfig

class EdgeNode:
    """Класс для представления edge-узла"""
    
    def __init__(self, id: int, config: NodeConfig):
        self.id = id                                                # ID узла
        self.cpu_capacity = config.cpu_capacity                     # Доступные ресурсы CPU узла
        self.memory_capacity = config.memory_capacity               # Доступные ресурсы MEM узла
        self.bandwidth = config.bandwidth                           # Пропускная способность узла

        self.base_price = {"CPU": 0.2, "MEM": 0.05}                 # Стоимость использования ресурсов $/unit
        
        self.cpu_used = 0                                           # Задействованные ресурсы CPU узла
        self.memory_used = 0                                        # Задействованные ресурсы MEM узла
        
        self.task_queue: Dict[int, Task] = {}                       # Задачи в очереди на выполнение
        self.task_executed: Dict[int, Task] = {}                    # Выполняемые задачи
        self.task_executed_time: Dict[int, int] = {}                # Оставшиеся время выполняемых задач (task_id -> remaining_time)
    
    def reset(self):
        self.cpu_used = 0 
        self.memory_used = 0
        self.task_queue: Dict[int, Task] = {} 
        self.task_executed: Dict[int, Task] = {}
        self.task_executed_time: Dict[int, int] = {}
    
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
    
    def accept_task(self, task: Task):
        """Назначить задачу на узел"""
        self.task_queue[task.id] = task

    def _can_allocate_task(self, task: Task) -> bool:
        """Может ли узел начать выполнение задачи?"""
        return (self.cpu_available >= task.cpu_required and
                self.memory_available >= task.memory_required)
    
    def _allocate_task(self, task: Task):
        """Начать выполнение задачи и выделить ей ресурсы"""
        self.cpu_used += task.cpu_required
        self.memory_used += task.memory_required

        del self.task_queue[task.id]
        self.task_executed[task.id] = task
        self.task_executed_time[task.id] = task.processing_time(self.cpu_capacity)

    def _completed_task(self, task: Task):
        """Завершить выполнение задачи"""
        self.cpu_used -= task.cpu_required
        self.memory_used -= task.memory_required
        del self.task_executed[task.id]
        del self.task_executed_time[task.id]
    
    def step(self, current_time: int) -> Tuple[List[Task], float]:
        """Выполнить один шаг моделирования"""

        # 1. Обновить время выполнения текущих задач
        completed_tasks = []
        total_latency = 0.0
        for task_id in list(self.task_executed.keys()):
            self.task_executed_time[task_id] -= 1
            if self.task_executed_time[task_id] <= 0:
                completed_task = self.task_executed[task_id]
                total_latency += completed_task.deadline - (current_time - completed_task.arrival_time)
                completed_tasks.append(completed_task)
                self._completed_task(completed_task)
        
        # 2. Взять на выполнение новые задачи
        for task_id in list(self.task_queue.keys()):
            task = self.task_queue[task_id]
            if (self._can_allocate_task(task)):
                self._allocate_task(task)
        
        return completed_tasks, total_latency
