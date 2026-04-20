from typing import Dict, List, Optional, Tuple
from .task import Task
from ..config import NodeConfig

class EdgeNode:
    """Класс для представления edge-узла"""
    
    def __init__(self, id: int, config: NodeConfig):
        self.id = id                                                # ID узла
        self.cpu_capacity = config.cpu_capacity                     # Доступные ресурсы CPU узла
        self.memory_capacity = config.memory_capacity               # Доступные ресурсы MEM узла
        self.bandwidth = config.bandwidth                           # Пропускная способность узла

        self.base_price = {
            "CPU": config.cpu_unit_price,
            "MEM": config.memory_unit_price,
            "NET": config.network_unit_price,
        }
        self.delay_sensitivity = config.delay_sensitivity
        self.transmission_energy_coeff = config.transmission_energy_coeff
        self.indirect_link_penalty = config.indirect_link_penalty

        self.cpu_used = 0
        self.memory_used = 0
        self.task_queue: Dict[int, Task] = {}
        self.task_executed: Dict[int, Task] = {}
        self.task_executed_time: Dict[int, int] = {}
        self.is_failed = False
        self.failed_until_step: Optional[int] = None
    
    def reset(self):
        self.cpu_used = 0 
        self.memory_used = 0
        self.task_queue = {}
        self.task_executed = {}
        self.task_executed_time = {}
        self.is_failed = False
        self.failed_until_step = None
    
    @property
    def cpu_available(self) -> int:
        if self.is_failed:
            return 0
        return self.cpu_capacity - self.cpu_used
    
    @property
    def memory_available(self) -> int:
        if self.is_failed:
            return 0
        return self.memory_capacity - self.memory_used
    
    @property
    def load(self) -> float:
        """Нормализованная нагрузка узла (0..1)"""
        if self.is_failed:
            return 1.0
        return (self.cpu_used / self.cpu_capacity + 
                self.memory_used / self.memory_capacity) / 2

    def update_failure_state(self, current_step: int):
        """Проверить, не пора ли восстановить узел после отказа."""
        if self.is_failed and self.failed_until_step is not None and current_step >= self.failed_until_step:
            self.is_failed = False
            self.failed_until_step = None

    def fail(self, recovery_step: int) -> List[Task]:
        """Перевести узел в состояние отказа и вернуть потерянные задачи."""
        dropped_tasks = list(self.task_queue.values()) + list(self.task_executed.values())
        self.cpu_used = 0
        self.memory_used = 0
        self.task_queue = {}
        self.task_executed = {}
        self.task_executed_time = {}
        self.is_failed = True
        self.failed_until_step = recovery_step
        return dropped_tasks
    
    def accept_task(self, task: Task):
        """Назначить задачу на узел"""
        if self.is_failed:
            return
        self.task_queue[task.id] = task

    def _can_allocate_task(self, task: Task) -> bool:
        """Может ли узел начать выполнение задачи?"""
        return (not self.is_failed and
                self.cpu_available >= task.cpu_required and
                self.memory_available >= task.memory_required)
    
    def _allocate_task(self, task: Task, step_duration_ms: int):
        """Начать выполнение задачи и выделить ей ресурсы"""
        self.cpu_used += task.cpu_required
        self.memory_used += task.memory_required

        del self.task_queue[task.id]
        self.task_executed[task.id] = task
        self.task_executed_time[task.id] = task.processing_steps(self.cpu_capacity, step_duration_ms)

    def _completed_task(self, task: Task):
        """Завершить выполнение задачи"""
        self.cpu_used -= task.cpu_required
        self.memory_used -= task.memory_required
        del self.task_executed[task.id]
        del self.task_executed_time[task.id]
    
    def step(self, current_time: int, step_duration_ms: int) -> Tuple[List[Task], List[float]]:
        """Выполнить один шаг моделирования"""
        self.update_failure_state(current_time)
        if self.is_failed:
            return [], []

        # 1. Обновить время выполнения текущих задач
        completed_tasks = []
        latencies_ms = []
        for task_id in list(self.task_executed.keys()):
            self.task_executed_time[task_id] -= 1
            if self.task_executed_time[task_id] <= 0:
                completed_task = self.task_executed[task_id]
                latency_ms = max(0.0, (current_time - completed_task.arrival_time + 1) * step_duration_ms)
                latencies_ms.append(latency_ms)
                completed_tasks.append(completed_task)
                self._completed_task(completed_task)
        
        # 2. Взять на выполнение новые задачи
        sorted_queue = sorted(
            self.task_queue.values(),
            key=lambda task: (-task.priority.value, task.deadline, task.id),
        )
        for task in sorted_queue:
            if (self._can_allocate_task(task)):
                self._allocate_task(task, step_duration_ms)
        
        return completed_tasks, latencies_ms
