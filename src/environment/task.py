import dataclasses
import math
from enum import Enum

class TaskPriority(Enum):
    LOW = 0.5
    MEDIUM = 0.75
    HIGH = 1.0

@dataclasses.dataclass
class Task:
    """Класс для представления задачи"""
    id: int                                         # ID задачи
    device_id: int                                  # Какое устройство выставило задачу
    arrival_time: int = 0                           # Время прихода в систему

    cpu_required: int                               # Требуемые CPU ресурсы
    memory_required: int                            # Требуемая память (MB)

    data_size: int                                  # Объем передаваемых данных (MB)
    priority: TaskPriority = TaskPriority.MEDIUM    # Приоритет задачи (по умолчанию MEDIUM)
    deadline: int = 5000                            # Крайний срок выполнения (мс)
    importance: float = 1.0                         # Важность для устройства (0..1)


    def is_expired(self, current_time: int, step_duration_ms: int = 1) -> bool:
        """Истёк ли дедлайн?"""
        elapsed_ms = max(0, current_time - self.arrival_time) * step_duration_ms
        return elapsed_ms > self.deadline
    
    def processing_time_ms(self, cpu_capacity: int) -> float:
        """Оценка времени обработки на узле в миллисекундах."""
        if cpu_capacity <= 0:
            return float("inf")
        return max(1.0, 1000.0 * self.cpu_required / cpu_capacity)

    def processing_steps(self, cpu_capacity: int, step_duration_ms: int) -> int:
        """Оценка времени обработки в шагах симуляции."""
        safe_step = max(step_duration_ms, 1)
        return max(1, math.ceil(self.processing_time_ms(cpu_capacity) / safe_step))
