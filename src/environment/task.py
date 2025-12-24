import dataclasses
from enum import Enum
from typing import Optional

class TaskPriority(Enum):
    LOW = 0.5
    MEDIUM = 1.0
    HIGH = 2.0

@dataclasses.dataclass
class Task:
    """Класс для представления задачи"""
    id: int
    device_id: int  # Какое устройство отправило задачу
    cpu_required: int  # Требуемые CPU циклы
    memory_required: int  # Требуемая память (MB)
    priority: TaskPriority = TaskPriority.MEDIUM
    arrival_time: int = 0  # Время прихода в систему
    deadline: int = 5000  # Дедлайн выполнения (мс)
    importance: float = 1.0  # Важность для устройства (0..1)
    
    @property
    def value(self) -> float:
        """Ценность задачи для устройства"""
        return self.priority.value * self.importance
    
    @property
    def is_expired(self, current_time: int) -> bool:
        """Истёк ли дедлайн?"""
        return current_time > self.arrival_time + self.deadline
    
    def get_processing_time(self, cpu_capacity: int) -> float:
        """Оценка времени обработки на узле с заданной CPU"""
        return self.cpu_required / cpu_capacity  # в условных единицах времени
