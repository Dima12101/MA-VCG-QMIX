import dataclasses
from enum import Enum

class TaskPriority(Enum):
    LOW = 0.5
    MEDIUM = 1.0
    HIGH = 2.0

@dataclasses.dataclass
class Task:
    """Класс для представления задачи"""
    id: int                                         # ID задачи
    device_id: int                                  # Какое устройство выставило задачу
    cpu_required: int                               # Требуемые CPU ресурсы
    memory_required: int                            # Требуемая память (MB)
    bandwidth_required: int                         # Требуемая пропускная способность (Mbps)
    data_size: int                                  # Объем данных (MB)
    priority: TaskPriority = TaskPriority.MEDIUM    # Приоритет задачи (по умолчанию MEDIUM)
    deadline: int = 5000                            # Дедлайн выполнения (мс)
    importance: float = 1.0                         # Важность для устройства (0..1)
    arrival_time: int = 0                           # Время прихода в систему

    def is_expired(self, current_time: int) -> bool:
        """Истёк ли дедлайн?"""
        return current_time > self.arrival_time + self.deadline
    
    def processing_time(self, cpu_capacity: int) -> int:
        """Оценка времени обработки на узле с заданной CPU"""
        return self.cpu_required / cpu_capacity  # в условных единицах времени
