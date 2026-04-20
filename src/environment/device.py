from typing import List
from .task import Task

class Device:
    """Класс для представления мобильного устройства"""
    
    def __init__(self, id: int, importance: float = 1.0):
        self.id = id                            # ID устройства
        self.importance = importance            # Важность устройства. От 0 до 1 (влияет на polarity)
        self.submitted_tasks: List[Task] = []   # Выставленные задачи
        self.completed_tasks: List[Task] = []   # Выполненные задачи
        self.rejected_tasks: List[Task] = []    # Отклоненные задачи
        self.total_payment: float = 0.0         # Общая сумма платежа

    def reset(self):
        self.submitted_tasks: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.rejected_tasks: List[Task] = []
        self.total_payment: float = 0.0
    
    def submit_task(self, task: Task):
        """Отправить задачу в систему"""
        self.submitted_tasks.append(task)
    
    def task_completed(self, task: Task):
        """Задача успешно выполнена"""
        self.completed_tasks.append(task)
    
    def task_rejected(self, task: Task):
        """Задача отклонена"""
        self.rejected_tasks.append(task)
    
    def record_payment(self, amount: float):
        """Учесть платёж устройства за успешно размещённую задачу."""
        self.total_payment += amount

    def receive_payment(self, amount: float):
        """Обратная совместимость со старым интерфейсом."""
        self.record_payment(amount)
    
    @property
    def success_rate(self) -> float:
        """Процент успешно завершённых задач"""
        if len(self.submitted_tasks) == 0:
            return 0.0
        return len(self.completed_tasks) / len(self.submitted_tasks)
    
    @property
    def avg_payment(self) -> float:
        """Средний платёж за задачу"""
        if len(self.completed_tasks) == 0:
            return 0.0
        return self.total_payment / len(self.completed_tasks)
