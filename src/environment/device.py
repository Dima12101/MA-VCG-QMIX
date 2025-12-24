from typing import List, Optional
from .task import Task

class Device:
    """Класс для представления мобильного устройства"""
    
    def __init__(self, device_id: int, importance: float = 1.0):
        self.device_id = device_id
        self.importance = importance  # От 0 до 1 (влияет на polarity)
        self.submitted_tasks: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.rejected_tasks: List[Task] = []
        self.total_payment: float = 0.0
        self.received_payment: float = 0.0
    
    def submit_task(self, task: Task):
        """Отправить задачу в систему"""
        self.submitted_tasks.append(task)
    
    def task_completed(self, task: Task):
        """Задача успешно выполнена"""
        self.completed_tasks.append(task)
    
    def task_rejected(self, task: Task):
        """Задача отклонена"""
        self.rejected_tasks.append(task)
    
    def receive_payment(self, amount: float):
        """Получить платёж"""
        self.total_payment += amount
        self.received_payment = amount
    
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
