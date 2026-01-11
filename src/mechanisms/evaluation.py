import math
from ..environment.task import Task
from ..environment.edge_node import EdgeNode

def evaluate_utility(task: Task, edge_node: EdgeNode):
    """Оценка выгоды или удовлетворенности устройства от выполнения его задачи на узле"""

    # Базовая ценность задачи определяет, сколько пользы принесет устройству успешное выполнение задачи
    base = task.priority.value * (task.cpu_required + task.memory_required) * task.importance

    # Функция штрафа за задержку определяет, насколько эта польза уменьшается из-за задержки
    time_penalty = math.exp(- 1.0 * task.processing_time(edge_node.cpu_capacity) / task.deadline)

    # Энергозатраты на выполнение задачи
    energy_cost = task.data_size * 0.0 # TODO

    return base * time_penalty - energy_cost

def evaluate_cost(task: Task, edge_node: EdgeNode):
    """Оценка затрат, которые несет узел при выполнении задачи от устройства"""

    # Вычислительная стоимость отражает реальные издержки на вычисления
    computation_cost = (task.cpu_required * edge_node.base_price["CPU"] 
                        + task.memory_required * edge_node.base_price["MEM"])
    
    # Коммуникационная стоимость моделирует цену передачи данных
    communication_cost = 0.0 # TODO требуется граф

    # Стоимость перегрузки позволяет учесть снижение производительности при высокой загрузке
    overload_cost = computation_cost * (1 + (edge_node.cpu_available / edge_node.cpu_capacity) ** 2)

    return computation_cost + communication_cost + overload_cost
