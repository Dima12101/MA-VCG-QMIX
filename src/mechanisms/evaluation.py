import math
from ..environment.task import Task
from ..environment.edge_node import EdgeNode

def evaluate_utility(
    task: Task,
    edge_node: EdgeNode,
    network_latency_ms: float = 0.0,
):
    """Оценка выгоды или удовлетворенности устройства от выполнения его задачи на узле"""

    # Базовая ценность задачи определяет, сколько пользы принесет устройству успешное выполнение задачи
    base = task.priority.value * (task.cpu_required + task.memory_required) * task.importance

    # Функция штрафа за задержку определяет, насколько эта польза уменьшается из-за задержки
    estimated_time_ms = task.processing_time_ms(edge_node.cpu_capacity) + network_latency_ms # TODO
    time_penalty = math.exp(
        -edge_node.delay_sensitivity * estimated_time_ms / max(task.deadline, 1.0)
    )

    # Энергозатраты на выполнение задачи
    energy_cost = task.data_size * network_latency_ms * edge_node.transmission_energy_coeff

    return max(base * time_penalty - energy_cost, 0.0)

def evaluate_cost(
    task: Task,
    edge_node: EdgeNode,
    network_latency_ms: float = 0.0,
    direct_connection: bool = True,
):
    """Оценка затрат, которые несет узел при выполнении задачи от устройства"""

    # Вычислительная стоимость отражает реальные издержки на вычисления
    computation_cost = (
        task.cpu_required * edge_node.base_price["CPU"]
        + task.memory_required * edge_node.base_price["MEM"]
    )
    
    # Коммуникационная стоимость моделирует цену передачи данных
    path_penalty = 1.0 if direct_connection else edge_node.indirect_link_penalty
    communication_cost = (
        network_latency_ms * task.data_size * edge_node.base_price["NET"] * path_penalty
    )

    # Стоимость перегрузки позволяет учесть снижение производительности при высокой загрузке
    overload_cost = (
           ((edge_node.cpu_used + task.cpu_required) / edge_node.cpu_capacity) ** 2
           + ((edge_node.memory_used + task.memory_required) / edge_node.memory_capacity) ** 2
    )

    return computation_cost + communication_cost + overload_cost
