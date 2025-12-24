"""
Управление вознаграждениями с интеграцией VCG платежей
"""

import numpy as np
from typing import Dict, List

class RewardManager:
    """Класс для управления вознаграждениями агентов"""
    
    def __init__(self, num_agents: int, num_devices: int):
        self.num_agents = num_agents
        self.num_devices = num_devices
        self.vcg_weight = 0.5  # Вес VCG платежей в функции вознаграждения
    
    def compute_local_reward(
        self,
        agent_id: int,
        task_accepted: bool,
        task_value: float,
        processing_time: float,
        energy_used: float
    ) -> float:
        """
        Вычислить локальное вознаграждение для агента
        
        Args:
            agent_id: идентификатор агента
            task_accepted: была ли задача принята
            task_value: ценность задачи
            processing_time: время обработки
            energy_used: потребленная энергия
        
        Returns:
            reward: локальное вознаграждение
        """
        if not task_accepted:
            return -0.5  # Штраф за отклонение
        
        # Положительное вознаграждение за принятие ценной задачи
        value_reward = task_value
        
        # Штраф за время обработки
        time_penalty = 0.1 * processing_time
        
        # Штраф за энергию
        energy_penalty = 0.05 * energy_used
        
        return value_reward - time_penalty - energy_penalty
    
    def compute_global_reward(
        self,
        social_welfare: float,
        fairness_index: float,
        gini_coefficient: float
    ) -> float:
        """
        Вычислить глобальное вознаграждение для всей системы
        
        Args:
            social_welfare: социальное благосустояние
            fairness_index: индекс справедливости
            gini_coefficient: коэффициент Джини
        
        Returns:
            reward: глобальное вознаграждение
        """
        # SW — основной компонент
        sw_component = social_welfare
        
        # Штраф за несправедливость
        fairness_penalty = 0.0 if fairness_index > 0.85 else (0.85 - fairness_index) * 10
        
        # Штраф за неравное распределение платежей
        gini_penalty = 0.0 if gini_coefficient < 0.3 else (gini_coefficient - 0.3) * 5
        
        return sw_component - fairness_penalty - gini_penalty
    
    def integrate_vcg_payments(
        self,
        local_rewards: np.ndarray,
        vcg_payments: np.ndarray
    ) -> np.ndarray:
        """
        Интегрировать VCG платежи в локальные вознаграждения
        
        Args:
            local_rewards: локальные вознаграждения [num_agents]
            vcg_payments: VCG платежи [num_devices]
        
        Returns:
            integrated_rewards: интегрированные вознаграждения
        """
        # Усреднить платежи по всем устройствам
        avg_payment = np.mean(vcg_payments[vcg_payments > 0])
        
        # Добавить VCG компоненту к локальным вознаграждениям
        # (с меньшим весом, так как QMIX уже учитывает глобальное благосустояние)
        integrated = local_rewards + self.vcg_weight * (vcg_payments[:len(local_rewards)] / (avg_payment + 1e-8))
        
        return integrated
