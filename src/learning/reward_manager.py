"""
Управление вознаграждениями с интеграцией VCG платежей
"""

import numpy as np

class RewardManager:
    """Класс для управления вознаграждениями агентов"""
    
    def __init__(
        self,
        num_agents: int,
        num_devices: int,
        vcg_weight: float,
        global_weight: float = 0.1,
        fairness_target: float = 0.85,
        gini_target: float = 0.3,
    ):
        self.num_agents = num_agents
        self.num_devices = num_devices
        self.vcg_weight = vcg_weight
        self.global_weight = global_weight
        self.fairness_target = fairness_target
        self.gini_target = gini_target
    
    def compute_local_reward(
        self,
        local_value: float,
        operational_cost: float,
        sla_penalty: float = 0.0,
    ) -> float:
        """
        Вычислить локальное вознаграждение для узла.
        
        Args:
            local_value: ценность выполненных или принятых узлом задач
            operational_cost: затраты узла на обработку
            sla_penalty: штраф за риск нарушения SLA
        
        Returns:
            reward: локальное вознаграждение
        """
        return local_value - operational_cost - sla_penalty
    
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
        fairness_penalty = max(0.0, self.fairness_target - fairness_index) * 10
        
        # Штраф за неравное распределение платежей
        gini_penalty = max(0.0, gini_coefficient - self.gini_target) * 5
        
        return sw_component - fairness_penalty - gini_penalty
    
    def integrate_vcg_payments(
        self,
        local_rewards: np.ndarray,
        node_revenues: np.ndarray
    ) -> np.ndarray:
        """
        Интегрировать VCG платежи в локальные вознаграждения
        
        Args:
            local_rewards: локальные вознаграждения [num_agents]
            node_revenues: агрегированные доходы узлов [num_agents]
        
        Returns:
            integrated_rewards: интегрированные вознаграждения
        """
        # Усреднить платежи по всем устройствам
        positive_revenue = node_revenues[node_revenues > 0]
        if positive_revenue.size == 0:
            return local_rewards.copy()
        avg_payment = np.mean(positive_revenue)
        
        # Добавить VCG компоненту к локальным вознаграждениям
        integrated = local_rewards + self.vcg_weight * (node_revenues / (avg_payment + 1e-8))
        
        return integrated

    def combine_rewards(
        self,
        local_rewards: np.ndarray,
        node_revenues: np.ndarray,
        social_welfare: float,
        fairness_index: float,
        gini_coefficient: float,
    ) -> np.ndarray:
        """Собрать гибридное вознаграждение из RL и аукционного сигналов."""
        integrated = self.integrate_vcg_payments(local_rewards, node_revenues)
        global_reward = self.compute_global_reward(
            social_welfare=social_welfare,
            fairness_index=fairness_index,
            gini_coefficient=gini_coefficient,
        )
        global_signal = np.tanh(global_reward / (abs(social_welfare) + 1.0))
        return integrated + self.global_weight * global_signal
