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
        deadline_violation_rate: float,
        drop_rate: float,
        load_imbalance: float,
        fairness_index: float,
        gini_coefficient: float,
        backlog_pressure: float = 0.0,
    ) -> float:
        """
        Вычислить глобальное вознаграждение для всей системы
        
        Args:
            social_welfare: социальное благосустояние
            deadline_violation_rate: доля задач с риском нарушения дедлайна
            drop_rate: доля отклонённых задач
            load_imbalance: дисбаланс загрузки узлов
            fairness_index: индекс справедливости
            gini_coefficient: коэффициент Джини
            backlog_pressure: усреднённое давление backlog на узлах
        
        Returns:
            reward: глобальное вознаграждение
        """
        sw_component = social_welfare / max(self.num_devices, 1)
        fairness_penalty = max(0.0, self.fairness_target - fairness_index)
        gini_penalty = max(0.0, gini_coefficient - self.gini_target)
        backlog_penalty = max(0.0, backlog_pressure)
        raw_reward = (
            sw_component
            - 0.9 * deadline_violation_rate
            - 0.8 * drop_rate
            - 0.6 * load_imbalance
            - 0.5 * backlog_penalty
            - fairness_penalty
            - gini_penalty
        )
        return float(np.tanh(raw_reward))

    def compute_auction_reward(
        self,
        completed_payments: np.ndarray,
        completed_welfare: np.ndarray,
    ) -> float:
        """Собрать аукционный компонент вознаграждения по завершённым задачам."""
        payment_signal = float(np.sum(completed_payments)) if completed_payments.size else 0.0
        welfare_signal = float(np.sum(completed_welfare)) if completed_welfare.size else 0.0
        raw_reward = (
            welfare_signal - 0.15 * payment_signal
        ) / max(self.num_devices, 1)
        return float(np.tanh(raw_reward))

    def combine_rewards(
        self,
        social_welfare: float,
        fairness_index: float,
        gini_coefficient: float,
        deadline_violation_rate: float,
        drop_rate: float,
        load_imbalance: float,
        completed_payments: np.ndarray,
        completed_welfare: np.ndarray,
        backlog_pressure: float = 0.0,
        stress_context: float = 0.0,
    ) -> np.ndarray:
        """Собрать общий командный reward с аукционной компонентой."""
        global_reward = self.compute_global_reward(
            social_welfare=social_welfare,
            deadline_violation_rate=deadline_violation_rate,
            drop_rate=drop_rate,
            load_imbalance=load_imbalance,
            fairness_index=fairness_index,
            gini_coefficient=gini_coefficient,
            backlog_pressure=backlog_pressure,
        )
        auction_reward = self.compute_auction_reward(
            completed_payments=completed_payments,
            completed_welfare=completed_welfare,
        )
        lambda_vcg = float(
            np.clip(
                self.vcg_weight * np.clip(stress_context, 0.0, 1.0),
                0.0,
                1.0,
            )
        )
        global_mix = float(np.clip(self.global_weight, 0.0, 1.0))
        hybrid_reward = (1.0 - lambda_vcg) * global_reward + lambda_vcg * auction_reward
        shared_reward = (1.0 - global_mix) * hybrid_reward + global_mix * global_reward
        return np.full(self.num_agents, shared_reward, dtype=float)
