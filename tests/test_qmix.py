"""
Тесты для QMIX обучения
"""

import unittest
import numpy as np
import torch
from src.config import NetworkConfig, TrainingConfig
from src.agents.networks import GRUAgent, MixingNetwork
from src.learning.trainer import QMIXTrainer

class TestQMIXNetworks(unittest.TestCase):
    """Тесты архитектуры QMIX"""
    
    def test_gru_agent_forward(self):
        """Тест прямого прохода GRU агента"""
        agent = GRUAgent(obs_size=8, hidden_size=64, action_size=4)
        
        # Создать тестовый вход [batch_size=2, seq_len=1, obs_size=8]
        obs = torch.randn(2, 1, 8)
        
        q_values, hidden = agent(obs)
        
        self.assertEqual(q_values.shape, (2, 4))  # [batch, actions]
        self.assertEqual(hidden.shape, (2, 64))   # [batch, hidden]
    
    def test_mixing_network_forward(self):
        """Тест прямого прохода Mixing Network"""
        num_agents = 3
        state_size = 10
        mixing_net = MixingNetwork(num_agents, state_size=state_size, hidden_size=64)
        
        # Локальные Q-значения уже выбраны по joint action [batch=2, num_agents=3]
        q_values = torch.randn(2, 3)
        # Состояние [batch=2, state_size=10]
        state = torch.randn(2, state_size)
        
        global_q = mixing_net(q_values, state)
        
        self.assertEqual(global_q.shape, (2,))  # [batch]

    def test_action_mask_is_respected(self):
        """QMIX не должен выбирать действия вне бинарной маски."""
        trainer = QMIXTrainer(
            num_agents=2,
            network_config=NetworkConfig(obs_size=13, action_size=4, state_size=26),
            training_config=TrainingConfig(epsilon_start=1.0, epsilon_end=1.0, epsilon_decay=1.0),
        )
        observations = np.array(
            [
                [0.9, 0.9, 0.0, 0.1, 0.0, 0.2, 0.3, 0.4, 0.5, 0.0, 1.0, 0.0, 0.0],
                [0.8, 0.7, 0.0, 0.2, 0.0, 0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        actions = trainer.select_actions(observations)

        np.testing.assert_array_equal(actions, np.array([1, 3]))

if __name__ == '__main__':
    unittest.main()
