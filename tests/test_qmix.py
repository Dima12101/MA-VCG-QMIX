"""
Тесты для QMIX обучения
"""

import unittest
import torch
from src.agents.networks import GRUAgent, MixingNetwork

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
        mixing_net = MixingNetwork(num_agents, action_size=4, hidden_size=64)
        
        # Локальные Q-значения [batch=2, num_agents=3, actions=4]
        q_values = torch.randn(2, 3, 4)
        # Состояние [batch=2, state_size=10]
        state = torch.randn(2, 10)
        
        global_q = mixing_net(q_values, state)
        
        self.assertEqual(global_q.shape, (2, 4))  # [batch, actions]

if __name__ == '__main__':
    unittest.main()
