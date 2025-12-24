"""
QMIX Agent - основной класс агента для обучения
"""

import torch
import numpy as np
from typing import Tuple
from .networks import GRUAgent, MixingNetwork
from .experience_buffer import ExperienceBuffer

class QMIXAgent:
    """Централизованный агент QMIX"""
    
    def __init__(
        self,
        num_agents: int,
        obs_size: int,
        action_size: int,
        buffer_size: int = 10000
    ):
        self.num_agents = num_agents
        self.obs_size = obs_size
        self.action_size = action_size
        
        # Агентские сети
        self.agent_networks = [
            GRUAgent(obs_size, 64, action_size)
            for _ in range(num_agents)
        ]
        
        # Целевые сети
        self.target_agent_networks = [
            GRUAgent(obs_size, 64, action_size)
            for _ in range(num_agents)
        ]
        
        # Копировать начальные веса
        for agent, target in zip(self.agent_networks, self.target_agent_networks):
            target.load_state_dict(agent.state_dict())
        
        # Mixing network
        self.mixing_network = MixingNetwork(num_agents, action_size, 64)
        self.target_mixing_network = MixingNetwork(num_agents, action_size, 64)
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
        # Буфер опыта
        self.buffer = ExperienceBuffer(buffer_size)
        
        # Скрытые состояния GRU
        self.hidden_states = [torch.zeros(1, 64) for _ in range(num_agents)]
    
    def store_experience(self, experience_dict):
        """Сохранить опыт в буфер"""
        self.buffer.add(
            experience_dict['state'],
            experience_dict['actions'],
            experience_dict['rewards'],
            experience_dict['next_state'],
            experience_dict['done']
        )
    
    def get_hidden_states(self):
        """Получить скрытые состояния GRU"""
        return self.hidden_states
    
    def reset_hidden_states(self):
        """Сбросить скрытые состояния"""
        self.hidden_states = [torch.zeros(1, 64) for _ in range(self.num_agents)]
