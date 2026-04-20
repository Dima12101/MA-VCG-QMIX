"""
QMIX Agent - основной класс агента для обучения
"""

import torch
import torch.nn as nn
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
        self.hidden_size = 64
        self.state_size = self.num_agents * self.obs_size
        self.buffer_size = buffer_size

        self._init_networks()
        self._init_hidden_states()
        self._init_buffer()
    
    def _init_networks(self):
        # Агентские сети
        self.agent_networks = nn.ModuleList([
            GRUAgent(self.obs_size, self.hidden_size, self.action_size)
            for _ in range(self.num_agents)
        ])
        
        # Целевые сети
        self.target_agent_networks = nn.ModuleList([
            GRUAgent(self.obs_size, self.hidden_size, self.action_size)
            for _ in range(self.num_agents)
        ])
        
        # Копировать начальные веса
        for agent, target in zip(self.agent_networks, self.target_agent_networks):
            target.load_state_dict(agent.state_dict())
        
        # Mixing network
        self.mixing_network = MixingNetwork(self.num_agents, self.state_size, self.hidden_size)
        self.target_mixing_network = MixingNetwork(self.num_agents, self.state_size, self.hidden_size)
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())

    def _init_hidden_states(self):
        # Скрытые состояния GRU
        self.hidden_states = [torch.zeros(1, self.hidden_size) for _ in range(self.num_agents)]

    def _init_buffer(self):
        # Буфер опыта
        self.buffer = ExperienceBuffer(self.buffer_size)

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
        self.hidden_states = [torch.zeros(1, self.hidden_size) for _ in range(self.num_agents)]
