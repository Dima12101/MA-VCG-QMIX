import torch
import torch.optim as optim
import numpy as np
from ..agents.networks import GRUAgent, MixingNetwork
from ..agents.experience_buffer import ExperienceBuffer
from ..mechanisms.payments import calculate_vcg_payments
from .metrics import calculate_td_error
from ..config import QMIX_CONFIG, VCG_CONFIG

class QMIXTrainer:
    """Тренер для обучения QMIX агентов"""
    
    def __init__(self, num_agents: int, obs_size: int, action_size: int):
        self.num_agents = num_agents
        self.obs_size = obs_size
        self.action_size = action_size
        
        # Создать агентские сети
        self.agent_networks = [
            GRUAgent(obs_size, 64, action_size)
            for _ in range(num_agents)
        ]
        
        # Целевые сети (для стабилизации)
        self.target_networks = [
            GRUAgent(obs_size, 64, action_size)
            for _ in range(num_agents)
        ]
        
        # Скопировать веса
        for agent, target in zip(self.agent_networks, self.target_networks):
            target.load_state_dict(agent.state_dict())
        
        # Mixing network
        self.mixing_network = MixingNetwork(num_agents, action_size, 64)
        self.target_mixing_network = MixingNetwork(num_agents, action_size, 64)
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
        # Оптимизаторы
        self.agent_optimizers = [
            optim.Adam(agent.parameters(), lr=QMIX_CONFIG.learning_rate)
            for agent in self.agent_networks
        ]
        self.mixing_optimizer = optim.Adam(
            self.mixing_network.parameters(),
            lr=QMIX_CONFIG.learning_rate
        )
        
        # Буфер опыта
        self.buffer = ExperienceBuffer(QMIX_CONFIG.buffer_size)
        self.update_counter = 0
        self.epsilon = QMIX_CONFIG.epsilon_start
    
    def add_experience(self, state, actions, rewards, next_state, done):
        """Добавить опыт в буфер"""
        self.buffer.add(state, actions, rewards, next_state, done)
    
    def select_actions(self, obs: np.ndarray) -> np.ndarray:
        """Выбрать действия для каждого агента (ε-жадная стратегия)"""
        actions = []
        with torch.no_grad():
            for i, agent_net in enumerate(self.agent_networks):
                obs_tensor = torch.FloatTensor(obs[i:i+1]).unsqueeze(0)
                q_values, _ = agent_net(obs_tensor)
                
                if np.random.random() < self.epsilon:
                    action = np.random.randint(0, self.action_size)
                else:
                    action = q_values.argmax(dim=1).item()
                
                actions.append(action)
        
        return np.array(actions)
    
    def train_step(self):
        """Выполнить один шаг обучения"""
        if not self.buffer.is_ready(QMIX_CONFIG.batch_size):
            return None
        
        # Выборка батча
        batch = self.buffer.sample(QMIX_CONFIG.batch_size)
        
        # Преобразовать в тензоры
        states = torch.FloatTensor(np.array(batch['states']))
        actions = torch.LongTensor(np.array(batch['actions']))
        rewards = torch.FloatTensor(np.array(batch['rewards']))
        next_states = torch.FloatTensor(np.array(batch['next_states']))
        dones = torch.FloatTensor(np.array(batch['dones']))
        
        # Вычислить текущие Q-значения
        q_values_list = []
        for i, agent_net in enumerate(self.agent_networks):
            q_vals, _ = agent_net(states[:, i:i+1, :])
            q_values_list.append(q_vals)
        
        q_values = torch.stack(q_values_list, dim=1)  # [batch, num_agents, actions]
        
        # Вычислить целевые Q-значения
        with torch.no_grad():
            q_targets_list = []
            for i, target_net in enumerate(self.target_networks):
                q_targets, _ = target_net(next_states[:, i:i+1, :])
                q_targets_list.append(q_targets)
            
            q_targets = torch.stack(q_targets_list, dim=1)
        
        # Вычислить глобальное Q через Mixing Network
        global_q = self.mixing_network(q_values, states.mean(dim=1))
        
        # Целевое глобальное Q
        with torch.no_grad():
            global_q_target = self.target_mixing_network(q_targets, next_states.mean(dim=1))
        
        # TD-ошибка
        target = rewards.mean(dim=1) + QMIX_CONFIG.gamma * global_q_target.max(dim=1) * (1 - dones)
        loss = ((global_q.max(dim=1) - target) ** 2).mean()
        
        # Оптимизация
        self.mixing_optimizer.zero_grad()
        loss.backward()
        self.mixing_optimizer.step()
        
        # Обновить целевые сети
        self.update_counter += 1
        if self.update_counter % QMIX_CONFIG.target_update_freq == 0:
            for agent, target in zip(self.agent_networks, self.target_networks):
                target.load_state_dict(agent.state_dict())
            for target in self.target_networks:
                target.load_state_dict(self.mixing_network.state_dict())
        
        # Снизить epsilon
        self.epsilon = max(
            QMIX_CONFIG.epsilon_end,
            self.epsilon * QMIX_CONFIG.epsilon_decay
        )
        
        return float(loss.item())
    
    def update_with_vcg_rewards(self, vcg_payments: np.ndarray):
        """Обновить буфер опыта с VCG платежами"""
        # Интегрировать платежи в вознаграждения
        # (Это делается в reward_manager.py)
        pass
