import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from src.agents.networks import GRUAgent, MixingNetwork
from src.agents.experience_buffer import ExperienceBuffer
from src.config import NetworkConfig, TrainingConfig

class QMIXTrainer:
    """Тренер для обучения QMIX агентов"""
    
    def __init__(self, num_agents: int,
                 network_config: NetworkConfig = None,
                 training_config: TrainingConfig = None):
        network_config = network_config or NetworkConfig()
        training_config = training_config or TrainingConfig()
        self.num_agents = num_agents
        self.obs_size = network_config.obs_size
        self.hidden_size = network_config.hidden_size
        self.action_size = network_config.action_size
        self.training_config = training_config
        self.state_size = network_config.state_size or (num_agents * self.obs_size)
        
        # Создать агентские сети
        self.agent_networks = nn.ModuleList([
            GRUAgent(self.obs_size, self.hidden_size, self.action_size)
            for _ in range(self.num_agents)
        ])
        
        # Целевые сети (для стабилизации)
        self.target_networks = nn.ModuleList([
            GRUAgent(self.obs_size,  self.hidden_size, self.action_size)
            for _ in range(self.num_agents)
        ])
        
        # Скопировать веса
        for agent, target in zip(self.agent_networks, self.target_networks):
            target.load_state_dict(agent.state_dict())
        
        # Mixing network
        self.mixing_network = MixingNetwork(self.num_agents, self.state_size, self.hidden_size)
        self.target_mixing_network = MixingNetwork(self.num_agents, self.state_size, self.hidden_size)
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
        # Оптимизаторы
        self.optimizer = optim.Adam(
            list(self.agent_networks.parameters()) + list(self.mixing_network.parameters()),
            lr=self.training_config.learning_rate
        )
        
        # Буфер опыта
        self.buffer = ExperienceBuffer(self.training_config.buffer_size)
        self.update_counter = 0
        self.epsilon = self.training_config.epsilon_start
    
    def add_experience(self, state, actions, rewards, next_state, done):
        """Добавить опыт в буфер"""
        self.buffer.add(state, actions, rewards, next_state, done)

    def _extract_action_mask(self, obs: np.ndarray) -> np.ndarray:
        """Извлечь бинарную маску допустимых действий из хвоста наблюдения."""
        mask = np.asarray(obs[-self.action_size:], dtype=np.float32) > 0.5
        if not np.any(mask):
            mask = np.zeros(self.action_size, dtype=bool)
            mask[0] = True
        return mask

    @staticmethod
    def _masked_argmax(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """Выбрать argmax только по допустимым действиям."""
        masked_q_values = q_values.masked_fill(~action_mask, torch.finfo(q_values.dtype).min)
        return masked_q_values.argmax(dim=1)
    
    def select_actions(self, obs: np.ndarray) -> np.ndarray:
        """Выбрать действия для каждого агента (ε-жадная стратегия)"""
        actions = []
        with torch.no_grad():
            for i, agent_net in enumerate(self.agent_networks):
                obs_tensor = torch.FloatTensor(obs[i:i+1]).unsqueeze(0)
                q_values, _ = agent_net(obs_tensor)
                valid_actions = np.flatnonzero(self._extract_action_mask(obs[i]))
                
                if np.random.random() < self.epsilon:
                    action = int(np.random.choice(valid_actions))
                else:
                    mask_tensor = torch.BoolTensor(self._extract_action_mask(obs[i])).unsqueeze(0)
                    action = int(self._masked_argmax(q_values, mask_tensor).item())
                
                actions.append(action)
        
        return np.array(actions)      
    
    def train_step(self):
        """Выполнить один шаг обучения"""
        if not self.buffer.is_ready(self.training_config.batch_size):
            return None
        
        # Выборка батча
        batch = self.buffer.sample(self.training_config.batch_size)
        
        # Преобразовать в тензоры
        states = torch.FloatTensor(np.array(batch['states']))
        actions = torch.LongTensor(np.array(batch['actions']))
        rewards = torch.FloatTensor(np.array(batch['rewards']))
        next_states = torch.FloatTensor(np.array(batch['next_states']))
        dones = torch.FloatTensor(np.array(batch['dones']))
        state_vectors = states.reshape(states.size(0), -1)
        next_state_vectors = next_states.reshape(next_states.size(0), -1)
        team_rewards = rewards.mean(dim=1)
        
        # Вычислить текущие Q-значения
        chosen_q_values = []
        for i, agent_net in enumerate(self.agent_networks):
            q_vals, _ = agent_net(states[:, i:i+1, :])
            chosen_q = q_vals.gather(1, actions[:, i:i+1]).squeeze(1)
            chosen_q_values.append(chosen_q)
        agent_qs = torch.stack(chosen_q_values, dim=1)
        
        # Вычислить целевые Q-значения
        with torch.no_grad():
            target_q_values = []
            for i, target_net in enumerate(self.target_networks):
                q_online_next, _ = self.agent_networks[i](next_states[:, i:i+1, :])
                next_action_mask = next_states[:, i, -self.action_size:] > 0.5
                greedy_next_actions = self._masked_argmax(
                    q_online_next,
                    next_action_mask,
                ).unsqueeze(1)
                q_target_next, _ = target_net(next_states[:, i:i+1, :])
                target_q = q_target_next.gather(1, greedy_next_actions).squeeze(1)
                target_q_values.append(target_q)
            target_agent_qs = torch.stack(target_q_values, dim=1)
        
        # Вычислить глобальное Q через Mixing Network
        global_q = self.mixing_network(agent_qs, state_vectors)
        
        # Целевое глобальное Q
        with torch.no_grad():
            global_q_target = self.target_mixing_network(target_agent_qs, next_state_vectors)
        
        # TD-ошибка
        loss = self.calculate_td_error(
            q_current=global_q,
            reward=team_rewards,
            q_next=global_q_target,
            gamma=self.training_config.gamma,
            done=dones
        )
        
        # Оптимизация
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent_networks.parameters()) + list(self.mixing_network.parameters()),
            self.training_config.grad_norm_clip,
        )
        self.optimizer.step()
        
        # Обновить целевые сети
        self.update_counter += 1
        if self.update_counter % self.training_config.target_update_freq == 0:
            for agent, target in zip(self.agent_networks, self.target_networks):
                target.load_state_dict(agent.state_dict())
            self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
        # Снизить epsilon
        self.epsilon = max(
            self.training_config.epsilon_end,
            self.epsilon * self.training_config.epsilon_decay
        )
        
        return float(loss.item())
    
    def calculate_td_error(
        self,
        q_current: torch.Tensor,
        reward: torch.Tensor,
        q_next: torch.Tensor,
        gamma: float = 0.99,
        done: torch.Tensor = None
    ) -> torch.Tensor:
        """TD-ошибка для обучения QMIX"""
        if done is None:
            done = torch.zeros_like(reward)
        target = reward + (gamma * q_next) * (1 - done)
        return F.mse_loss(q_current, target)
