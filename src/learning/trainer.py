import torch
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
        self.num_agents = num_agents
        self.obs_size = network_config.obs_size
        self.hidden_size = network_config.hidden_size
        self.action_size = network_config.action_size
        self.training_config = training_config
        
        # Создать агентские сети
        self.agent_networks = [
            GRUAgent(self.obs_size, self.hidden_size, self.action_size)
            for _ in range(self.num_agents)
        ]
        
        # Целевые сети (для стабилизации)
        self.target_networks = [
            GRUAgent(self.obs_size,  self.hidden_size, self.action_size)
            for _ in range(self.num_agents)
        ]
        
        # Скопировать веса
        for agent, target in zip(self.agent_networks, self.target_networks):
            target.load_state_dict(agent.state_dict())
        
        # Mixing network
        self.mixing_network = MixingNetwork(self.num_agents, self.action_size,  self.hidden_size)
        self.target_mixing_network = MixingNetwork(self.num_agents, self.action_size,  self.hidden_size)
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
        # Оптимизаторы
        self.agent_optimizers = [
            optim.Adam(agent.parameters(), lr=self.training_config.learning_rate)
            for agent in self.agent_networks
        ]
        self.mixing_optimizer = optim.Adam(
            self.mixing_network.parameters(),
            lr=self.training_config.learning_rate
        )
        
        # Буфер опыта
        self.buffer = ExperienceBuffer(self.training_config.buffer_size)
        self.update_counter = 0
        self.epsilon = self.training_config.epsilon_start
    
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
        global_q = self.mixing_network(q_values, states.mean(dim=2))
        
        # Целевое глобальное Q
        with torch.no_grad():
            global_q_target = self.target_mixing_network(q_targets, next_states.mean(dim=2))
        
        # TD-ошибка
        loss = self.calculate_td_error(
            q_current=global_q.max(dim=1).values,
            reward=rewards.mean(dim=1),
            q_next=global_q_target.max(dim=1).values,
            gamma=self.training_config.gamma,
            done=dones
        )
        
        # Оптимизация
        self.mixing_optimizer.zero_grad()
        loss.backward()
        self.mixing_optimizer.step()
        
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
        q_current: float,
        reward: float,
        q_next: float,
        gamma: float = 0.99,
        done: bool = False
    ) -> float:
        """TD-ошибка для обучения QMIX"""
        target = reward + (gamma * q_next) * (1 - done)
        error = abs(target - q_current)
        return error.mean()
