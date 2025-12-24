import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUAgent(nn.Module):
    """Агент с GRU для запоминания истории"""
    
    def __init__(self, obs_size: int, hidden_size: int, action_size: int):
        super().__init__()
        self.obs_size = obs_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        
        self.gru = nn.GRU(obs_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()
    
    def forward(self, obs: torch.Tensor, hidden: torch.Tensor = None):
        """
        Args:
            obs: [batch_size, seq_len, obs_size]
            hidden: [batch_size, hidden_size]
        
        Returns:
            q_values: [batch_size, action_size]
            hidden: [batch_size, hidden_size]
        """
        if hidden is None:
            hidden = torch.zeros(obs.size(0), self.hidden_size)
        
        # GRU слой
        gru_out, hidden = self.gru(obs, hidden.unsqueeze(0))
        gru_out = gru_out[:, -1, :]  # Взять последний выход
        
        # Полносвязные слои
        x = self.relu(self.fc1(gru_out))
        q_values = self.fc2(x)
        
        return q_values, hidden.squeeze(0)

class MixingNetwork(nn.Module):
    """Mixing Network для объединения локальных Q-функций в глобальную"""
    
    def __init__(self, num_agents: int, num_actions: int, hidden_size: int):
        super().__init__()
        self.num_agents = num_agents
        self.num_actions = num_actions
        
        # Гиперсеть для генерации весов
        self.hyper_w = nn.Linear(num_agents, num_agents * hidden_size)
        self.hyper_b = nn.Linear(num_agents, hidden_size)
        
        # Финальный слой
        self.fc = nn.Linear(hidden_size, num_actions)
    
    def forward(self, q_values: torch.Tensor, state: torch.Tensor = None):
        """
        Args:
            q_values: [batch_size, num_agents, num_actions]
            state: [batch_size, state_size] (опционально)
        
        Returns:
            global_q: [batch_size, num_actions]
        """
        batch_size = q_values.size(0)
        
        # Линейное объединение с неотрицательными весами
        # Это обеспечивает монотонность
        w = F.softmax(self.hyper_w(state), dim=1)  # [batch, num_agents * hidden]
        b = self.hyper_b(state)  # [batch, hidden]
        
        # Reshape и объединение
        w = w.view(batch_size, self.num_agents, -1)  # [batch, num_agents, hidden]
        q_vals = q_values[:, :, 0]  # Берём первое действие для простоты
        
        # Взвешенная сумма
        mixed = torch.sum(w * q_vals.unsqueeze(2), dim=1)  # [batch, hidden]
        mixed = mixed + b
        
        # Финальный выход
        global_q = self.fc(F.relu(mixed))
        
        return global_q
