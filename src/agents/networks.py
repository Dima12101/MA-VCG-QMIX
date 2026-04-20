import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUAgent(nn.Module):
    """Агент с GRU для запоминания истории"""
    
    def __init__(self, obs_size: int, hidden_size: int, action_size: int):
        super().__init__()
        self.obs_size = obs_size        # Размер слоя запоминания
        self.hidden_size = hidden_size  # Размер скрытого слоя
        self.action_size = action_size  # Размер слоя действий
        
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
            hidden = torch.zeros(
                obs.size(0),
                self.hidden_size,
                device=obs.device,
                dtype=obs.dtype,
            )
        
        # GRU слой
        gru_out, hidden = self.gru(obs, hidden.unsqueeze(0))
        gru_out = gru_out[:, -1, :]  # Взять последний выход
        
        # Полносвязные слои
        x = self.relu(self.fc1(gru_out))
        q_values = self.fc2(x)
        
        return q_values, hidden.squeeze(0)

class MixingNetwork(nn.Module):
    """Mixing Network для объединения локальных Q-функций в глобальную"""
    
    def __init__(self, num_agents: int, state_size: int, hidden_size: int):
        super().__init__()
        self.num_agents = num_agents
        self.state_size = state_size
        self.hidden_size = hidden_size
        
        self.hyper_w1 = nn.Linear(state_size, num_agents * hidden_size)
        self.hyper_b1 = nn.Linear(state_size, hidden_size)
        self.hyper_w2 = nn.Linear(state_size, hidden_size)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor):
        """
        Args:
            agent_qs: [batch_size, num_agents]
            state: [batch_size, state_size]
        
        Returns:
            global_q: [batch_size]
        """
        batch_size = agent_qs.size(0)

        w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.num_agents, self.hidden_size)
        b1 = self.hyper_b1(state).view(batch_size, 1, self.hidden_size)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.hidden_size, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(batch_size)
