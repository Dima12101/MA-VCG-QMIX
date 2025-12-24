import numpy as np
from collections import deque
from typing import Tuple

class ExperienceBuffer:
    """Буфер для хранения опыта взаимодействия с окружением"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, actions, rewards, next_state, done):
        """Добавить переход в буфер"""
        self.buffer.append({
            'state': state,
            'actions': actions,
            'rewards': rewards,
            'next_state': next_state,
            'done': done,
        })
    
    def sample(self, batch_size: int) -> Tuple:
        """Выборка мини-батча из буфера"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        batch = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
        }
        
        for i in indices:
            transition = self.buffer[i]
            batch['states'].append(transition['state'])
            batch['actions'].append(transition['actions'])
            batch['rewards'].append(transition['rewards'])
            batch['next_states'].append(transition['next_state'])
            batch['dones'].append(transition['done'])
        
        return batch
    
    def __len__(self):
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Достаточно ли опыта для обучения?"""
        return len(self.buffer) >= batch_size
