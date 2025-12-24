"""
Логирование событий и метрик
"""

import logging
import json
from pathlib import Path
from datetime import datetime

class ExperimentLogger:
    """Класс для логирования экспериментов"""
    
    def __init__(self, log_dir: str = 'experiments/logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Создать logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Обработчик для файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"experiment_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Формат логирования
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def log_episode_start(self, episode: int, num_episodes: int):
        """Логировать начало эпизода"""
        self.logger.info(f"Starting episode {episode + 1}/{num_episodes}")
    
    def log_step(self, step: int, metrics: dict):
        """Логировать метрики шага"""
        self.logger.info(f"Step {step}: {json.dumps(metrics, indent=2)}")
    
    def log_episode_summary(self, episode: int, summary: dict):
        """Логировать итоги эпизода"""
        self.logger.info(f"Episode {episode} summary: {json.dumps(summary, indent=2)}")
    
    def log_error(self, error_msg: str):
        """Логировать ошибку"""
        self.logger.error(error_msg)
