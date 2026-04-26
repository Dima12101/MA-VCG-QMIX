"""
Интеграционные тесты (VCG + QMIX вместе)
"""

import unittest
from dataclasses import replace
import random
import numpy as np
import torch
from src.config import (
    AUCTION_CONFIG,
    ENV_CONFIG,
    NETWORK_CONFIG,
    NODE_CONFIG,
    TASK_CONFIG,
    TRAINING_CONFIG,
)
from src.environment.environment import EdgeComputingSystem
from src.learning.trainer import QMIXTrainer

class TestIntegration(unittest.TestCase):
    """Интеграционные тесты"""
    
    def test_environment_and_trainer_pipeline(self):
        """Тест всей цепочки: среда -> аукцион -> награды -> шаг обучения QMIX."""
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        env_config = replace(ENV_CONFIG, num_nodes=3, num_devices=5, task_lambda_arrival=3.0)
        training_config = replace(
            TRAINING_CONFIG,
            batch_size=1,
            buffer_size=8,
            target_update_freq=1,
        )
        env = EdgeComputingSystem(
            env_config=env_config,
            node_config=NODE_CONFIG,
            task_config=TASK_CONFIG,
            auction_config=AUCTION_CONFIG,
        )
        network_config = replace(
            NETWORK_CONFIG,
            obs_size=env.observation_size,
            state_size=env_config.num_nodes * env.observation_size,
        )
        trainer = QMIXTrainer(
            num_agents=env_config.num_nodes,
            network_config=network_config,
            training_config=training_config,
        )

        current_state = env.get_observations()
        actions = trainer.select_actions(current_state)
        rewards, info, metrics = env.step(actions)
        next_state = env.get_observations()

        trainer.add_experience(
            state=current_state,
            actions=actions,
            rewards=rewards,
            next_state=next_state,
            done=False,
        )
        loss = trainer.train_step()

        self.assertEqual(rewards.shape, (env_config.num_nodes,))
        self.assertEqual(current_state.shape, (env_config.num_nodes, env.observation_size))
        self.assertEqual(next_state.shape, (env_config.num_nodes, env.observation_size))
        self.assertEqual(len(actions), env_config.num_nodes)
        self.assertLessEqual(info["accepted"] + info["rejected"], env_config.num_devices)
        self.assertTrue(np.isfinite(metrics["social_welfare"]))
        self.assertIsNotNone(loss)
        self.assertGreaterEqual(loss, 0.0)

if __name__ == '__main__':
    unittest.main()
