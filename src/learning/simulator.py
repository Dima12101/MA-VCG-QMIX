"""
Симулятор работы всей системы.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.environment.environment import EdgeComputingSystem
from src.learning.trainer import QMIXTrainer
from src.config import (EnvironmentConfig,
                        NodeConfig,
                        TaskConfig,
                        NetworkConfig,
                        TrainingConfig,
                        AuctionConfig)

class Simulator:

    def __init__(self,
                 env_config: EnvironmentConfig = None,
                 node_config: NodeConfig = None,
                 task_config: TaskConfig = None,
                 network_config: NetworkConfig = None,
                 training_config: TrainingConfig = None,
                 auction_config: AuctionConfig = None):
        
        # Параметры
        self.env_config = env_config or EnvironmentConfig()
        self.edge_config = node_config or NodeConfig()
        self.task_config = task_config or TaskConfig()
        self.network_config = network_config or NetworkConfig()
        self.training_config = training_config or TrainingConfig()
        self.auction_config = auction_config or AuctionConfig()

        # Создать среду
        self.env = EdgeComputingSystem(self.env_config, self.edge_config, self.task_config, self.auction_config)

        # Создать сеть обучения
        effective_network_config = NetworkConfig(
            hidden_size=self.network_config.hidden_size,
            obs_size=self.env.observation_size,
            action_size=self.network_config.action_size,
            state_size=self.env_config.num_nodes * self.env.observation_size,
        )
        self.trainer = QMIXTrainer(
            self.env_config.num_nodes,
            effective_network_config,
            self.training_config,
        )
    
    def run(self): 
        self.results = []
        for episode in range(self.training_config.num_episodes):
            self.env.reset()

            episode_results = {
                'time': [],
                'accepted_tasks': [],
                'rejected_tasks': [],
                'completed_tasks': [],
                'avg_latency': [],
                'acceptance_rate': [],
                'resource_utilization': [],
                'gini_payment': [],
                'fairness_index': [],
                'social_welfare': [],
                'td_error': []
            }

            # Симуляция
            for step in range(self.training_config.max_steps_per_episode):
                current_state = self.env.get_observations()

                # 1. QMIX выбирает действия (стратегию узлов)
                actions = self.trainer.select_actions(current_state)
                
                # 2. Применить действия в окружении
                rewards, info, metrics = self.env.step(actions)
                next_state = self.env.get_observations()              

                # 3. Сохранить опыт
                self.trainer.add_experience(
                    state = current_state,
                    actions = actions,
                    rewards = rewards,
                    next_state = next_state,
                    done = step == self.training_config.max_steps_per_episode - 1
                )

                # 4. Обучить QMIX (если накоплено достаточно опыта)
                td_error = self.trainer.train_step()                    
                
                # 5. Обновить статистику
                episode_results['time'].append(step)
                episode_results['accepted_tasks'].append(info['accepted'])
                episode_results['rejected_tasks'].append(info['rejected'])
                episode_results['completed_tasks'].append(info['completed'])
                episode_results['avg_latency'].append(metrics['avg_latency'])
                episode_results['acceptance_rate'].append(metrics['acceptance_rate'])
                episode_results['resource_utilization'].append(metrics['resource_utilization'])
                episode_results['gini_payment'].append(metrics['gini_payment'])
                episode_results['fairness_index'].append(metrics['fairness_index'])
                episode_results['social_welfare'].append(metrics['social_welfare'])
                episode_results['td_error'].append(td_error)
            
            # Сохранить результаты
            self.results.append(episode_results)

            # Логировать результаты
            if episode % 50 == 0:
                print(f"Итоги эпизода {episode + 1}:")
                print(f"  Средняя задержка: {np.mean(episode_results['avg_latency']):.2f} мс")
                print(f"  Процент принятых: {np.mean(episode_results['acceptance_rate']):.1f}%")
                print(f"  Среднее SW: {np.mean(episode_results['social_welfare']):.1f}")
                print(f"  Средний Джини: {np.mean(episode_results['gini_payment']):.3f}\n")
        return self.results
    
    def save_results(self, scenario_name: str, results_path: str):
        """Сохранить результаты в файлы"""
        results_dir = Path(results_path)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохранить статистику по эпизодам
        results_episodes_dir = results_dir / 'episodes'
        results_episodes_dir.mkdir(parents=True, exist_ok=True)
        for i, episode_results in enumerate(self.results):
            df = pd.DataFrame({
                'time': episode_results['time'],
                'accepted_tasks': episode_results['accepted_tasks'],
                'rejected_tasks': episode_results['rejected_tasks'],
                'completed_tasks': episode_results['completed_tasks'],
                'avg_latency': episode_results['avg_latency'],
                'acceptance_rate': episode_results['acceptance_rate'],
                'gini_payment': episode_results['gini_payment'],
                'fairness_index': episode_results['fairness_index'],
                'social_welfare': episode_results['social_welfare'],
                'td_error': episode_results['td_error']
            })
            
            # Добавить нагрузки узлов
            loads = np.array(episode_results['resource_utilization']).T
            for j in range(self.env_config.num_nodes):
                df[f'load_node_{j}'] = loads[j]
            
            csv_path = results_episodes_dir / f'result_episode_{i}.csv'
            df.to_csv(csv_path, index=False)
        
        # Сохранить общую статистику
        summary = {
            'scenario': scenario_name,
            'num_nodes': self.env_config.num_nodes,
            'num_devices': self.env_config.num_devices,
            'arrival_rate': self.env_config.task_lambda_arrival,
            'episode_length': self.training_config.max_steps_per_episode,
            'avg_acceptance_rate': np.mean([
                np.mean(r['acceptance_rate']) for r in self.results
            ]),
            'avg_sw': np.mean([
                np.mean(r['social_welfare']) for r in self.results
            ]),
            'avg_gini': np.mean([
                np.mean(r['gini_payment']) for r in self.results
            ]),
        }
        summary_df = pd.DataFrame([summary])
        summary_path = results_dir / 'summary.csv'
        summary_df.to_csv(summary_path, index=False)
        return summary
