import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.edge_network import EdgeNetwork
from src.learning.trainer import QMIXTrainer
from src.mechanisms.payments import calculate_vcg_payments
from src.config import ENV_CONFIG
from src.learning.metrics import (
    calculate_gini_coefficient,
    calculate_fairness_index,
    calculate_social_welfare,
    calculate_acceptance_rate,
    calculate_avg_latency,
)

class Scenario1Baseline:
    """Сценарий 1: Базовый сценарий"""
    
    def __init__(self):
        self.env = EdgeNetwork()
        self.trainer = QMIXTrainer(
            num_agents=ENV_CONFIG.num_edges,
            obs_size=8,  # 8 компонентов наблюдения
            action_size=4  # Accept, Reject, Priority_High, Priority_Low
        )
        self.results = []
        self.results_dir = Path('experiments/results/scenario_1')
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, num_episodes: int = 1):
        """Запустить сценарий"""
        print(f"Запуск Сценария 1: Базовый сценарий")
        print(f"Параметры: {ENV_CONFIG}")
        print(f"Количество эпизодов: {num_episodes}\n")
        
        for episode in range(num_episodes):
            print(f"Эпизод {episode + 1}/{num_episodes}")
            
            # Сбросить окружение
            self.env = EdgeNetwork()
            
            episode_results = {
                'time': [],
                'accepted_tasks': [],
                'rejected_tasks': [],
                'avg_latency': [],
                'social_welfare': [],
                'gini_payment': [],
                'fairness_index': [],
                'load_per_node': [[] for _ in range(ENV_CONFIG.num_edges)],
            }
            
            accepted_total = 0
            rejected_total = 0
            latencies = []
            
            # Симуляция
            for step in range(ENV_CONFIG.episode_length):
                # Выполнить один шаг в окружении
                metrics = self.env.step()
                
                # Сгенерировать полезности и стоимости (для примера)
                utility_matrix = np.random.uniform(0.5, 1.0, (ENV_CONFIG.num_devices, ENV_CONFIG.num_edges))
                cost_matrix = np.random.uniform(0.2, 0.5, (ENV_CONFIG.num_devices, ENV_CONFIG.num_edges))
                
                # Создать случайное распределение (в реальности из QMIX)
                allocation = np.random.randint(0, 2, (ENV_CONFIG.num_devices, ENV_CONFIG.num_edges))
                for i in range(ENV_CONFIG.num_devices):
                    if allocation[i].sum() == 0:
                        allocation[i, np.random.randint(0, ENV_CONFIG.num_edges)] = 1
                
                # Вычислить платежи VCG
                payments, sw = calculate_vcg_payments(allocation, utility_matrix, cost_matrix)
                
                # Метрики
                accepted = min(metrics['accepted'], 100)
                rejected = min(metrics['rejected'], 100)
                gini = calculate_gini_coefficient(payments.tolist())
                fairness = calculate_fairness_index(allocation)
                
                episode_results['time'].append(step)
                episode_results['accepted_tasks'].append(accepted)
                episode_results['rejected_tasks'].append(rejected)
                episode_results['avg_latency'].append(metrics['avg_latency'])
                episode_results['social_welfare'].append(sw)
                episode_results['gini_payment'].append(gini)
                episode_results['fairness_index'].append(fairness)
                
                for i, edge in enumerate(self.env.edges):
                    episode_results['load_per_node'][i].append(edge.load)
                
                accepted_total += accepted
                rejected_total += rejected
                
                if step % 100 == 0:
                    print(f"  Шаг {step}: SW={sw:.1f}, Gini={gini:.3f}, Acceptance={accepted}%")
            
            # Итоги эпизода
            avg_acceptance = calculate_acceptance_rate(accepted_total, accepted_total + rejected_total)
            
            print(f"Итоги эпизода {episode + 1}:")
            print(f"  Средняя задержка: {np.mean(episode_results['avg_latency']):.2f} мс")
            print(f"  Процент принятых: {avg_acceptance:.1f}%")
            print(f"  Среднее SW: {np.mean(episode_results['social_welfare']):.1f}")
            print(f"  Средний Джини: {np.mean(episode_results['gini_payment']):.3f}\n")
            
            # Сохранить результаты
            self.results.append(episode_results)
        
        # Сохранить в CSV
        self._save_results()
    
    def _save_results(self):
        """Сохранить результаты в файлы"""
        for i, episode_results in enumerate(self.results):
            df = pd.DataFrame({
                'time': episode_results['time'],
                'accepted_tasks': episode_results['accepted_tasks'],
                'rejected_tasks': episode_results['rejected_tasks'],
                'avg_latency': episode_results['avg_latency'],
                'social_welfare': episode_results['social_welfare'],
                'gini_payment': episode_results['gini_payment'],
                'fairness_index': episode_results['fairness_index'],
            })
            
            # Добавить нагрузки узлов
            for j in range(ENV_CONFIG.num_edges):
                df[f'load_node_{j}'] = episode_results['load_per_node'][j]
            
            csv_path = self.results_dir / f'scenario_1_episode_{i}.csv'
            df.to_csv(csv_path, index=False)
            print(f"Результаты сохранены в {csv_path}")
        
        # Сохранить общую статистику
        summary = {
            'scenario': 'Baseline',
            'num_edges': ENV_CONFIG.num_edges,
            'num_devices': ENV_CONFIG.num_devices,
            'arrival_rate': ENV_CONFIG.arrival_rate,
            'episode_length': ENV_CONFIG.episode_length,
            'avg_acceptance_rate': np.mean([
                np.mean(r['accepted_tasks']) for r in self.results
            ]),
            'avg_sw': np.mean([
                np.mean(r['social_welfare']) for r in self.results
            ]),
            'avg_gini': np.mean([
                np.mean(r['gini_payment']) for r in self.results
            ]),
        }
        
        summary_df = pd.DataFrame([summary])
        summary_path = self.results_dir / 'summary.csv'
        summary_df.to_csv(summary_path, index=False)

if __name__ == '__main__':
    scenario = Scenario1Baseline()
    scenario.run(num_episodes=2)
