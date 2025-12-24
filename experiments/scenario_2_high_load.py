"""
Сценарий 2: Высокая нагрузка (High Load Scenario)
Интенсивность трафика λ = 5.0 (2x от baseline)
"""

import numpy as np
import torch
from src.environment.edge_network import EdgeNetwork
from src.mechanisms.vcg_auction import VCGAuction
from src.learning.trainer import QMIXTrainer
from src.config import ENV_CONFIG

def run_high_load_scenario():
    """Запустить сценарий высокой нагрузки"""
    
    print("=" * 60)
    print("SCENARIO 2: HIGH LOAD (2x traffic intensity)")
    print("=" * 60)
    
    # Параметры (от baseline, но λ = 5.0)
    config = ENV_CONFIG.copy()
    config.lambda_arrival = 5.0  # 2x от baseline (2.5)
    config.num_episodes = 500
    config.max_steps_per_episode = 1000
    
    # Создать окружение
    env = EdgeNetwork(
        num_edges=config.num_edges,
        num_devices=config.num_devices,
        lambda_arrival=config.lambda_arrival
    )
    
    # Создать VCG аукцион
    auction = VCGAuction(config.num_devices, config.num_edges)
    
    # Создать тренер
    trainer = QMIXTrainer(
        num_agents=config.num_edges,
        obs_size=config.obs_size,
        action_size=config.action_size,
        learning_rate=config.learning_rate,
        gamma=config.gamma
    )
    
    results = {
        'latencies': [],
        'acceptance_rates': [],
        'gini_coefficients': [],
        'social_welfare': [],
        'node_loads': []
    }
    
    # Обучение
    for episode in range(config.num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_latencies = []
        episode_accepted = 0
        episode_total = 0
        
        for step in range(config.max_steps_per_episode):
            # QMIX выбирает действия
            actions = trainer.select_actions(state)
            
            # Применить действия в окружении
            next_state, rewards, latencies, accepted = env.step(actions)
            
            # Сохранить опыт
            trainer.store_experience({
                'state': state,
                'actions': actions,
                'rewards': rewards,
                'next_state': next_state,
                'done': False
            })
            
            # Обновить статистику
            episode_latencies.extend(latencies)
            episode_accepted += accepted
            episode_total += len(latencies)
            episode_reward += np.sum(rewards)
            
            # Обучить QMIX
            if trainer.buffer.size() > config.batch_size:
                loss = trainer.train()
            
            state = next_state
        
        # Запустить VCG аукцион
        valuations = np.random.uniform(0.5, 1.0, (config.num_devices, config.num_edges))
        costs = np.random.uniform(0.2, 0.5, (config.num_devices, config.num_edges))
        vcg_result = auction.run_auction(valuations, costs, episode)
        
        # Логировать результаты
        if episode % 50 == 0:
            avg_latency = np.mean(episode_latencies) if episode_latencies else 0
            acceptance_rate = episode_accepted / episode_total if episode_total > 0 else 0
            gini = auction.get_average_gini()
            
            print(f"Episode {episode + 1}/{config.num_episodes}")
            print(f"  Avg Latency: {avg_latency:.2f} ms")
            print(f"  Acceptance Rate: {acceptance_rate:.1%}")
            print(f"  Gini Coefficient: {gini:.3f}")
            print(f"  Social Welfare: {vcg_result.social_welfare:.2f}")
            print()
            
            results['latencies'].append(avg_latency)
            results['acceptance_rates'].append(acceptance_rate)
            results['gini_coefficients'].append(gini)
            results['social_welfare'].append(vcg_result.social_welfare)
    
    # Сохранить результаты
    import json
    with open('experiments/results/scenario_2_high_load.json', 'w') as f:
        json.dump(results, f)
    
    print("\n✅ Сценарий 2 завершён!")
    print(f"Результаты сохранены в experiments/results/scenario_2_high_load.json")
    
    return results

if __name__ == '__main__':
    results = run_high_load_scenario()
