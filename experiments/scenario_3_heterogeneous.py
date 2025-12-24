"""
Сценарий 3: Гетерогенные узлы (Heterogeneous Nodes)
Узлы класса A (быстрые), B (средние), C (медленные)
"""

import numpy as np
from src.environment.edge_network import EdgeNetwork
from src.config import ENV_CONFIG

class HeterogeneousNodeConfig:
    """Конфигурация для гетерогенных узлов"""
    
    def __init__(self):
        self.node_types = {
            'A': {'capacity': 100, 'latency_base': 10, 'energy': 5},    # Быстрые
            'B': {'capacity': 50, 'latency_base': 30, 'energy': 8},     # Средние
            'C': {'capacity': 20, 'latency_base': 60, 'energy': 12}     # Медленные
        }

def run_heterogeneous_scenario():
    """Запустить сценарий с гетерогенными узлами"""
    
    print("=" * 60)
    print("SCENARIO 3: HETEROGENEOUS NODES")
    print("=" * 60)
    
    # Конфигурация
    config = ENV_CONFIG.copy()
    het_config = HeterogeneousNodeConfig()
    config.num_edges = 6  # 2 узла каждого типа
    config.num_episodes = 300
    
    # Создать окружение
    env = EdgeNetwork(
        num_edges=config.num_edges,
        num_devices=config.num_devices,
        lambda_arrival=config.lambda_arrival
    )
    
    # Назначить типы узлов
    node_types = ['A', 'A', 'B', 'B', 'C', 'C']
    node_capacities = [
        het_config.node_types[t]['capacity'] for t in node_types
    ]
    
    results = {
        'node_types': node_types,
        'task_distribution': {t: 0 for t in set(node_types)},
        'latencies_by_type': {t: [] for t in set(node_types)},
        'utilization': {t: 0 for t in set(node_types)}
    }
    
    # Обучение
    for episode in range(config.num_episodes):
        state = env.reset()
        
        for step in range(500):
            # Простое распределение на основе типа узла
            actions = np.zeros(config.num_edges)
            
            # Вычислить выбор на основе пропускной способности
            for i in range(config.num_edges):
                node_type = node_types[i]
                capacity = het_config.node_types[node_type]['capacity']
                # Узлы с большей пропускной способностью получают больше задач
                actions[i] = capacity / 100
            
            # Нормализовать
            actions = actions / (np.sum(actions) + 1e-8)
            
            next_state, rewards, latencies, accepted = env.step(actions)
            
            # Собрать статистику
            for i, (latency, node_type) in enumerate(zip(latencies, node_types)):
                if accepted > 0:
                    results['latencies_by_type'][node_type].append(latency)
                    results['task_distribution'][node_type] += accepted / len(node_types)
            
            state = next_state
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{config.num_episodes}")
            for node_type in set(node_types):
                if results['latencies_by_type'][node_type]:
                    avg_latency = np.mean(results['latencies_by_type'][node_type])
                    print(f"  Type {node_type}: Avg Latency = {avg_latency:.2f} ms")
            print()
    
    # Вычислить итоговую статистику
    for node_type in set(node_types):
        total_tasks = results['task_distribution'][node_type]
        total_capacity = sum(het_config.node_types[t]['capacity'] 
                           for t in set(node_types) 
                           if t == node_type)
        results['utilization'][node_type] = min(total_tasks / total_capacity, 1.0)
    
    # Сохранить результаты
    import json
    with open('experiments/results/scenario_3_heterogeneous.json', 'w') as f:
        json.dump(results, f)
    
    print("✅ Сценарий 3 завершён!")
    
    return results

if __name__ == '__main__':
    results = run_heterogeneous_scenario()
