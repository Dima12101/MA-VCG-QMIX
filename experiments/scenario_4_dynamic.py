"""
Сценарий 4: Динамические условия (Dynamic Conditions)
Отказы узлов и восстановление в процессе работы
"""

import numpy as np
from src.environment.edge_network import EdgeNetwork
from src.config import ENV_CONFIG

def run_dynamic_scenario():
    """Запустить сценарий с динамическими условиями"""
    
    print("=" * 60)
    print("SCENARIO 4: DYNAMIC CONDITIONS (Node Failures)")
    print("=" * 60)
    
    # Конфигурация
    config = ENV_CONFIG.copy()
    config.num_episodes = 100
    config.max_steps_per_episode = 2000  # Более длинные эпизоды
    
    # Создать окружение
    env = EdgeNetwork(
        num_edges=config.num_edges,
        num_devices=config.num_devices,
        lambda_arrival=config.lambda_arrival
    )
    
    results = {
        'periods': {
            'normal': {'latencies': [], 'sw': [], 'nodes': config.num_edges},
            'failure': {'latencies': [], 'sw': [], 'nodes': config.num_edges - 1},
            'recovery': {'latencies': [], 'sw': [], 'nodes': config.num_edges},
        },
        'timeline': []
    }
    
    # Обучение
    for episode in range(config.num_episodes):
        state = env.reset()
        active_nodes = list(range(config.num_edges))
        failure_time = 500  # Отказ в шаг 500
        recovery_time = 700  # Восстановление в шаг 700
        
        for step in range(config.max_steps_per_episode):
            # Имитировать отказ узла
            if step == failure_time:
                failed_node = np.random.choice(active_nodes)
                active_nodes.remove(failed_node)
                print(f"  Node {failed_node} FAILED at step {step}")
                results['timeline'].append({'event': 'failure', 'step': step, 'node': failed_node})
            
            # Восстановить узел
            if step == recovery_time and len(active_nodes) < config.num_edges:
                recovered_node = min(set(range(config.num_edges)) - set(active_nodes))
                active_nodes.append(recovered_node)
                print(f"  Node {recovered_node} RECOVERED at step {step}")
                results['timeline'].append({'event': 'recovery', 'step': step, 'node': recovered_node})
            
            # Выбрать действия только для активных узлов
            actions = np.zeros(config.num_edges)
            for node in active_nodes:
                actions[node] = 1.0 / len(active_nodes)
            
            # Применить действия
            next_state, rewards, latencies, accepted = env.step(actions)
            
            # Записать период
            if step < failure_time:
                period = 'normal'
            elif step < recovery_time:
                period = 'failure'
            else:
                period = 'recovery'
            
            if latencies:
                results['periods'][period]['latencies'].extend(latencies)
                results['periods'][period]['sw'].append(np.sum(rewards))
            
            state = next_state
        
        if (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1}/{config.num_episodes}")
            for period in results['periods']:
                if results['periods'][period]['latencies']:
                    avg_lat = np.mean(results['periods'][period]['latencies'])
                    print(f"  {period.upper()}: Avg Latency = {avg_lat:.2f} ms")
            print()
    
    # Сохранить результаты
    import json
    with open('experiments/results/scenario_4_dynamic.json', 'w') as f:
        # Конвертировать списки в средние значения для JSON
        data = {}
        for period in results['periods']:
            data[period] = {
                'avg_latency': np.mean(results['periods'][period]['latencies']) if results['periods'][period]['latencies'] else 0,
                'avg_sw': np.mean(results['periods'][period]['sw']) if results['periods'][period]['sw'] else 0,
                'nodes': results['periods'][period]['nodes']
            }
        data['timeline'] = results['timeline']
        json.dump(data, f)
    
    print("✅ Сценарий 4 завершён!")
    
    return results

if __name__ == '__main__':
    results = run_dynamic_scenario()
