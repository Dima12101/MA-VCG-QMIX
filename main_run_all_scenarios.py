"""
Главный скрипт для запуска всех 4 сценариев подряд
"""

import sys
from pathlib import Path

# Добавить src в путь
sys.path.insert(0, str(Path(__file__).parent))

from experiments.scenario_1_baseline import run_baseline_scenario
from experiments.scenario_2_high_load import run_high_load_scenario
from experiments.scenario_3_heterogeneous import run_heterogeneous_scenario
from experiments.scenario_4_dynamic import run_dynamic_scenario

def main():
    """Запустить все сценарии"""
    
    print("\n" + "=" * 80)
    print("INTEGRATED EDGE RESOURCE MANAGEMENT SYSTEM - ALL SCENARIOS")
    print("=" * 80 + "\n")
    
    results = {}
    
    # Запустить каждый сценарий
    scenarios = [
        ("Baseline", run_baseline_scenario),
        ("High Load", run_high_load_scenario),
        ("Heterogeneous", run_heterogeneous_scenario),
        ("Dynamic", run_dynamic_scenario),
    ]
    
    for scenario_name, scenario_func in scenarios:
        print(f"\n{'#' * 80}")
        print(f"# Running: {scenario_name}")
        print(f"{'#' * 80}\n")
        
        try:
            result = scenario_func()
            results[scenario_name] = result
            print(f"\n✅ {scenario_name} scenario completed successfully!")
        except Exception as e:
            print(f"\n❌ {scenario_name} scenario FAILED!")
            print(f"Error: {str(e)}")
            results[scenario_name] = None
    
    # Итоговый отчет
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for scenario_name, result in results.items():
        status = "✅ COMPLETED" if result is not None else "❌ FAILED"
        print(f"{scenario_name}: {status}")
    
    print("\nAll results saved to: experiments/results/")
    print("Generate plots with: python visualization/plot_results.py")

if __name__ == '__main__':
    main()
