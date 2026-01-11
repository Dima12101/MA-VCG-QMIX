"""
Сценарий 1: Базовый сценарий
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.learning.simulator import Simulator

def run_baseline_scenario():
    """Запустить базовый сценарий"""

    simulator = Simulator()

    print("=" * 60)
    print("SCENARIO 1: BASELINE")
    print("=" * 60)
    simulator.run()
    simulator.save_results('scenario_1', Path('experiments/results/scenario_1'))

if __name__ == '__main__':
    run_baseline_scenario()
