"""Сценарий 3: разнородные вычислительные ресурсы."""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import NODE_CONFIG
from src.learning.simulator import Simulator


def run_heterogeneous_scenario():
    """Запустить сценарий с неоднородными edge-узлами."""
    node_config = replace(NODE_CONFIG, heterogeneous_resources=True)
    simulator = Simulator(node_config=node_config)

    print("=" * 60)
    print("SCENARIO 3: HETEROGENEOUS RESOURCES")
    print("=" * 60)
    simulator.run()
    return simulator.save_results('scenario_3', Path('experiments/results/scenario_3'))


if __name__ == '__main__':
    run_heterogeneous_scenario()
