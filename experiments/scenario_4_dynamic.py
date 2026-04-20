"""Сценарий 4: отказ части узлов и восстановление."""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ENV_CONFIG, TRAINING_CONFIG
from src.learning.simulator import Simulator


def run_dynamic_scenario():
    """Запустить сценарий с отказами узлов в середине эпизода."""
    training_config = replace(TRAINING_CONFIG, max_steps_per_episode=200)
    env_config = replace(
        ENV_CONFIG,
        failure_fraction=0.2,
        failure_start_step=100,
        failure_recovery_steps=100,
    )
    simulator = Simulator(env_config=env_config, training_config=training_config)

    print("=" * 60)
    print("SCENARIO 4: NODE FAILURES")
    print("=" * 60)
    simulator.run()
    return simulator.save_results('scenario_4', Path('experiments/results/scenario_4'))


if __name__ == '__main__':
    run_dynamic_scenario()
