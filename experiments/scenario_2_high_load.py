"""Сценарий 2: динамическая нагрузка с пиками."""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ENV_CONFIG, TRAINING_CONFIG
from src.learning.simulator import Simulator


def run_high_load_scenario():
    """Запустить сценарий с перегрузкой и всплесками нагрузки."""
    env_config = replace(
        ENV_CONFIG,
        task_lambda_arrival=5.0,
        load_spike_probability=0.15,
        load_spike_multiplier=2.0,
    )
    training_config = replace(TRAINING_CONFIG, max_steps_per_episode=120)
    simulator = Simulator(env_config=env_config, training_config=training_config)

    print("=" * 60)
    print("SCENARIO 2: LOAD SPIKES")
    print("=" * 60)
    simulator.run()
    return simulator.save_results('scenario_2', Path('experiments/results/scenario_2'))


if __name__ == '__main__':
    run_high_load_scenario()
