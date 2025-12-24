.PHONY: help install test run-all run-baseline clean docs

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run all tests"
	@echo "  make run-baseline  - Run baseline scenario only"
	@echo "  make run-all       - Run all 4 scenarios"
	@echo "  make plots         - Generate visualization plots"
	@echo "  make clean         - Clean up generated files"

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v --cov=src

run-baseline:
	python experiments/scenario_1_baseline.py

run-all:
	python main_run_all_scenarios.py

plots:
	python visualization/plot_results.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
	rm -rf experiments/results/*.json experiments/results/plots/*.png
	rm -rf experiments/logs/*.log
