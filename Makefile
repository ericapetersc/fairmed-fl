.PHONY: clean clean-hydra setup test run

PYTHON := python
PIP := pip
REQUIREMENTS_FILE := requirements.txt

all: setup run

setup: $(REQUIREMENTS_FILE)
	@echo "Setting up the environment..."
	$(PIP) install -r $(REQUIREMENTS_FILE)

format:
	isort src/fairmed_fl/*.py
	black src/fairmed_fl/*.py
	docformatter src/fairmed_fl/*.py

run-partitioning:
	@echo "Running partitioner.py with config file: centralized_base.yaml"
	$(PYTHON) -m src.fairmed_fl.partitioner

run-centralized:
	@echo "Running main.py with config file: centralized_base.yaml"
	$(PYTHON) -m src.fairmed_fl.main

run-federated:
	@echo "Running federated.py with config file: federated_base.yaml"
	$(PYTHON) -m src.fairmed_fl.federated

run-simulation:
	@echo "Running simulation.py with config file: federated_base.yaml"
	$(PYTHON) -m src.fairmed_fl.simulation

clean:
	@echo "Cleaning caches..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -r {} +
	rm -rf .pytest_cache
	rm -rf .ipynb_checkpoints

clean-outputs:
	@echo "Cleaning clients data and hydra outputs..."
	rm -rf ./outputs
	rm -rf ./clients