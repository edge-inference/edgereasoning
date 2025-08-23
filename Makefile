.PHONY: setup jetson server collect validate info help venv
.PHONY: synthetic prefill decode tt-scaling
.PHONY: tegra-base tegra-budget tegra-scaling tegra-synthetic tegra-all
.PHONY: server-mmlu planner platform env-info

PYTHON := $(shell which python 2>/dev/null || which python3 2>/dev/null || echo "python-not-found")
DETECTED_PLATFORM := $(shell $(PYTHON) runners/collect_env.py --format platform 2>/dev/null || echo "unknown")

help:
	@echo "Edge Reasoning Evaluation Framework"
	@echo "=================================="
	@echo ""
	@echo "SETUP (Run Once):"
	@echo "  venv           - Create Python virtual environment (server only)"
	@echo "  setup          - Auto-detect platform and setup environment"
	@echo "  setup-server   - Force server platform setup"
	@echo "  setup-tegra    - Force Tegra platform setup" 
	@echo "  info           - Show device information"
	@echo "  platform       - Show detected platform only"
	@echo ""
	@echo "Tegra/Jetson Evaluations (after setup):"
	@echo "  tegra-base     - Base MMLU evaluation"
	@echo "  tegra-budget   - Budget-optimized evaluation"
	@echo "  tegra-scaling  - Test-time scaling evaluation"
	@echo "  tegra-synthetic - All synthetic benchmarks"
	@echo "  tegra-all      - All Tegra evaluations"
	@echo ""
	@echo "Individual Synthetic Benchmarks (after setup):"
	@echo "  prefill        - Prefill experiments"
	@echo "  decode         - Decode experiments"
	@echo "  tt-scaling     - Test-time scaling synthetic"
	@echo "  synthetic      - All synthetic benchmarks"
	@echo ""
	@echo "Server Evaluations (after setup):"
	@echo "  server         - Original server evaluation"
	@echo "  server-mmlu    - Server MMLU evaluation"
	@echo "  planner        - Planner evaluation"
	@echo ""
	@echo "Analysis Commands:"
	@echo "  collect        - Collect environment information"  
	@echo "  validate       - Validate analytical models"
	@echo "  env-info       - Show comprehensive environment info"
	@echo ""
	@echo "TYPICAL WORKFLOW:"
	@echo "  1. make venv && source .venv/bin/activate  # (server only)"
	@echo "  2. make setup                             # (once per platform)"
	@echo "  3. make server-mmlu                       # (run benchmarks)"

# Environment setup targets
setup:
	@if [ "$(PYTHON)" = "python-not-found" ]; then \
		echo "ERROR: Python not found. Please install python3"; \
		exit 1; \
	fi
	$(PYTHON) setup.py
	@if [ "$(DETECTED_PLATFORM)" = "tegra" ]; then \
		echo "Bootstrapping Tegra container via open.sh..."; \
		cd eval/tegra && ./open.sh; \
	fi

setup-server:
	$(PYTHON) setup.py --platform server

setup-tegra:
	$(PYTHON) setup.py --platform tegra
	cd eval/tegra && ./open.sh

info:
	$(PYTHON) setup.py --info-only

# === TEGRA EVALUATION TARGETS ===
jetson:
	@if [ "$(DETECTED_PLATFORM)" != "tegra" ]; then \
		echo "ERROR: Tegra platform required but detected: $(DETECTED_PLATFORM)"; \
		echo "TIP: Try make server-* targets instead"; \
		exit 1; \
	fi
	python runners/run_bench.py --cfg configs/jetson.yaml --outdir results/jetson

# Tegra MMLU evaluations
tegra-base:
	@if [ "$(DETECTED_PLATFORM)" != "tegra" ]; then \
		echo "ERROR: Tegra platform required but detected: $(DETECTED_PLATFORM)"; \
		echo "TIP: Try make server-mmlu instead"; \
		exit 1; \
	fi
	cd eval/tegra/mmlu && ./launch.sh base

tegra-budget: 
	@if [ "$(DETECTED_PLATFORM)" != "tegra" ]; then \
		echo "ERROR: Tegra platform required"; \
		exit 1; \
	fi
	cd eval/tegra/mmlu && ./launch.sh budget

tegra-scaling:
	@if [ "$(DETECTED_PLATFORM)" != "tegra" ]; then \
		echo "ERROR: Tegra platform required"; \
		exit 1; \
	fi
	cd eval/tegra/mmlu && ./launch.sh scaling

tegra-synthetic:
	@if [ "$(DETECTED_PLATFORM)" != "tegra" ]; then \
		echo "ERROR: Tegra platform required"; \
		exit 1; \
	fi
	cd eval/tegra/mmlu && ./launch.sh synthetic

tegra-all:
	@if [ "$(DETECTED_PLATFORM)" != "tegra" ]; then \
		echo "ERROR: Tegra platform required"; \
		exit 1; \
	fi
	cd eval/tegra/mmlu && ./launch.sh all

prefill:
	@if [ "$(DETECTED_PLATFORM)" != "tegra" ]; then \
		echo "ERROR: Tegra platform required for synthetic benchmarks"; \
		exit 1; \
	fi
	@CID=$$(docker ps --filter "ancestor=dustynv/vllm:0.8.6-r36.4-cu128-24.04" --format "{{.ID}}" | head -n1); \
	if [ -z "$$CID" ]; then \
		echo "ERROR: Tegra container not running (image dustynv/vllm:0.8.6-r36.4-cu128-24.04)"; \
		echo "TIP: Run 'make setup' (which starts the container) or 'cd eval/tegra && ./open.sh 1'"; \
		exit 1; \
	fi; \
	docker exec -i "$$CID" bash -lc 'cd /workspace/edgereasoning/eval/tegra/mmlu && ./launch.sh prefill'

decode:
	@if [ "$(DETECTED_PLATFORM)" != "tegra" ]; then \
		echo "ERROR: Tegra platform required for synthetic benchmarks"; \
		exit 1; \
	fi
	@CID=$$(docker ps --filter "ancestor=dustynv/vllm:0.8.6-r36.4-cu128-24.04" --format "{{.ID}}" | head -n1); \
	if [ -z "$$CID" ]; then \
		echo "ERROR: Tegra container not running (image dustynv/vllm:0.8.6-r36.4-cu128-24.04)"; \
		echo "TIP: Run 'make setup' (which starts the container) or 'cd eval/tegra && ./open.sh 1'"; \
		exit 1; \
	fi; \
	docker exec -i "$$CID" bash -lc 'cd /workspace/edgereasoning/eval/tegra/mmlu && ./launch.sh decode'

tt-scaling:
	@if [ "$(DETECTED_PLATFORM)" != "tegra" ]; then \
		echo "ERROR: Tegra platform required for synthetic benchmarks"; \
		exit 1; \
	fi
	@CID=$$(docker ps --filter "ancestor=dustynv/vllm:0.8.6-r36.4-cu128-24.04" --format "{{.ID}}" | head -n1); \
	if [ -z "$$CID" ]; then \
		echo "ERROR: Tegra container not running (image dustynv/vllm:0.8.6-r36.4-cu128-24.04)"; \
		echo "TIP: Run 'make setup' (which starts the container) or 'cd eval/tegra && ./open.sh 1'"; \
		exit 1; \
	fi; \
	docker exec -i "$$CID" bash -lc 'cd /workspace/edgereasoning/eval/tegra/mmlu && ./launch.sh tt_scaling'

synthetic: prefill decode tt-scaling

# === SERVER EVALUATION TARGETS ===
server: 
	@if [ "$(DETECTED_PLATFORM)" != "server" ]; then \
		echo "ERROR: Server platform required but detected: $(DETECTED_PLATFORM)"; \
		echo "TIP: Try make tegra-* targets instead"; \
		exit 1; \
	fi
	conda env update -f envs/server/conda.yml
	python runners/run_bench.py --cfg configs/server.yaml --outdir results/server

server-mmlu:
	@if [ "$(DETECTED_PLATFORM)" != "server" ]; then \
		echo "ERROR: Server platform required but detected: $(DETECTED_PLATFORM)"; \
		echo "TIP: Try make tegra-base instead"; \
		exit 1; \
	fi
	cd eval/server/mmlu && ./run.sh base

planner:
	@if [ "$(DETECTED_PLATFORM)" != "server" ]; then \
		echo "ERROR: Planner requires server platform but detected: $(DETECTED_PLATFORM)"; \
		exit 1; \
	fi
	cd eval/server/planner && ./run.sh base

# === ANALYSIS TARGETS ===
collect:
	$(PYTHON) runners/collect_env.py --output data/env.json

validate:
	$(PYTHON) validate_analyticals.py

platform:
	@echo "Detected platform: $(DETECTED_PLATFORM)"

env-info:
	$(PYTHON) runners/collect_env.py

venv:
	@if [ -d ".venv" ]; then \
		echo "Virtual environment already exists!"; \
		echo "To recreate it: rm -rf .venv && make venv"; \
		echo "To activate it: source .venv/bin/activate"; \
	else \
		echo "Creating Python virtual environment..."; \
		python3 -m venv .venv; \
		echo ""; \
		echo "Virtual environment created!"; \
		echo "Activate it with: source .venv/bin/activate"; \
		echo "Then run: make setup"; \
	fi