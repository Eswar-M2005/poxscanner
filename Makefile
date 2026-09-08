.DEFAULT_GOAL := help
.PHONY: help install-api install-ml install-web audit train evaluate api web test compose-up

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"; printf "Pox Scanner targets:\n"} /^[a-zA-Z0-9_-]+:.*?##/ { printf "  %-16s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install-api: ## Install API Python deps (server/requirements.txt or extras)
	python -m pip install -U pip
	@if [ -f server/requirements.txt ]; then \
		python -m pip install -r server/requirements.txt; \
	else \
		python -m pip install -e ".[server]"; \
	fi

install-ml: ## Install ML extras with CPU PyTorch wheels
	python -m pip install -U pip
	python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	python -m pip install -e ".[ml]"

install-web: ## Install Next.js client deps
	cd client && npm install

audit: ## Audit images under data/raw
	python -m ml.src.audit

train: ## Train default backbone (refuses if splits are empty)
	python -m ml.src.train --model efficientnet_b0

evaluate: ## Evaluate the default checkpoint on the test split
	python -m ml.src.evaluate --checkpoint ml/artifacts/checkpoints/efficientnet_b0_best.pt --split test

api: ## Run FastAPI with reload on PORT (default 8000)
	cd server && PYTHONPATH=..:. uvicorn app.main:app --reload --host 0.0.0.0 --port $${PORT:-8000}

web: ## Run the Next.js dev server
	cd client && npm run dev

test: ## Run pytest and client lint if present
	./scripts/test.sh

compose-up: ## Build and start api + web via Docker Compose
	@test -f .env || cp .env.example .env
	docker compose up --build
