# ============================================================
#  Tech Challenge Fase 2 — atalhos de desenvolvimento
# ============================================================

.PHONY: help install repro test lint format mlflow-ui app

help:            ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-12s %s\n", $$1, $$2}'

install:         ## Cria o ambiente e instala as dependências (uv)
	uv sync

repro:           ## Executa o pipeline completo (preprocess → train) via DVC
	uv run dvc repro

test:            ## Roda a suíte de testes (pytest)
	uv run pytest

lint:            ## Verifica lint e formatação (ruff)
	uv run ruff check .
	uv run ruff format --check .

format:          ## Corrige lint e formatação (ruff)
	uv run ruff check . --fix
	uv run ruff format .

mlflow-ui:       ## Sobe a UI do MLflow em http://127.0.0.1:5000
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

app:             ## Sobe o app Streamlit de demonstração
	uv run streamlit run src/app.py
