FROM python:3.12-slim

# Instalar uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Configurar diretorio
WORKDIR /app

# Copiar arquivos de configuracao do uv
COPY pyproject.toml uv.lock ./

# Instalar dependencias em cache
ENV UV_PROJECT_ENVIRONMENT="/opt/.venv"
RUN uv sync --frozen --no-dev

# Copiar projeto inteiro
COPY . .

# Expor porta pro MLflow UI se necessario (opcional)
EXPOSE 5000

# Executar pipeline DVC por padrao
CMD ["uv", "run", "dvc", "repro"]
