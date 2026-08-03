# Tech Challenge Fase 2 - Sistema Preditivo de Propensão de Compra

Este projeto contém a solução para o Tech Challenge da Fase 2, focada em Engenharia de Machine Learning.

## 🚀 Tecnologias Utilizadas
- **Python 3.12+**
- **uv**: Gerenciador de dependências ultrarrápido (substituto do Poetry).
- **scikit-learn**: Para a modelagem em si (Random Forest).
- **DVC**: Data Version Control para versionamento de dados e pipeline.
- **MLflow**: Rastreamento de experimentos (Tracking) e Model Registry.
- **Docker**: Containerização do ambiente completo.

## 📂 Estrutura do Projeto
O projeto foi organizado visando Clean Code e boas práticas:
- `src/`: Código-fonte (`pipeline.py` para processamento e `modelo.py` para treinamento).
- `data/`: Datasets (gerenciados pelo DVC).
- `models/`: Modelos salvos (se aplicável localmente, mas priorizamos MLflow Registry).
- `configs/`: Configurações extras.
- `tests/`: Scripts de testes básicos (pytest).

## ⚙️ Como Executar Localmente

### 1. Pré-requisitos
Instale o `uv` no seu sistema:
```bash
pip install uv
```

### 2. Instalação das dependências
Com o `uv` instalado, rode o comando abaixo na raiz do projeto para criar o ambiente e instalar tudo:
```bash
uv sync
```

### 3. Executando o Pipeline (DVC)
Todo o fluxo de preprocessamento e treino foi definido em `dvc.yaml`. Para executá-lo:
```bash
uv run dvc repro
```
Isso irá gerar os dados preprocessados e treinar o modelo Random Forest, logando tudo automaticamente no MLflow.

### 4. Acessando o MLflow UI
Para visualizar os experimentos e o Model Registry, rode o MLflow localmente:
```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Em seguida, acesse `http://127.0.0.1:5000` no seu navegador.

## 🐳 Como Executar via Docker (Produção / OCI)
Para executar o pipeline completo e subir os serviços web (MLflow UI e o Web App de Predição), utilize o Docker Compose:
```bash
docker compose up --build -d
```
Isso irá inicializar 3 serviços:
1. `mlet_pipeline`: Vai rodar o `dvc repro` para garantir que o modelo existe.
2. `mlet_mlflow_ui`: Estará disponível na porta **5000**.
3. `mlet_streamlit_app`: O app de demonstração estará disponível na porta **8501**.

Você pode usar essas portas para configurar seu túnel do Cloudflare Zero Trust e acessar tudo publicamente.

## ✅ Testes e Linting
Para rodar a verificação de Clean Code (Ruff) e testes unitários:
```bash
uv run ruff check .
uv run pytest
```
