import streamlit as st
import pandas as pd
import mlflow.sklearn
import os

st.set_page_config(page_title="Predição de Compra", page_icon="🛒")

st.title("🛒 Predição de Propensão de Compra")
st.markdown("Este aplicativo carrega o modelo treinado diretamente do **MLflow Model Registry** e realiza predições em usuários de teste.")

# Configurar MLflow local
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

@st.cache_resource
def load_model():
    # Tenta carregar a última versão do modelo registrado
    try:
        model_uri = "models:/RF_Propensao_Compra/latest"
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo do MLflow: {e}")
        return None

model = load_model()

if model:
    st.success("Modelo `RF_Propensao_Compra` carregado com sucesso pelo MLflow!")
    
    st.subheader("Testar Modelo")
    st.write("Para demonstrar, vamos sortear um cliente aleatório da base de testes (já preprocessada) e prever se ele finalizará a compra.")
    
    try:
        df_test = pd.read_csv("data/test.csv")
        
        if st.button("Sortear Cliente e Prever"):
            sample = df_test.sample(1)
            
            # Separar features do alvo real
            X_sample = sample.drop(columns=["Revenue"])
            y_real = sample["Revenue"].values[0]
            
            # Mostrar os dados do cliente
            st.write("**Dados do Cliente (Features Preprocessadas):**")
            st.dataframe(X_sample)
            
            # Fazer a predição
            pred = model.predict(X_sample)[0]
            pred_proba = model.predict_proba(X_sample)[0][1]
            
            # Mostrar o resultado
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predição do Modelo", "COMPROU 🛍️" if pred == 1 else "NÃO COMPROU ❌")
                st.metric("Confiança (Probabilidade)", f"{pred_proba * 100:.2f}%")
                
            with col2:
                st.metric("Ação Real do Cliente", "COMPROU 🛍️" if y_real == 1 else "NÃO COMPROU ❌")
                
            if pred == y_real:
                st.success("O modelo acertou a predição!")
            else:
                st.warning("O modelo errou esta predição.")
                
    except FileNotFoundError:
        st.warning("Arquivo `data/test.csv` não encontrado. Execute o pipeline do DVC primeiro.")
else:
    st.info("Execute o pipeline do DVC (`uv run dvc repro`) para treinar e registrar o modelo antes de usar o app.")
