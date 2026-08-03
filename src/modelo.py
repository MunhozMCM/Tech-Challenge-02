import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import os

def treinar(train_path: str, test_path: str) -> None:
    """Treina um modelo RandomForest e registra no MLflow com MLflow Registry."""
    print("Iniciando treinamento...")
    
    # Carregar dados
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["Revenue"])
    y_train = train_df["Revenue"]

    X_test = test_df.drop(columns=["Revenue"])
    y_test = test_df["Revenue"]

    # Configurar MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("Propensao_Compra")

    with mlflow.start_run():
        # Hiperparâmetros
        n_estimators = 100
        max_depth = 5
        random_state = 42

        # Log hiperparâmetros
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        # Treinar modelo
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        clf.fit(X_train, y_train)

        # Previsões
        preds = clf.predict(X_test)

        # Métricas
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        # Log métricas
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        print(f"Modelo treinado. Acurácia: {acc:.4f}, F1-Score: {f1:.4f}")

        # Log modelo
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="random_forest_model",
            registered_model_name="RF_Propensao_Compra"
        )
        print("Modelo registrado no MLflow com sucesso!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    args = parser.parse_args()
    treinar(args.train, args.test)
