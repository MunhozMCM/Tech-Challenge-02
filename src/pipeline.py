import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def executar_preprocessamento(input_path: str, output_train: str, output_test: str) -> None:
    """Carrega, limpa, codifica e divide o dataset de Shoppers Intention."""
    print(f"Carregando dados de {input_path}...")
    df = pd.read_csv(input_path)

    # Separar Features e Target
    X = df.drop(columns=["Revenue"])
    y = df["Revenue"]

    # Codificar variáveis categóricas
    cat_cols = X.select_dtypes(include=['object', 'bool']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Codificar target
    y = LabelEncoder().fit_transform(y)

    # Dividir em Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalonamento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Salvar em arquivos
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df["Revenue"] = y_train

    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df["Revenue"] = y_test

    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)
    print(f"Dados salvos em {output_train} e {output_test}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-train", required=True)
    parser.add_argument("--output-test", required=True)
    args = parser.parse_args()
    executar_preprocessamento(args.input, args.output_train, args.output_test)
