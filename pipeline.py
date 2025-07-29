
# pipeline.py

# 1. Importações
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def treinar_pipeline(caminho_csv='Obesity.csv'):
    print("🚀 Iniciando treinamento da pipeline...")

    # 2. Carregar os dados
    df = pd.read_csv(caminho_csv)
    print("✅ Dados carregados!")

    # 3. Visualização da variável alvo
    plt.figure(figsize=(8,5))
    sns.countplot(x='Obesity', data=df)
    plt.title('Distribuição da Obesidade')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('grafico_obesidade.png')
    print("📊 Gráfico de distribuição salvo como grafico_obesidade.png")

    # 4. Feature Engineering
    target = df['Obesity']
    features = df.drop(columns=['Obesity'])
    features_encoded = pd.get_dummies(features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_encoded)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.3, random_state=42)

    # 5. Treinamento
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("🧠 Modelo treinado!")

    # 6. Avaliação
    y_pred = model.predict(X_test)
    print("\n✅ Relatório de classificação:")
    print(classification_report(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Acurácia: {acc:.2f}")

    # 7. Matriz de confusão
    plt.figure(figsize=(6,5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig('matriz_confusao.png')
    print("📌 Matriz de confusão salva como matriz_confusao.png")

    # 8. Salvamento do modelo
    joblib.dump(model, 'modelo_obesidade.pkl')
    print("💾 Modelo salvo como modelo_obesidade.pkl")

if __name__ == '__main__':
    treinar_pipeline()
