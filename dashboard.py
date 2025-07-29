import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Título
st.title("📊 Painel Analítico sobre Obesidade")
st.subheader("Análise exploratória dos fatores que influenciam a obesidade")

# Carregando os dados
df = pd.read_csv('Obesity.csv')
st.success("✅ Dados carregados com sucesso!")

# Mostrar primeiras linhas
st.write("### 📌 Pré-visualização dos dados:")
st.dataframe(df.head())

# Distribuição da variável alvo
st.write("### 🧮 Distribuição de obesidade:")
fig1 = plt.figure(figsize=(8,4))
sns.countplot(x='Obesity', data=df)
plt.xticks(rotation=45)
st.pyplot(fig1)

# Obesidade por gênero
st.write("### ⚖️ Obesidade por gênero:")
fig2 = plt.figure(figsize=(8,4))
sns.countplot(x='Gender', hue='Obesity', data=df)
plt.xticks(rotation=0)
st.pyplot(fig2)

# Obesidade por idade
st.write("### 🎂 Distribuição por idade:")
fig3 = plt.figure(figsize=(8,4))
sns.histplot(df['Age'], kde=True)
plt.xlabel('Idade')
st.pyplot(fig3)

# Obesidade vs atividade física
st.write("### 🏃‍♂️ Relação entre obesidade e atividade física:")
fig4 = plt.figure(figsize=(8,4))
sns.boxplot(x='Obesity', y='FAF', data=df)
plt.xlabel('Obesidade')
plt.ylabel('Horas de atividade física por dia')
st.pyplot(fig4)

# Correlação entre variáveis numéricas
st.write("### 🔍 Mapa de correlação entre variáveis numéricas:")
num_cols = df.select_dtypes(include='number').columns
fig5 = plt.figure(figsize=(10,8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
st.pyplot(fig5)

# Observações
st.write("### 💡 Insights e observações:")
st.markdown("""
- Indivíduos com menos atividade física tendem a apresentar maior prevalência de obesidade.
- Há uma correlação moderada entre peso e obesidade, como esperado.
- Diferenças entre gêneros podem indicar abordagens de prevenção específicas.
- O histórico familiar é um fator relevante a ser investigado mais profundamente.
""")
