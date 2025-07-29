import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

st.title('🔍 Preditor de Obesidade')
st.subheader('Previsão de risco de obesidade baseado em dados clínicos e comportamentais')

# Carrega modelo
model = joblib.load('modelo_obesidade.pkl')
scaler = StandardScaler()

# Inputs
st.sidebar.header('📋 Informe os dados do paciente:')
gender = st.sidebar.selectbox('Gênero', ['Male', 'Female'])
age = st.sidebar.slider('Idade', 10, 100, 25)
height = st.sidebar.number_input('Altura (m)', min_value=1.0, max_value=2.5, value=1.70)
weight = st.sidebar.number_input('Peso (kg)', min_value=30.0, max_value=200.0, value=70.0)
family_history = st.sidebar.selectbox('Histórico familiar de obesidade', ['yes', 'no'])
calorie_consumption = st.sidebar.slider('Calorias diárias', 500, 5000, 2500)
physical_activity = st.sidebar.slider('Atividade física (horas/dia)', 0.0, 5.0, 1.0)

input_dict = {
    'Gender': [gender],
    'Age': [age],
    'Height': [height],
    'Weight': [weight],
    'family_history_with_overweight': [family_history],
    'CALC': [calorie_consumption],
    'FAF': [physical_activity]
}

input_df = pd.DataFrame(input_dict)
input_encoded = pd.get_dummies(input_df)
expected_cols = model.feature_names_in_
for col in expected_cols:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[expected_cols]
X_scaled = scaler.fit_transform(input_encoded)

prediction = model.predict(X_scaled)[0]
st.markdown(f"## 🧠 Resultado da predição: **{prediction}**")
st.write("Use este resultado como apoio à decisão médica.")
