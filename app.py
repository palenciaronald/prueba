import streamlit as st
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ----------------------------
# Cargar modelos y metadatos
# ----------------------------
modelo = joblib.load("modelo.pkl")
columnas = joblib.load("columnas.pkl")
categorias = ["Riesgo Bajo", "Riesgo Medio", "Riesgo Alto"]
cat_vars = joblib.load("cat_vars.pkl")

kmeans = joblib.load("kmeans.pkl")
scaler = joblib.load("scaler.pkl")
prob_cols = joblib.load("prob_cols.pkl")

# ----------------------------
# Título
# ----------------------------
st.title("📊 Calificación Crediticia + Cluster de Riesgo")

# ----------------------------
# Entradas del cliente
# ----------------------------
st.header("🧾 Datos del Cliente")

entrada = {}
###nj

# Inputs numéricos
entrada['Numero Creditos Vigentes'] = st.number_input("Número Créditos Vigentes", value=0)
entrada['Monto Desembolsado'] = st.number_input("Monto Desembolsado", value=0.0)
entrada['Numero Creditos Cerrados'] = st.number_input("Número Créditos Cerrados", value=0)
entrada['Ventas Negocio'] = st.number_input("Ventas del Negocio", value=0.0)
entrada['Maximo Dias Mora'] = st.number_input("Máximo Días en Mora", value=0)
entrada['Experiencia Bancaria'] = st.number_input("Experiencia Bancaria (años)", value=0)
entrada['Ingresos Operativos Negocio'] = st.number_input("Ingresos Operativos del Negocio", value=0.0)
entrada['Antiguedad Negocio'] = st.number_input("Antigüedad del Negocio (años)", value=0)
entrada['Ingresos Mensuales'] = st.number_input("Ingresos Mensuales", value=0.0)
entrada['Gastos Negocio'] = st.number_input("Gastos del Negocio", value=0.0)

# Inputs categóricos
entrada['Zona Geografica'] = st.selectbox("Zona Geográfica", ['Norte', 'Sur', 'Centro', 'Este', 'Oeste'])
entrada['Zona Comercial'] = st.selectbox("Zona Comercial", ['Alta', 'Media', 'Baja'])

entrada_df = pd.DataFrame([entrada])

# ----------------------------
# Procesamiento para predicción
# ----------------------------
entrada_cat = pd.get_dummies(entrada_df[cat_vars], drop_first=True)
entrada_num = entrada_df.drop(columns=cat_vars)
entrada_final = pd.concat([entrada_num, entrada_cat], axis=1)

# Asegurar columnas correctas
for col in columnas:
    if col not in entrada_final.columns:
        entrada_final[col] = 0

entrada_final = entrada_final[columnas]
entrada_final = sm.add_constant(entrada_final, has_constant='add')

# ----------------------------
# Botón de acción
# ----------------------------
if st.button("📈 Predecir Calificación y Cluster"):
    # --- Paso 1: Calificación
    probas = modelo.predict(entrada_final)
    clase_predicha = np.argmax(probas.values)
    nombre_clase = categorias[clase_predicha]

    st.subheader("📋 Calificación Estimada")
    st.success(f"Este cliente tiene una calificación de **{nombre_clase}**")

    # --- Paso 2: Mostrar probabilidades
    st.subheader("📊 Probabilidades por nivel de riesgo")
    fig, ax = plt.subplots()
    barras = ax.bar(categorias, probas.values[0], color='skyblue')
    ax.set_ylabel("Probabilidad")
    ax.set_ylim(0, 1.05)
    ax.set_title("Distribución de Probabilidades")

    for i, bar in enumerate(barras):
        ax.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.02,
                f"{probas.values[0][i]:.2f}", ha='center', va='bottom', fontsize=10)
    st.pyplot(fig)

    # --- Paso 3: Clustering sobre el perfil de riesgo
    perfil_riesgo_df = pd.DataFrame(probas.values, columns=prob_cols)
    perfil_riesgo_scaled = scaler.transform(perfil_riesgo_df)
    cluster = kmeans.predict(perfil_riesgo_scaled)[0]

    st.subheader("🧭 Cluster de Riesgo")
    st.info(f"Este cliente pertenece al **Cluster {cluster}** de riesgo.")

    # Mostrar interpretación del cluster
    st.subheader("📌 Características del Cluster")

    explicacion_cluster = {
        0: """
    🟢 **Cluster 0 – Perfil de Bajo Riesgo**  
    - Alta probabilidad de riesgo bajo (75%)  
    - Predomina la calificación “Alto”  
    - Montos desembolsados mayores al promedio  
    - Perfil ideal para ofrecer productos financieros con mejores condiciones o expansión de líneas de crédito  
    📢 **Recomendación:** Priorizar este grupo en campañas comerciales y programas de fidelización.
        """,
        1: """
    🔴 **Cluster 1 – Perfil de Riesgo Alto**  
    - Probabilidad del 75% de riesgo alto  
    - Todos tienen calificación “Medio”, pero el modelo los asocia a mayor riesgo  
    - Perfil financiero similar al resto, lo cual sugiere que hay factores no observados (como comportamiento histórico o variables cualitativas) que elevan su riesgo  
    📢 **Recomendación:** Requieren análisis adicional, garantías adicionales o esquemas de monitoreo intensivo.
        """,
        2: """
    🟡 **Cluster 2 – Perfil de Riesgo Medio**  
    - Riesgo medio con alta probabilidad (76%)  
    - Calificación original “Bajo” en muchos casos, lo cual podría indicar riesgo subestimado  
    - Existe un valor atípico en los créditos cerrados que puede afectar el perfil  
    📢 **Recomendación:** Grupo clave para intervenciones preventivas: educación financiera, revisión de condiciones o ajustes en scoring.
        """,
        3: """
    🔵 **Cluster 3 – Perfil Mixto**  
    - Probabilidades casi uniformes entre riesgo bajo, medio y alto (~33%)  
    - Alta heterogeneidad en calificación crediticia  
    - Monto desembolsado ligeramente menor  
    📢 **Recomendación:** Aplicar análisis más profundo o mejorar la segmentación incluyendo nuevas variables.
        """
    }

    st.markdown(explicacion_cluster[cluster])

