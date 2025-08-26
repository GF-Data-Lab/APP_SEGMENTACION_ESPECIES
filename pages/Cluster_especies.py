# pages/03_modelo_clasificacion_variedad.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import plotly.express as px
import plotly.figure_factory as ff
from utils import show_logo

# --------------------------------------------------
# Configuración de la página
# --------------------------------------------------
st.set_page_config(
    page_title="Modelo de Clasificación",
    page_icon="🧮",
    layout="wide"
)

# --------------------------------------------------
# Estilos del sidebar
# --------------------------------------------------
st.markdown(
    """
    <style>
      /* Sólo los botones dentro del sidebar */
      [data-testid="stSidebar"] div.stButton > button {
        background-color: #D32F2F !important;
        color: white !important;
        border: none !important;
      }
      [data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #B71C1C !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Menú lateral personalizado
# --------------------------------------------------
def generarMenu():
    with st.sidebar:
        show_logo()
        if st.button('Página de Inicio 🏚️'):
            st.switch_page('app.py')
        if st.button('Carga de archivos 📁'):
            st.switch_page('pages/carga_datos.py')
        if st.button('Segmentación Ciruela 🍑'):
            st.switch_page('pages/segmentacion_ciruela.py')
        if st.button('Segmentación Nectarina 🍑'):
            st.switch_page('pages/segmentacion_nectarina.py')
        if st.button('Modelo de Clasificación'):
            st.switch_page('pages/Cluster_especies.py')
        if st.button('Análisis exploratorio'):
            st.switch_page('pages/analisis.py')
generarMenu()

st.title("🧪 Modelo de Clasificación de Variedad")
st.markdown(
    """
    En esta página vamos a **entrenar** un clasificador para predecir la *Variedad*
    (por ejemplo, candy vs sugar) usando tus datos procesados de carozos.
    Ajusta hiperparámetros, observa métricas y ¡que empiece la magia de la ML! ✨
    """
)

# —————————————————————————————————————————————
# 1. Recuperar DataFrame ya procesado
# —————————————————————————————————————————————
if "df_seg_especies" not in st.session_state:
    st.error(
        "❗ No encuentro los datos procesados. Primero ve a “Segmentación de especies” y sube tu Excel allí."
    )
    st.stop()

df = st.session_state["df_seg_especies"]
st.success("✅ Datos heredados de Segmentación de especies")
st.dataframe(df.head(), use_container_width=True)

# --------------------------------------------------
# 2. Selección de variables
# --------------------------------------------------
st.subheader("⚙️ Selección de etiqueta y características")

# — Etiqueta (target): columnas de tipo object o categórico
cat_cols = [c for c in df.columns if df[c].dtype == "object"]
default_target = "Variedad" if "Variedad" in cat_cols else cat_cols[0]
target_col = st.selectbox(
    "Etiqueta (target) a predecir",
    options=cat_cols,
    index=cat_cols.index(default_target)
)

# — Features: columnas numéricas
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
features = st.multiselect(
    "Variables predictoras (features)",
    options=numeric_cols,
    default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
)
if len(features) < 2:
    st.error("Selecciona al menos **2** variables predictoras.")
    st.stop()

# --------------------------------------------------
# 3. Partición de datos con fallback si falla stratify
# --------------------------------------------------
st.subheader("📊 División Train / Test")
test_size = st.slider("Tamaño del test (%)", 10, 50, 25, step=5)

X = df[features].fillna(0).values
y = df[target_col].astype(str).values

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size/100,
        random_state=42,
        stratify=y
    )
except ValueError as e:
    st.warning(
        "⚠️ No puedo estratificar porque alguna clase tiene <2 muestras. "
        "Realizaré el split sin estratificación."
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size/100,
        random_state=42
    )

st.write(f"- 👨‍🏫 Entrenamiento: {len(y_train)} muestras")
st.write(f"- 🧪 Test:         {len(y_test)} muestras")

# --------------------------------------------------
# 4. Hiperparámetros del modelo
# --------------------------------------------------
st.subheader("🔧 Hiperparámetros Random Forest")
n_estimators = st.slider("Número de árboles", 10, 200, 50, step=10)
max_depth    = st.slider("Profundidad máxima (None=sin límite)", 1, 20, 5)
if max_depth == 1:
    max_depth = None  # dejamos sin límite si está en 1

# --------------------------------------------------
# 5. Entrenamiento y métricas
# --------------------------------------------------
if st.button("🚀 Entrenar modelo"):
    with st.spinner("Entrenando Random Forest…"):
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    st.success("✅ Modelo entrenado")

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy (test)", f"{acc*100:.2f}%")
    
    # Reporte de clasificación
    st.markdown("**Reporte de Clasificación**")
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).T
    st.dataframe(df_report, use_container_width=True)
    
    # Matriz de confusión
    st.markdown("**Matriz de Confusión**")
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=list(clf.classes_),
        y=list(clf.classes_),
        colorscale="Blues",
        showscale=True
    )
    fig_cm.update_layout(xaxis_title="Predicho", yaxis_title="Verdadero")
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Importancia de features
    st.markdown("**Importancia de Variables**")
    imp = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    fig_imp = px.bar(
        imp, 
        x=imp.values, y=imp.index, orientation="h",
        labels={'x':'Importancia','y':'Variable'}
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.balloons()
else:
    st.info("Ajusta los parámetros y dale a **Entrenar modelo** 😎")
