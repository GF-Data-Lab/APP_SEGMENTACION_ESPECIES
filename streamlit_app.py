import streamlit as st
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

@st.cache_resource
def cargar_modelo():
    data = load_iris()
    X = data.data
    y = data.target
    modelo = Pipeline([
        ("escalador", StandardScaler()),
        ("clasificador", LogisticRegression(max_iter=200))
    ])
    modelo.fit(X, y)
    return modelo, data.target_names

modelo, nombres = cargar_modelo()

st.title("Clasificación de especies (Iris)")

sepal_length = st.number_input(
    "Longitud del sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1
)
sepal_width = st.number_input(
    "Anchura del sépalo (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1
)
petal_length = st.number_input(
    "Longitud del pétalo (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1
)
petal_width = st.number_input(
    "Anchura del pétalo (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1
)

if st.button("Predecir especie"):
    datos = [[sepal_length, sepal_width, petal_length, petal_width]]
    pred = modelo.predict(datos)[0]
    st.success(f"La especie predicha es: {nombres[pred]}")

