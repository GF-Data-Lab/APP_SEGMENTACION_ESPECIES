
import streamlit as st
from utils import show_logo  # Asegúrate de tener esta función que muestra el logo


st.markdown(
    """
    <style>
      /* Sólo los botones dentro del sidebar */
      [data-testid="stSidebar"] div.stButton > button {
        background-color: #D32F2F !important;  /* rojo fuerte */
        color: white !important;
        border: none !important;
      }
      [data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #B71C1C !important;  /* rojo más oscuro al pasar */
      }
    </style>
    """,
    unsafe_allow_html=True
)
# Función para generar el menú con botones en la barra lateral
def generarMenu():
    with st.sidebar:
        # Mostrar el logo en la barra lateral
        show_logo()

        # Crear los botones debajo del logo en la barra lateral
        boton_inicio = st.button('Página de Inicio 🏚️')
        boton_ciruela = st.button('Segmentación Ciruela 🍑')
        boton_nectarina = st.button('Segmentación Nectarina 🍑')
        boton_cluster = st.button('Modelo de Clasificación')
        boton_analisis = st.button('Análisis exploratorio')
    # Acción de los botones: redirigir a la página correspondiente
    if boton_inicio:
        st.switch_page('app.py')  # Redirige a la página principal
    if boton_ciruela:
        st.switch_page('pages/segmentacion_ciruela.py')
    if boton_nectarina:
        st.switch_page('pages/segmentacion_nectarina.py')
    if boton_cluster:
        st.switch_page('pages/Cluster_especies.py')  
    if boton_analisis:
        st.switch_page('pages/analisis.py')

# Llamar a la función para generar el menú en la barra lateral
generarMenu()

# Título en la página principal
st.title("Bienvenido a la aplicación de Segmentación de Especies")

