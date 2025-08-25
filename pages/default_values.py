import streamlit as st
from utils import show_logo


st.set_page_config(page_title="Valores por defecto", page_icon="⚙️", layout="wide")

st.markdown(
    """
    <style>
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
    unsafe_allow_html=True,
)


def generar_menu():
    with st.sidebar:
        show_logo()
        if st.button('Página de Inicio 🏚️'):
            st.switch_page('app.py')
        if st.button('Carga de archivos 📁'):
            st.switch_page('pages/carga_datos.py')
        if st.button('Valores por defecto ⚙️'):
            st.switch_page('pages/default_values.py')
        if st.button('Bandas por indicador 🎯'):
            st.switch_page('pages/bandas_indicador.py')
        if st.button('Modelo y clustering 🧠'):
            st.switch_page('pages/modelo_cluster.py')
        if st.button('Análisis exploratorio 🔍'):
            st.switch_page('pages/analisis.py')


def main():
    generar_menu()
    st.title("Valores por defecto")

    # Inicializar valores en la sesión
    if "default_plum_subtype" not in st.session_state:
        st.session_state["default_plum_subtype"] = "sugar"
    if "sugar_upper" not in st.session_state:
        st.session_state["sugar_upper"] = 60.0
    if "default_color" not in st.session_state:
        st.session_state["default_color"] = "Amarilla"
    if "default_period" not in st.session_state:
        st.session_state["default_period"] = "tardia"

    st.header("Ciruela")
    st.selectbox(
        "Tipo de ciruela por defecto si el peso no está disponible",
        options=["sugar", "candy"],
        index=["sugar", "candy"].index(st.session_state["default_plum_subtype"]),
        key="default_plum_subtype",
    )
    st.number_input(
        "Peso máximo para sugar (g)",
        min_value=10.0,
        max_value=200.0,
        value=float(st.session_state["sugar_upper"]),
        step=1.0,
        key="sugar_upper",
    )

    st.header("Nectarina")
    st.selectbox(
        "Color de pulpa por defecto (si falta)",
        options=["Amarilla", "Blanca"],
        index=["Amarilla", "Blanca"].index(st.session_state["default_color"]),
        key="default_color",
    )
    st.selectbox(
        "Periodo de cosecha por defecto (si falta fecha)",
        options=["muy_temprana", "temprana", "tardia", "sin_fecha"],
        index=["muy_temprana", "temprana", "tardia", "sin_fecha"].index(st.session_state["default_period"]),
        key="default_period",
    )

    st.success("Los valores se guardan automáticamente en la sesión.")


if __name__ == "__main__":
    main()

