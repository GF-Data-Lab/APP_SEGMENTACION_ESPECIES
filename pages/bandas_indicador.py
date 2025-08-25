import streamlit as st
from utils import show_logo
from segmentacion_base import (
    DEFAULT_PLUM_RULES,
    DEFAULT_NECT_RULES,
    plum_rules_to_df,
    nect_rules_to_df,
)


st.set_page_config(page_title="Bandas por indicador", page_icon="🎯", layout="wide")

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
    st.title("Bandas por indicador")

    # Inicializar dataframes de reglas
    if "plum_rules_df" not in st.session_state:
        st.session_state["plum_rules_df"] = plum_rules_to_df(DEFAULT_PLUM_RULES)
    if "nect_rules_df" not in st.session_state:
        st.session_state["nect_rules_df"] = nect_rules_to_df(DEFAULT_NECT_RULES)

    especie = st.radio("Especie", ["Ciruela", "Nectarina"], horizontal=True)
    if especie == "Ciruela":
        st.session_state["plum_rules_df"] = st.data_editor(
            st.session_state["plum_rules_df"],
            num_rows="dynamic",
            key="plum_rules_editor",
        )
    else:
        st.session_state["nect_rules_df"] = st.data_editor(
            st.session_state["nect_rules_df"],
            num_rows="dynamic",
            key="nect_rules_editor",
        )

    st.info("Los cambios se guardan automáticamente en la sesión.")


if __name__ == "__main__":
    main()

