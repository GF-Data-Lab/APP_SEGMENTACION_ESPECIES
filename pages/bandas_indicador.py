import yaml
from pathlib import Path
import streamlit as st
from utils import show_logo

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
    path = Path("bandas.yaml")
    if "bandas" not in st.session_state:
        if path.exists():
            st.session_state["bandas"] = yaml.safe_load(path.read_text()) or {}
        else:
            st.session_state["bandas"] = {}
    data = st.session_state["bandas"]
    text = st.text_area(
        "Editar bandas (YAML)",
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        height=300,
    )
    if st.button("Guardar"):
        try:
            parsed = yaml.safe_load(text) or {}
            st.session_state["bandas"] = parsed
            path.write_text(yaml.safe_dump(parsed, allow_unicode=True, sort_keys=False))
            st.success("Bandas guardadas en bandas.yaml")
        except yaml.YAMLError as e:
            st.error(f"YAML inválido: {e}")


if __name__ == "__main__":
    main()
