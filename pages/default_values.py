import json
from pathlib import Path
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
    path = Path("defaults.json")
    if "default_values" not in st.session_state:
        if path.exists():
            st.session_state["default_values"] = json.loads(path.read_text())
        else:
            st.session_state["default_values"] = {}
    data = st.session_state["default_values"]
    text = st.text_area(
        "Editar valores (JSON)",
        json.dumps(data, indent=2, ensure_ascii=False),
        height=300,
    )
    if st.button("Guardar"):
        try:
            parsed = json.loads(text)
            st.session_state["default_values"] = parsed
            path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False))
            st.success("Valores guardados en defaults.json")
        except json.JSONDecodeError as e:
            st.error(f"JSON inválido: {e}")


if __name__ == "__main__":
    main()
