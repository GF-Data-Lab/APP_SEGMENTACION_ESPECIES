import streamlit as st
import pandas as pd
from utils import show_logo

# Configuración de la página
def configurar_pagina():
    st.set_page_config(page_title="Carga de archivos", page_icon="📁", layout="wide")
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
    configurar_pagina()
    generar_menu()

    st.title("Carga de datos")
    st.write("Sube los archivos Excel que se utilizarán en las diferentes páginas de análisis.")

    carozos_file = st.file_uploader("Archivo de carozos", type=["xls", "xlsx"], key="upload_carozos")
    if carozos_file is not None:
        try:
            df_carozos = pd.read_excel(carozos_file, sheet_name="CAROZOS", usecols="A:AP", skiprows=2, dtype=str)
            st.session_state["carozos_df"] = df_carozos
            st.success("Archivo de carozos cargado correctamente")
            st.dataframe(df_carozos.head(), use_container_width=True)
            csv_carozos = df_carozos.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Guardar datos de carozos", csv_carozos, "carozos_cargado.csv", "text/csv")
        except Exception as e:
            st.error(f"Error al leer el archivo de carozos: {e}")

    cerezas_file = st.file_uploader("Archivo de cerezas", type=["xls", "xlsx"], key="upload_cerezas")
    if cerezas_file is not None:
        try:
            df_cerezas = pd.read_excel(cerezas_file, dtype=str)
            st.session_state["cerezas_df"] = df_cerezas
            st.success("Archivo de cerezas cargado correctamente")
            st.dataframe(df_cerezas.head(), use_container_width=True)
            csv_cerezas = df_cerezas.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Guardar datos de cerezas", csv_cerezas, "cerezas_cargado.csv", "text/csv")
        except Exception as e:
            st.error(f"Error al leer el archivo de cerezas: {e}")


if __name__ == "__main__":
    main()
