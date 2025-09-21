import streamlit as st
import pandas as pd
from common_styles import configure_page, generarMenu

# Configuración de la página con estilos unificados
configure_page("Carga de archivos", "📁")

def main():
    generarMenu()

    st.title("Carga de datos")
    st.write("Sube los archivos Excel que se utilizarán en las páginas de segmentación.")

    carozos_file = st.file_uploader("Archivo de carozos", type=["xls", "xlsx"], key="upload_carozos")
    if carozos_file is not None:
        try:
            # Importar la función mejorada de carga
            from segmentacion_base import load_excel_with_headers_detection
            
            df_carozos = load_excel_with_headers_detection(carozos_file, "CAROZOS", "A:AP")
            st.session_state["carozos_df"] = df_carozos
            st.success("✅ Archivo de carozos cargado correctamente con detección automática de encabezados")
            
            # Mostrar información sobre la carga
            st.info(f"📊 Cargadas {len(df_carozos)} filas y {len(df_carozos.columns)} columnas")
            
            # Mostrar los primeros encabezados detectados
            st.write("**Encabezados detectados:**")
            st.write(", ".join(df_carozos.columns[:10].tolist()) + ("..." if len(df_carozos.columns) > 10 else ""))
            
            st.dataframe(df_carozos.head(), use_container_width=True)
            csv_carozos = df_carozos.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Guardar datos de carozos", csv_carozos, "carozos_cargado.csv", "text/csv")
        except Exception as e:
            st.error(f"❌ Error al leer el archivo de carozos: {e}")
            # Mostrar información adicional para debugging
            st.write("**Información del archivo:**")
            st.write(f"- Nombre: {carozos_file.name}")
            st.write(f"- Tamaño: {carozos_file.size} bytes")

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
