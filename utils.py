import streamlit as st
import base64

# Función para cargar la imagen y convertirla a base64
def encode_image_to_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        st.error(f"Error al cargar la imagen: {e}")
        return ""

# Función para mostrar el logo en la sidebar
def show_logo():
    img_base64 = encode_image_to_base64("garces_data_analytics.png")  # Asegúrate de que el logo esté en el directorio correcto
    if img_base64:
        st.sidebar.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 1rem;">
                <img src="data:image/png;base64,{img_base64}" 
                     alt="Logo" style="height:160px;">
            </div>
            """,
            unsafe_allow_html=True
        )