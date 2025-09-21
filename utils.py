"""
Módulo de utilidades para la aplicación de segmentación de carozos.

Este módulo contiene funciones auxiliares utilizadas en toda la aplicación,
principalmente para el manejo de recursos visuales como el logo corporativo.

Funciones principales:
    - encode_image_to_base64: Convierte imágenes a formato base64
    - show_logo: Muestra el logo corporativo en la barra lateral

Autor: Equipo de Análisis de Datos
Fecha: Septiembre 2024
"""

import streamlit as st
import base64


def encode_image_to_base64(file_path: str) -> str:
    """
    Convierte una imagen a formato base64 para embedding en HTML.

    Esta función lee un archivo de imagen desde el sistema de archivos
    y lo convierte a una cadena base64 que puede ser embebida directamente
    en elementos HTML, evitando problemas de rutas relativas.

    Args:
        file_path (str): Ruta al archivo de imagen a convertir.
                        Formatos soportados: PNG, JPG, JPEG, GIF

    Returns:
        str: Cadena base64 de la imagen, vacía si hay error

    Raises:
        Exception: Si no se puede leer el archivo (archivo no existe,
                  permisos insuficientes, formato no válido)

    Example:
        >>> base64_img = encode_image_to_base64("logo.png")
        >>> if base64_img:
        ...     html = f'<img src="data:image/png;base64,{base64_img}">'
    """
    try:
        # Abrir archivo en modo binario para lectura
        with open(file_path, "rb") as f:
            data = f.read()

        # Convertir bytes a base64 y decodificar a string
        return base64.b64encode(data).decode()

    except FileNotFoundError:
        st.error(f"❌ Error: No se encontró el archivo de imagen: {file_path}")
        return ""
    except PermissionError:
        st.error(f"❌ Error: Sin permisos para leer el archivo: {file_path}")
        return ""
    except Exception as e:
        st.error(f"❌ Error inesperado al cargar la imagen: {e}")
        return ""


def show_logo() -> None:
    """
    Muestra el logo corporativo en la barra lateral de Streamlit.

    Esta función muestra el logo de la empresa. Si existe el archivo
    'garces_data_analytics.png' lo muestra, sino muestra un placeholder elegante.

    Requirements:
        - Streamlit debe estar configurado para permitir HTML no seguro

    Example:
        >>> # En cualquier página de Streamlit
        >>> show_logo()  # Muestra el logo en la sidebar
    """
    # Intentar cargar y convertir el logo a base64
    img_base64 = encode_image_to_base64("garces_data_analytics.png")

    # Solo mostrar si la conversión fue exitosa
    if img_base64:
        st.sidebar.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 1rem;">
                <img src="data:image/png;base64,{img_base64}"
                     alt="Logo Garces Data Analytics"
                     style="height: 160px; max-width: 100%; object-fit: contain;">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Mostrar logo elegante con CSS cuando no hay imagen
        st.sidebar.markdown(
            """
            <div style="text-align: center; margin-bottom: 1.5rem;
                        padding: 25px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 15px; color: white;
                        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);">
                <div style="font-size: 2.2em; margin-bottom: 8px;">📊</div>
                <h3 style="margin: 0; font-weight: 600; font-size: 1.1em;">
                    Garces Data Analytics
                </h3>
                <p style="margin: 5px 0 0 0; font-size: 0.8em; opacity: 0.9;">
                    Segmentación de Carozos
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )