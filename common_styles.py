"""Shared styles and navigation helpers for the Streamlit app."""

from typing import Dict
import unicodedata
import re

COMMON_STYLES = """
<style>
    [data-testid="stSidebar"] div.stButton > button {
        background-color: #D32F2F !important;
        color: white !important;
        border: none !important;
    }
    [data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #B71C1C !important;
    }

    .stContainer {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #D32F2F;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    div[data-testid="column"] .stButton > button {
        background-color: #D32F2F !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: bold !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }

    div[data-testid="column"] .stButton > button:hover {
        background-color: #B71C1C !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(211, 47, 47, 0.3) !important;
    }
</style>
"""




def generarMenu() -> None:
    """Render the shared sidebar navigation with stable unique keys."""
    import streamlit as st
    from utils import show_logo

    menu_items = [
        ("Pagina de Inicio", "app.py"),
        ("Carga de archivos", "pages/carga_datos.py"),
        ("Deteccion Outliers", "pages/outliers.py"),
        ("Segmentacion Ciruela", "pages/segmentacion_ciruela.py"),
        ("Segmentacion Nectarina", "pages/segmentacion_nectarina.py"),
        ("Metricas y Bandas", "pages/metricas_bandas.py"),
        ("Verificar Calculos", "pages/verificar_calculos.py"),
        ("Evolucion Variedad", "pages/evolucion_variedad.py"),
    ]

    def _menu_key(label: str, idx: int) -> str:
        normalized = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode('ascii')
        slug = re.sub(r'[^a-z0-9]+', '_', normalized.lower()).strip('_')
        if not slug:
            slug = f'item_{idx}'
        return f'menu_{slug}_{idx}'

    with st.sidebar:
        show_logo()
        st.markdown('---')
        for idx, (label, target) in enumerate(menu_items):
            if st.button(label, key=_menu_key(label, idx)):
                st.switch_page(target)




def get_cluster_colors() -> Dict[str, Dict[int, str]]:
    """
    Retorna la paleta de colores pastel unificada para toda la aplicación.

    Esta paleta está diseñada para ser:
    - Visualmente agradable con tonos pastel suaves
    - Accesible para personas con daltonismo
    - Consistente en todas las visualizaciones
    - Profesional y clara en contextos científicos

    Returns:
        Dict con colores para background, text, solid y plotly por cluster
    """
    return {
        # Colores de fondo pastel para tablas y containers
        "background": {
            1: "#E8F5E8",  # Verde menta muy suave - Excelente
            2: "#FFF4E6",  # Durazno muy suave - Bueno
            3: "#FFE4E1",  # Rosa salmón muy suave - Regular
            4: "#F0E6FF",  # Lavanda muy suave - Deficiente
        },
        # Colores de texto contrastantes para legibilidad
        "text": {
            1: "#2D5016",  # Verde oscuro
            2: "#8B4000",  # Marrón naranja
            3: "#8B0000",  # Rojo oscuro
            4: "#4B0082",  # Índigo
        },
        # Colores sólidos pastel para gráficos y elementos destacados
        "solid": {
            1: "#90EE90",  # Verde claro pastel
            2: "#FFB347",  # Naranja durazno pastel
            3: "#FFA07A",  # Salmón claro pastel
            4: "#DDA0DD",  # Ciruela pastel
        },
        # Colores para gráficos Plotly (más saturados pero armoniosos)
        "plotly": {
            1: "#7FB069",  # Verde sage
            2: "#FF9F40",  # Naranja cálido
            3: "#FF6B6B",  # Coral
            4: "#A8A8FF",  # Lavanda suave
        },
        # Colores alternativos para hover y estados activos
        "hover": {
            1: "#98FB98",  # Verde pálido
            2: "#FFDAB9",  # Durazno suave
            3: "#FFB6C1",  # Rosa claro
            4: "#E6E6FA",  # Lavanda
        },
        # Nombres descriptivos para tooltips y leyendas
        "names": {
            1: "Excelente",
            2: "Bueno",
            3: "Regular",
            4: "Deficiente",
        },
        # Códigos hex para compatibilidad con sistemas externos
        "hex": {
            1: "#90EE90",
            2: "#FFB347",
            3: "#FFA07A",
            4: "#DDA0DD",
        }
    }


def get_cluster_style_function():
    """
    Retorna una función de estilo para colorear filas de DataFrame por cluster.

    Esta función utiliza la paleta de colores pastel estándar y aplica
    estilos consistentes a las tablas de datos según el cluster asignado.

    Returns:
        function: Función que mapea valores de cluster a estilos CSS
    """
    import pandas as pd

    colors = get_cluster_colors()

    def color_cluster(val):
        """
        Aplica estilo CSS basado en el valor del cluster.

        Args:
            val: Valor del cluster (1-4) o NaN

        Returns:
            str: Estilo CSS con background-color y color
        """
        if pd.isna(val):
            return "background-color: #F8F9FA; color: #6C757D"  # Gris neutral para NaN

        try:
            cluster_num = int(val)
            bg_color = colors["background"].get(cluster_num, "#F8F9FA")
            text_color = colors["text"].get(cluster_num, "#6C757D")

            return f"background-color: {bg_color}; color: {text_color}; font-weight: 500; border-left: 3px solid {colors['solid'].get(cluster_num, '#DEE2E6')}"
        except (ValueError, TypeError):
            return "background-color: #F8F9FA; color: #6C757D"

    return color_cluster


def get_plotly_color_sequence() -> list:
    """
    Retorna la secuencia de colores pastel para gráficos Plotly.

    Esta función proporciona una lista ordenada de colores que se
    utilizará consistentemente en todos los gráficos de la aplicación.

    Returns:
        list: Lista de colores hex en orden de preferencia
    """
    colors = get_cluster_colors()
    return [colors["plotly"][i] for i in range(1, 5)]


def get_plotly_color_map() -> Dict[int, str]:
    """
    Retorna un mapa de colores para discrete color mapping en Plotly.

    Returns:
        Dict: Mapeo de cluster (int) a color (str)
    """
    colors = get_cluster_colors()
    return {
        1: colors["plotly"][1],
        2: colors["plotly"][2],
        3: colors["plotly"][3],
        4: colors["plotly"][4],
        "1": colors["plotly"][1],  # String versions
        "2": colors["plotly"][2],
        "3": colors["plotly"][3],
        "4": colors["plotly"][4],
    }


def configure_page(page_title: str, page_icon: str = "🍑", layout: str = "wide") -> None:
    """
    Aplica la configuración y estilos compartidos de Streamlit.

    Esta función centraliza la configuración de páginas para mantener
    consistencia visual y funcional en toda la aplicación.

    Args:
        page_title: Título que aparece en la pestaña del navegador
        page_icon: Emoji o URL de icono para la pestaña
        layout: Layout de Streamlit ('wide', 'centered')
    """
    import streamlit as st

    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout=layout,
        initial_sidebar_state="expanded",
    )
    st.markdown(COMMON_STYLES, unsafe_allow_html=True)
