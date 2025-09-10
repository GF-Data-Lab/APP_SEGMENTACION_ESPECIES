
import streamlit as st
from utils import show_logo  # Asegúrate de tener esta función que muestra el logo

# Configuración de la página
st.set_page_config(
    page_title="Segmentación de Especies de Carozos", 
    page_icon="🍑", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
      /* Botones del sidebar */
      [data-testid="stSidebar"] div.stButton > button {
        background-color: #D32F2F !important;  /* rojo fuerte */
        color: white !important;
        border: none !important;
      }
      [data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #B71C1C !important;  /* rojo más oscuro al pasar */
      }
      
      /* Mejorar el aspecto de los containers */
      .stContainer {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #D32F2F;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      }
      
      /* Botones de navegación principales */
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
        boton_carga = st.button('Carga de archivos 📁')
        boton_ciruela = st.button('Segmentación Ciruela 🍑')
        boton_nectarina = st.button('Segmentación Nectarina 🍑')
        boton_cluster = st.button('Modelo de Clasificación')
        boton_analisis = st.button('Análisis exploratorio')
        boton_metricas = st.button('Métricas y Bandas 📊')
        boton_outliers = st.button('Detección Outliers 🎯')
    # Acción de los botones: redirigir a la página correspondiente
    if boton_inicio:
        st.switch_page('app.py')  # Redirige a la página principal
    if boton_carga:
        st.switch_page('pages/carga_datos.py')
    if boton_ciruela:
        st.switch_page('pages/segmentacion_ciruela.py')
    if boton_nectarina:
        st.switch_page('pages/segmentacion_nectarina.py')
    if boton_cluster:
        st.switch_page('pages/Cluster_especies.py')
    if boton_analisis:
        st.switch_page('pages/analisis.py')
    if boton_metricas:
        st.switch_page('pages/metricas_bandas.py')
    if boton_outliers:
        st.switch_page('pages/outliers.py')

# Llamar a la función para generar el menú en la barra lateral
generarMenu()

# ========================================================================================
# PÁGINA DE INICIO - DESCRIPCIÓN DEL PROYECTO Y GUÍA DE NAVEGACIÓN
# ========================================================================================

st.title("🍑 Sistema de Segmentación y Análisis de Especies de Carozos")

st.markdown("""
## 📋 Descripción del Proyecto

Esta aplicación de **análisis avanzado** está diseñada para procesar y analizar datos de calidad de frutos de carozo 
(ciruelas, nectarinas, duraznos, etc.), implementando algoritmos de **segmentación automática**, **detección de outliers** 
y **clustering inteligente** para la clasificación de variedades.

### 🎯 Objetivos Principales:
- **Automatizar** la clasificación de frutos según métricas de calidad
- **Detectar anomalías** en los datos de producción
- **Agrupar variedades** similares mediante técnicas de machine learning
- **Visualizar patrones** en los datos de cosecha y postcosecha

### 📊 Características Técnicas:
- ✅ **Análisis multivariado** de firmeza, acidez, sólidos solubles y color
- ✅ **Clustering automático** con algoritmos de k-means y PCA
- ✅ **Detección de outliers** usando Z-Score e IQR  
- ✅ **Reglas configurables** para diferentes especies y periodos
- ✅ **Filtrado inteligente** de datos atípicos
- ✅ **Visualizaciones interactivas** con gráficos dinámicos

---
""")

# Crear las tarjetas de navegación
st.markdown("## 🗺️ Guía de Navegación")
st.markdown("Sigue el **flujo de trabajo recomendado** para obtener los mejores resultados:")

# Crear columnas para organizar las tarjetas
col1, col2 = st.columns(2)

with col1:
    # Tarjeta 1: Carga de datos
    with st.container():
        st.markdown("""
        ### 📁 **1. Carga de Archivos**
        **🎯 Propósito:** Importar y validar datos de Excel
        
        **📝 Qué hacer:**
        - Subir archivo "MAESTRO CAROZOS FINAL COMPLETO CG.xlsx"
        - Verificar que los datos se carguen correctamente
        - Revisar estructura y columnas disponibles
        
        **⏱️ Tiempo estimado:** 2-3 minutos
        """)
        if st.button("➡️ Ir a Carga de Archivos", key="nav_carga"):
            st.switch_page('pages/carga_datos.py')
    
    # Tarjeta 3: Segmentación Ciruela
    with st.container():
        st.markdown("""
        ### 🍑 **3. Segmentación Ciruela**
        **🎯 Propósito:** Analizar y clasificar variedades de ciruela
        
        **📝 Qué hacer:**
        - Configurar parámetros específicos para ciruelas
        - Ejecutar segmentación por tipos (Candy/Sugar)
        - Revisar resultados de clustering
        - Descargar datos procesados
        
        **⏱️ Tiempo estimado:** 5-10 minutos
        """)
        if st.button("➡️ Ir a Segmentación Ciruela", key="nav_ciruela"):
            st.switch_page('pages/segmentacion_ciruela.py')
    
    # Tarjeta 5: Modelo de Clasificación
    with st.container():
        st.markdown("""
        ### 🤖 **5. Modelo de Clasificación**
        **🎯 Propósito:** Crear modelos predictivos de clasificación
        
        **📝 Qué hacer:**
        - Entrenar modelos de machine learning
        - Evaluar precisión y métricas de desempeño
        - Aplicar modelos a nuevos datos
        - Comparar diferentes algoritmos
        
        **⏱️ Tiempo estimado:** 10-15 minutos
        """)
        if st.button("➡️ Ir a Modelo de Clasificación", key="nav_modelo"):
            st.switch_page('pages/Cluster_especies.py')
    
    # Tarjeta 7: Métricas y Bandas
    with st.container():
        st.markdown("""
        ### 📊 **7. Métricas y Bandas**
        **🎯 Propósito:** Configurar reglas de clasificación personalizadas
        
        **📝 Qué hacer:**
        - Ajustar umbrales de firmeza, acidez y brix
        - Definir bandas de clasificación por especie
        - Crear reglas específicas por periodo de cosecha
        - Guardar configuraciones personalizadas
        
        **⏱️ Tiempo estimado:** 15-20 minutos
        """)
        if st.button("➡️ Ir a Métricas y Bandas", key="nav_metricas"):
            st.switch_page('pages/metricas_bandas.py')

with col2:
    # Tarjeta 2: Detección de Outliers
    with st.container():
        st.markdown("""
        ### 🎯 **2. Detección de Outliers**  
        **🎯 Propósito:** Identificar y filtrar datos atípicos
        
        **📝 Qué hacer:**
        - Configurar métodos de detección (Z-Score/IQR)
        - Visualizar distribuciones y anomalías
        - Aplicar filtros para excluir outliers
        - Exportar datos limpios
        
        **⏱️ Tiempo estimado:** 5-8 minutos
        """)
        if st.button("➡️ Ir a Detección de Outliers", key="nav_outliers"):
            st.switch_page('pages/outliers.py')
    
    # Tarjeta 4: Segmentación Nectarina
    with st.container():
        st.markdown("""
        ### 🍑 **4. Segmentación Nectarina**
        **🎯 Propósito:** Analizar y clasificar variedades de nectarina
        
        **📝 Qué hacer:**
        - Configurar parámetros para nectarinas
        - Segmentar por color (Amarilla/Blanca) y periodo
        - Generar clusters automáticos
        - Analizar patrones de calidad
        
        **⏱️ Tiempo estimado:** 5-10 minutos
        """)
        if st.button("➡️ Ir a Segmentación Nectarina", key="nav_nectarina"):
            st.switch_page('pages/segmentacion_nectarina.py')
    
    # Tarjeta 6: Análisis Exploratorio
    with st.container():
        st.markdown("""
        ### 🔍 **6. Análisis Exploratorio**
        **🎯 Propósito:** Explorar patrones y realizar clustering avanzado
        
        **📝 Qué hacer:**
        - Generar gráficos de dispersión y correlación
        - Aplicar PCA para reducción dimensional
        - Ejecutar clustering K-means interactivo
        - Exportar visualizaciones
        
        **⏱️ Tiempo estimado:** 10-20 minutos
        """)
        if st.button("➡️ Ir a Análisis Exploratorio", key="nav_analisis"):
            st.switch_page('pages/analisis.py')

# Sección de flujo de trabajo recomendado
st.markdown("---")
st.markdown("## 🔄 Flujo de Trabajo Recomendado")

st.markdown("""
### 📋 **Para Usuarios Nuevos:**
1. **📁 Carga de Archivos** → Subir datos
2. **🎯 Detección Outliers** → Limpiar datos  
3. **🍑 Segmentación [Especie]** → Procesar y analizar
4. **📊 Métricas y Bandas** → Ajustar reglas (si necesario)
5. **🔍 Análisis Exploratorio** → Explorar resultados

### 🚀 **Para Usuarios Avanzados:**
1. **📊 Métricas y Bandas** → Configurar reglas personalizadas
2. **📁 Carga de Archivos** → Subir datos
3. **🎯 Detección Outliers** → Configurar filtros específicos
4. **🍑 Segmentación [Especie]** → Ejecutar con configuración avanzada
5. **🤖 Modelo de Clasificación** → Entrenar modelos predictivos
6. **🔍 Análisis Exploratorio** → Clustering y PCA avanzado

---

### 📞 **Soporte y Documentación**
- **📊 Visualizaciones:** Todos los gráficos son interactivos y exportables
- **💾 Datos:** Los resultados se pueden descargar en formato Excel/CSV  
- **🔧 Configuración:** Las reglas y parámetros se guardan automáticamente
- **🔄 Flujo:** La aplicación recuerda el progreso entre páginas

### 🎓 **Interpretación de Resultados**
- **Clusters 1-4:** Representan diferentes niveles de calidad (1=Excelente, 4=Deficiente)
- **Gráficos PCA:** Muestran agrupaciones naturales en los datos
- **Outliers:** Datos que se desvían significativamente del patrón normal
- **Métricas:** Firmeza, acidez y brix son los indicadores clave de calidad
""")

st.markdown("---")

# Mostrar métricas del sistema si hay datos cargados
if "carozos_df" in st.session_state:
    st.markdown("## 📊 Estado Actual del Sistema")
    
    df = st.session_state["carozos_df"]
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    
    with col_metric1:
        st.metric("📋 Total Registros", len(df), help="Número total de registros en el dataset")
    
    with col_metric2:
        especies_count = df['Especie'].nunique() if 'Especie' in df.columns else 0
        st.metric("🍑 Especies", especies_count, help="Número de especies diferentes")
    
    with col_metric3:
        variedades_count = df['Variedad'].nunique() if 'Variedad' in df.columns else 0
        st.metric("🌱 Variedades", variedades_count, help="Número de variedades diferentes")
    
    with col_metric4:
        # Verificar si hay datos filtrados
        if "carozos_df_filtered" in st.session_state:
            filtered_count = len(st.session_state["carozos_df_filtered"])
            delta = filtered_count - len(df)
            st.metric("🎯 Datos Filtrados", filtered_count, delta=delta, help="Registros después de aplicar filtros")
        else:
            st.metric("🎯 Datos Filtrados", "No aplicados", help="No se han aplicado filtros de outliers")
    
    # Mostrar distribución de especies si existe la columna
    if 'Especie' in df.columns:
        st.markdown("### 📈 Distribución de Especies")
        especies_dist = df['Especie'].value_counts()
        col_chart1, col_chart2 = st.columns([3, 1])
        
        with col_chart1:
            st.bar_chart(especies_dist)
        
        with col_chart2:
            st.markdown("**Resumen:**")
            for especie, count in especies_dist.items():
                percentage = (count / len(df)) * 100
                st.write(f"• **{especie}:** {count:,} ({percentage:.1f}%)")

st.markdown("---")
st.success("🚀 **¡Comienza tu análisis!** Utiliza el menú lateral para navegar entre las diferentes funcionalidades.")

