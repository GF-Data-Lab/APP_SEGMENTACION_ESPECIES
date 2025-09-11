import streamlit as st
import pandas as pd
import numpy as np
from utils import show_logo

# Configuración de página
st.set_page_config(
    page_title="🔍 Verificación de Cálculos", 
    page_icon="🔍", 
    layout="wide"
)

# Estilos CSS
st.markdown("""
    <style>
      [data-testid="stSidebar"] div.stButton > button {
        background-color: #D32F2F !important;
        color: white !important;
        border: none !important;
      }
      [data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #B71C1C !important;
      }
      
      .calculation-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
      }
      
      .error-box {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #f44336;
      }
      
      .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2196f3;
      }
    </style>
""", unsafe_allow_html=True)

def generarMenu():
    with st.sidebar:
        show_logo()
        if st.button('Página de Inicio 🏚️'):
            st.switch_page('app.py')
        if st.button('Carga de archivos 📁'):
            st.switch_page('pages/carga_datos.py')
        if st.button('Segmentación Ciruela 🍑'):
            st.switch_page('pages/segmentacion_ciruela.py')
        if st.button('Segmentación Nectarina 🍑'):
            st.switch_page('pages/segmentacion_nectarina.py')
        if st.button('Modelo de Clasificación'):
            st.switch_page('pages/Cluster_especies.py')
        if st.button('Análisis exploratorio'):
            st.switch_page('pages/analisis.py')
        if st.button('Métricas y Bandas 📊'):
            st.switch_page('pages/metricas_bandas.py')
        if st.button('Detección Outliers 🎯'):
            st.switch_page('pages/outliers.py')
        if st.button('Verificar Cálculos 🔍'):
            st.switch_page('pages/verificar_calculos.py')

generarMenu()

# Título principal
st.title("🔍 Verificación de Cálculos de Clustering")
st.markdown("""
Esta página permite verificar los cálculos de clustering para una variedad específica,
mostrando paso a paso cómo se obtienen los valores de bandas y el cluster final.
""")

# Verificar que existan datos
if "agg_groups_plum" not in st.session_state and "agg_groups_nect" not in st.session_state:
    st.warning("⚠️ No hay datos procesados. Por favor, ejecuta primero la segmentación en Ciruela o Nectarina.")
    st.stop()

# Seleccionar especie
especies_disponibles = []
if "agg_groups_plum" in st.session_state:
    especies_disponibles.append("Ciruela")
if "agg_groups_nect" in st.session_state:
    especies_disponibles.append("Nectarina")

especie_seleccionada = st.selectbox(
    "Selecciona la especie:",
    options=especies_disponibles,
    key="especie_verificacion"
)

# Obtener datos según especie
if especie_seleccionada == "Ciruela":
    agg_data = st.session_state.get("agg_groups_plum")
    raw_data = st.session_state.get("df_processed_plum")
else:
    agg_data = st.session_state.get("agg_groups_nect")
    raw_data = st.session_state.get("df_processed_nect")

if agg_data is None or raw_data is None:
    st.error("❌ No hay datos completos para la especie seleccionada.")
    st.stop()

# Columnas importantes
ESPECIE_COLUMN = "Especie"
VAR_COLUMN = "Variedad"
FRUTO_COLUMN = "N° Muestra"
COL_BRIX = "Sólidos Solubles (%)"
COL_ACIDEZ = "Acidez (%)"

# Obtener lista de variedades únicas
variedades = agg_data[VAR_COLUMN].unique() if VAR_COLUMN in agg_data.columns else []

if len(variedades) == 0:
    st.error("❌ No se encontraron variedades en los datos.")
    st.stop()

# Seleccionar variedad a verificar
st.markdown("---")
variedad_seleccionada = st.selectbox(
    "Selecciona la variedad a verificar:",
    options=sorted(variedades),
    key="variedad_verificacion"
)

# Filtrar datos por variedad
datos_variedad = raw_data[raw_data[VAR_COLUMN] == variedad_seleccionada]
agg_variedad = agg_data[agg_data[VAR_COLUMN] == variedad_seleccionada]

st.markdown(f"## 📊 Análisis de la variedad: **{variedad_seleccionada}**")

# Información general
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_muestras = len(datos_variedad)
    st.metric("Total Muestras", total_muestras)
with col2:
    frutos_unicos = datos_variedad[FRUTO_COLUMN].nunique() if FRUTO_COLUMN in datos_variedad.columns else 0
    st.metric("Frutos Únicos", frutos_unicos)
with col3:
    temporadas = agg_variedad['harvest_period'].unique() if 'harvest_period' in agg_variedad.columns else []
    st.metric("Temporadas", len(temporadas))
with col4:
    cluster_asignado = agg_variedad['cluster_grp'].mode().iloc[0] if 'cluster_grp' in agg_variedad.columns and not agg_variedad['cluster_grp'].empty else "N/A"
    st.metric("Cluster Predominante", cluster_asignado)

# Paso 1: Datos crudos por muestra
st.markdown("---")
st.markdown("### 📋 Paso 1: Datos crudos por muestra")
st.markdown('<div class="info-box">Estos son los valores individuales de cada muestra de la variedad seleccionada.</div>', unsafe_allow_html=True)

# Mostrar datos relevantes
columnas_mostrar = [col for col in [VAR_COLUMN, FRUTO_COLUMN, 'harvest_period', COL_BRIX, COL_ACIDEZ, 'Firmeza punto valor'] 
                   if col in datos_variedad.columns]

if columnas_mostrar:
    st.dataframe(
        datos_variedad[columnas_mostrar].sort_values([FRUTO_COLUMN] if FRUTO_COLUMN in columnas_mostrar else columnas_mostrar[0]),
        use_container_width=True,
        height=300
    )

# Paso 2: Promedios por fruto
st.markdown("---")
st.markdown("### 📊 Paso 2: Promedios por fruto individual")
st.markdown('<div class="calculation-box">Se calcula el promedio de BRIX, Acidez y Firmeza para cada fruto individual.</div>', unsafe_allow_html=True)

if FRUTO_COLUMN in datos_variedad.columns:
    # Calcular promedios por fruto
    promedios_fruto = (
        datos_variedad
        .groupby([FRUTO_COLUMN, 'harvest_period'] if 'harvest_period' in datos_variedad.columns else [FRUTO_COLUMN])
        .agg({
            COL_BRIX: 'mean',
            COL_ACIDEZ: 'mean' if COL_ACIDEZ in datos_variedad.columns else lambda x: np.nan,
            'Firmeza punto valor': ['mean', 'min'] if 'Firmeza punto valor' in datos_variedad.columns else lambda x: np.nan
        })
        .round(2)
    )
    
    # Aplanar columnas multinivel
    promedios_fruto.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in promedios_fruto.columns]
    promedios_fruto = promedios_fruto.reset_index()
    
    st.dataframe(promedios_fruto, use_container_width=True)
    
    # Estadísticas
    col1, col2, col3 = st.columns(3)
    with col1:
        brix_promedio_frutos = promedios_fruto[f'{COL_BRIX}_mean'].mean() if f'{COL_BRIX}_mean' in promedios_fruto.columns else 0
        st.metric("BRIX Promedio (todos los frutos)", f"{brix_promedio_frutos:.2f}")
    with col2:
        if f'{COL_ACIDEZ}_mean' in promedios_fruto.columns:
            acidez_promedio_frutos = promedios_fruto[f'{COL_ACIDEZ}_mean'].mean()
            st.metric("Acidez Promedio (todos los frutos)", f"{acidez_promedio_frutos:.2f}")
    with col3:
        if 'Firmeza punto valor_min' in promedios_fruto.columns:
            firmeza_min_global = promedios_fruto['Firmeza punto valor_min'].min()
            st.metric("Firmeza Mínima Global", f"{firmeza_min_global:.2f}")

# Paso 3: Agregación por variedad-temporada
st.markdown("---")
st.markdown("### 🎯 Paso 3: Agregación por variedad-temporada")
st.markdown('<div class="calculation-box">Se promedian los valores de todos los frutos de cada combinación variedad-temporada.</div>', unsafe_allow_html=True)

if 'harvest_period' in agg_variedad.columns:
    # Mostrar agregación por temporada
    temporadas_data = []
    for temporada in agg_variedad['harvest_period'].unique():
        datos_temp = datos_variedad[datos_variedad['harvest_period'] == temporada] if 'harvest_period' in datos_variedad.columns else datos_variedad
        
        # Calcular estadísticas
        row_data = {
            'Temporada': temporada,
            'Frutos': datos_temp[FRUTO_COLUMN].nunique() if FRUTO_COLUMN in datos_temp.columns else 0,
            'Muestras': len(datos_temp),
            'BRIX Promedio': datos_temp[COL_BRIX].mean() if COL_BRIX in datos_temp.columns else np.nan,
            'Acidez Promedio': datos_temp[COL_ACIDEZ].mean() if COL_ACIDEZ in datos_temp.columns else np.nan,
            'Firmeza Mínima': datos_temp['Firmeza punto valor'].min() if 'Firmeza punto valor' in datos_temp.columns else np.nan,
        }
        temporadas_data.append(row_data)
    
    df_temporadas = pd.DataFrame(temporadas_data)
    st.dataframe(df_temporadas.round(2), use_container_width=True)

# Paso 4: Asignación de bandas
st.markdown("---")
st.markdown("### 🎖️ Paso 4: Asignación de bandas (1-4)")
st.markdown('<div class="info-box">Se asignan bandas basadas en cuartiles donde 1=mejor, 4=peor.</div>', unsafe_allow_html=True)

# Colores para bandas
band_colors = {
    1: '#a8e6cf',  # verde claro - Excelente
    2: '#ffd3b6',  # naranja claro - Bueno  
    3: '#ffaaa5',  # coral - Regular
    4: '#ff8b94',  # rojo rosado - Deficiente
}

# Mostrar información de bandas si está disponible
if 'banda_brix' in agg_variedad.columns:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🍇 Banda BRIX**")
        for _, row in agg_variedad.iterrows():
            banda = row.get('banda_brix', 0)
            valor = row.get('promedio_brix', np.nan)
            if banda > 0:
                color = band_colors.get(int(banda), '#cccccc')
                st.markdown(f'<div style="background-color: {color}; padding: 10px; border-radius: 5px; text-align: center;">Banda {int(banda)} - Valor: {valor:.2f}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("**💪 Banda Firmeza**")
        for _, row in agg_variedad.iterrows():
            banda = row.get('banda_firmeza', 0)
            valor = row.get('promedio_firmeza_punto', np.nan)
            metodo = row.get('firmeza_metodo_usado', 'N/A')
            if banda > 0:
                color = band_colors.get(int(banda), '#cccccc')
                st.markdown(f'<div style="background-color: {color}; padding: 10px; border-radius: 5px; text-align: center;">Banda {int(banda)} - Valor: {valor:.2f}<br>Método: {metodo}</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown("**🧪 Banda Acidez**")
        for _, row in agg_variedad.iterrows():
            banda = row.get('banda_acidez', 0)
            valor = row.get('promedio_acidez', np.nan)
            if banda > 0:
                color = band_colors.get(int(banda), '#cccccc')
                st.markdown(f'<div style="background-color: {color}; padding: 10px; border-radius: 5px; text-align: center;">Banda {int(banda)} - Valor: {valor:.2f}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="background-color: #e0e0e0; padding: 10px; border-radius: 5px; text-align: center;">Sin datos</div>', unsafe_allow_html=True)

# Paso 5: Cálculo del cluster final
st.markdown("---")
st.markdown("### 🏆 Paso 5: Cálculo del cluster final")
st.markdown('<div class="calculation-box">Se suman las bandas y se asigna el cluster según rangos específicos.</div>', unsafe_allow_html=True)

if 'suma_bandas' in agg_variedad.columns:
    for _, row in agg_variedad.iterrows():
        temporada = row.get('harvest_period', 'N/A')
        suma = row.get('suma_bandas', 0)
        cluster = row.get('cluster_grp', 0)
        
        # Calcular componentes de la suma
        banda_brix = row.get('banda_brix', 0)
        banda_firmeza = row.get('banda_firmeza', 0)
        banda_acidez = row.get('banda_acidez', 0)
        
        st.markdown(f"#### Temporada: {temporada}")
        
        # Mostrar cálculo
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"**BRIX**: Banda {int(banda_brix)}")
        with col2:
            st.markdown(f"**Firmeza**: Banda {int(banda_firmeza)}")
        with col3:
            st.markdown(f"**Acidez**: Banda {int(banda_acidez)}")
        with col4:
            st.markdown(f"**SUMA TOTAL**: {int(suma)}")
        
        # Mostrar cluster asignado
        cluster_color = band_colors.get(int(cluster), '#cccccc')
        cluster_name = {1: "Excelente", 2: "Bueno", 3: "Regular", 4: "Deficiente"}.get(int(cluster), "N/A")
        
        st.markdown(f'<div style="background-color: {cluster_color}; padding: 15px; border-radius: 10px; text-align: center; font-size: 18px; font-weight: bold; margin-top: 10px;">CLUSTER ASIGNADO: {int(cluster)} - {cluster_name}</div>', unsafe_allow_html=True)
        
        # Explicación del rango
        st.markdown("**Rangos de clustering (para 3 métricas):**")
        st.markdown("- Suma 3-5 → Cluster 1 (Excelente)")
        st.markdown("- Suma 6-8 → Cluster 2 (Bueno)")
        st.markdown("- Suma 9-11 → Cluster 3 (Regular)")
        st.markdown("- Suma 12+ → Cluster 4 (Deficiente)")

# Verificación de consistencia
st.markdown("---")
st.markdown("### ✅ Verificación de consistencia")

# Verificar si hay inconsistencias
inconsistencias = []

# Verificar que los promedios estén bien calculados
if FRUTO_COLUMN in datos_variedad.columns and COL_BRIX in datos_variedad.columns:
    brix_real = datos_variedad[COL_BRIX].mean()
    brix_reportado = agg_variedad['promedio_brix'].mean() if 'promedio_brix' in agg_variedad.columns else 0
    
    if abs(brix_real - brix_reportado) > 0.1:
        inconsistencias.append(f"⚠️ Diferencia en BRIX promedio: Real={brix_real:.2f}, Reportado={brix_reportado:.2f}")

# Mostrar resultados de verificación
if inconsistencias:
    st.markdown('<div class="error-box">', unsafe_allow_html=True)
    st.markdown("**Se encontraron las siguientes inconsistencias:**")
    for inc in inconsistencias:
        st.markdown(inc)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.success("✅ Los cálculos son consistentes y correctos.")

# Exportar resultados
st.markdown("---")
if st.button("📥 Exportar verificación a Excel"):
    # Crear archivo Excel con múltiples hojas
    output = pd.ExcelWriter('verificacion_calculos.xlsx', engine='xlsxwriter')
    
    # Hoja 1: Datos crudos
    datos_variedad[columnas_mostrar].to_excel(output, sheet_name='Datos Crudos', index=False)
    
    # Hoja 2: Promedios por fruto
    if 'promedios_fruto' in locals():
        promedios_fruto.to_excel(output, sheet_name='Promedios Fruto', index=False)
    
    # Hoja 3: Agregación variedad-temporada
    if 'df_temporadas' in locals():
        df_temporadas.to_excel(output, sheet_name='Variedad-Temporada', index=False)
    
    # Hoja 4: Resumen clustering
    agg_variedad.to_excel(output, sheet_name='Clustering', index=False)
    
    output.close()
    
    with open('verificacion_calculos.xlsx', 'rb') as f:
        st.download_button(
            label="Descargar archivo Excel",
            data=f.read(),
            file_name=f'verificacion_{variedad_seleccionada}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )