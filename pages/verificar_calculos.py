import streamlit as st
import pandas as pd
import numpy as np
from common_styles import configure_page, generarMenu, get_cluster_colors
from data_columns import (
    COL_ESPECIE,
    COL_VARIEDAD,
    COL_FRUTO,
    COL_BRIX as BRIX_COLUMN,
    COL_ACIDEZ as ACIDEZ_COLUMN,
)

# Configuración de página con estilos unificados
configure_page("🔍 Verificación de Cálculos", "✅")
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
ESPECIE_COLUMN = COL_ESPECIE
VAR_COLUMN = COL_VARIEDAD
FRUTO_COLUMN = COL_FRUTO
COL_BRIX = BRIX_COLUMN
COL_ACIDEZ = ACIDEZ_COLUMN

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
    agg_dict = {COL_BRIX: 'mean'}

    # Agregar columnas solo si existen
    if COL_ACIDEZ in datos_variedad.columns:
        agg_dict[COL_ACIDEZ] = 'mean'

    if 'Firmeza punto valor' in datos_variedad.columns:
        agg_dict['Firmeza punto valor'] = ['mean', 'min']

    promedios_fruto = (
        datos_variedad
        .groupby([FRUTO_COLUMN, 'harvest_period'] if 'harvest_period' in datos_variedad.columns else [FRUTO_COLUMN])
        .agg(agg_dict)
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

# Usar paleta de colores pastel estándar de la aplicación
def get_band_colors():
    """Obtiene los colores estándar para las bandas de calidad."""
    colors = get_cluster_colors()
    return colors['solid']  # Usar colores sólidos pastel estándar

band_colors = get_band_colors()


def _safe_positive_int(value):
    if value is None or pd.isna(value):
        return None
    try:
        as_int = int(value)
    except (ValueError, TypeError):
        return None
    return as_int if as_int > 0 else None

def _format_metric_value(value):
    return f"{value:.2f}" if value is not None and not pd.isna(value) else "Sin dato"

def _render_band_card(band_value, metric_value, extra_text=None):
    band_int = _safe_positive_int(band_value)
    if band_int is None:
        st.markdown("<div style='background-color: #e0e0e0; padding: 10px; border-radius: 5px; text-align: center;'>Sin datos</div>", unsafe_allow_html=True)
        return
    colors = get_cluster_colors()
    color = colors['solid'].get(band_int, '#F8F9FA')  # Usar color estándar o gris neutral
    metric_text = _format_metric_value(metric_value)
    extra_html = f"<br>{extra_text}" if extra_text else ""
    st.markdown(
        f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; text-align: center;'>Banda {band_int} - Valor: {metric_text}{extra_html}</div>",
        unsafe_allow_html=True,
    )

# Mostrar información de bandas si está disponible
if 'banda_brix' in agg_variedad.columns:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Banda BRIX**")
        for _, row in agg_variedad.iterrows():
            _render_band_card(row.get('banda_brix'), row.get('promedio_brix'))

    with col2:
        st.markdown("**Banda Firmeza**")
        for _, row in agg_variedad.iterrows():
            band_value = row.get('banda_firmeza')
            if band_value is None:
                band_value = row.get('banda_firmeza_punto')
            metodo = row.get('firmeza_metodo_usado', 'N/A')
            if pd.isna(metodo):
                metodo = 'N/A'
            _render_band_card(
                band_value,
                row.get('promedio_firmeza_punto'),
                extra_text=f"Metodo: {metodo}",
            )

    with col3:
        st.markdown("**Banda Acidez**")
        for _, row in agg_variedad.iterrows():
            _render_band_card(row.get('banda_acidez'), row.get('promedio_acidez'))

# Paso 5: Calculo del cluster final
st.markdown("---")
st.markdown("### Paso 5: Calculo del cluster final")
st.markdown('<div class="calculation-box">Se suman las bandas y se asigna el cluster segun rangos especificos.</div>', unsafe_allow_html=True)

if 'suma_bandas' in agg_variedad.columns:
    for _, row in agg_variedad.iterrows():
        temporada = row.get('harvest_period', 'N/A')
        st.markdown(f"#### Temporada: {temporada}")

        banda_brix = _safe_positive_int(row.get('banda_brix'))
        banda_firmeza = _safe_positive_int(row.get('banda_firmeza', row.get('banda_firmeza_punto')))
        banda_acidez = _safe_positive_int(row.get('banda_acidez'))
        suma_bandas = _safe_positive_int(row.get('suma_bandas'))
        cluster_val = _safe_positive_int(row.get('cluster_grp'))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"**BRIX**: Banda {banda_brix}" if banda_brix is not None else "**BRIX**: Sin datos")
        with col2:
            st.markdown(f"**Firmeza**: Banda {banda_firmeza}" if banda_firmeza is not None else "**Firmeza**: Sin datos")
        with col3:
            st.markdown(f"**Acidez**: Banda {banda_acidez}" if banda_acidez is not None else "**Acidez**: Sin datos")
        with col4:
            st.markdown(f"**SUMA TOTAL**: {suma_bandas}" if suma_bandas is not None else "**SUMA TOTAL**: Sin datos")

        colors = get_cluster_colors()
        cluster_color = colors['solid'].get(cluster_val, '#F8F9FA') if cluster_val else '#E9ECEF'
        cluster_name = {1: "Excelente", 2: "Bueno", 3: "Regular", 4: "Deficiente"}.get(cluster_val, "") if cluster_val else ""
        cluster_text = f"{cluster_val} - {cluster_name}" if cluster_val else "Sin datos"

        st.markdown(
            f"<div style='background-color: {cluster_color}; padding: 15px; border-radius: 10px; text-align: center; font-size: 18px; font-weight: bold; margin-top: 10px;'>CLUSTER ASIGNADO: {cluster_text}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("**Rangos de clustering (para 3 metricas):**")
        st.markdown("- Suma 3-5 -> Cluster 1 (Excelente)")
        st.markdown("- Suma 6-8 -> Cluster 2 (Bueno)")
        st.markdown("- Suma 9-11 -> Cluster 3 (Regular)")
        st.markdown("- Suma 12+ -> Cluster 4 (Deficiente)")

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
