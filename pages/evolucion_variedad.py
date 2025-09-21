import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations
from common_styles import configure_page, get_cluster_colors, get_cluster_style_function, get_plotly_color_map, get_plotly_color_sequence, generarMenu
from data_columns import (
    COL_ESPECIE,
    COL_VARIEDAD,
    COL_FRUTO,
    COL_BRIX as BRIX_COLUMN,
    COL_ACIDEZ as ACIDEZ_COLUMN,
    COL_FECHA_EVALUACION,
    COL_CAMPO,
    COL_PORTAINJERTO,
)

# Configuración de página con estilos unificados
configure_page("📈 Evolución de Variedad", "📈")

# Estilos adicionales específicos para esta página
st.markdown("""
    <style>
      .improvement-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
      }
      
      .degradation-box {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3);
      }
      
      .stable-box {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
      }
    </style>
""", unsafe_allow_html=True)

generarMenu()

# Título principal
st.title("📈 Evolución Temporal de Variedades")
st.markdown("""
Esta página permite analizar la evolución de una variedad específica a través de las temporadas,
comparando métricas de calidad, clusters asignados y valores individuales de cada muestra.
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

col1, col2 = st.columns(2)
with col1:
    especie_seleccionada = st.selectbox(
        "Selecciona la especie:",
        options=especies_disponibles,
        key="especie_evolucion"
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
LOCALIDAD_COLUMN = "Localidad"
CAMPO_COLUMN = COL_CAMPO

# Obtener lista de variedades únicas y limpiar espacios
variedades_raw = agg_data[VAR_COLUMN].unique() if VAR_COLUMN in agg_data.columns else []

# Limpiar espacios en variedades y crear mapeo
variedades_clean = []
variedades_mapping = {}
for var in variedades_raw:
    var_clean = str(var).strip().replace(' ', '') if pd.notna(var) else str(var)
    variedades_clean.append(var_clean)
    variedades_mapping[var_clean] = var

# Mostrar advertencia si hay variedades con espacios
variedades_con_espacios = [var for var in variedades_raw if ' ' in str(var)]
if variedades_con_espacios:
    st.warning(f"⚠️ Se encontraron variedades con espacios que serán limpiadas: {variedades_con_espacios}")

if len(variedades_clean) == 0:
    st.error("❌ No se encontraron variedades en los datos.")
    st.stop()

# Filtros y selección
st.markdown("### 🔍 **Filtros de Selección**")
col_filter1, col_filter2 = st.columns(2)

with col_filter1:
    # Filtro por texto
    filtro_texto = st.text_input(
        "🔍 Filtrar variedades (escribe parte del nombre):",
        placeholder="Ej: I4, candy, sugar...",
        key="filtro_variedad"
    )
    
    # Aplicar filtro
    if filtro_texto:
        variedades_filtradas = [v for v in variedades_clean if filtro_texto.lower() in v.lower()]
    else:
        variedades_filtradas = variedades_clean
    
    st.info(f"📊 {len(variedades_filtradas)} de {len(variedades_clean)} variedades mostradas")

with col_filter2:
    # Selección múltiple de variedades
    variedades_seleccionadas = st.multiselect(
        "🎯 Selecciona variedades para analizar (múltiple):",
        options=sorted(variedades_filtradas),
        default=[sorted(variedades_filtradas)[0]] if variedades_filtradas else [],
        key="variedades_evolucion",
        help="Puedes seleccionar una o múltiples variedades para comparar su evolución"
    )

if not variedades_seleccionadas:
    st.warning("⚠️ Por favor selecciona al menos una variedad para analizar.")
    st.stop()

# Filtrar datos por variedades seleccionadas (mapear de vuelta a nombres originales)
variedades_originales = [variedades_mapping[v] for v in variedades_seleccionadas]
datos_variedades = raw_data[raw_data[VAR_COLUMN].isin(variedades_originales)]
agg_variedades = agg_data[agg_data[VAR_COLUMN].isin(variedades_originales)]
variedades_inverse_mapping = {v: k for k, v in variedades_mapping.items()}
season_col = 'temporada' if 'temporada' in agg_variedades.columns else 'harvest_period'
fecha_col = COL_FECHA_EVALUACION if COL_FECHA_EVALUACION in agg_variedades.columns else None

match_columns = [ESPECIE_COLUMN, VAR_COLUMN, season_col, COL_CAMPO, COL_PORTAINJERTO]
if fecha_col:
    match_columns.append(fecha_col)

timeline_rows = []
if not agg_variedades.empty:
    for _, grupo_row in agg_variedades.iterrows():
        var_original = grupo_row.get(VAR_COLUMN)
        var_clean_lookup = variedades_inverse_mapping.get(var_original, str(var_original).strip().replace(' ', ''))
        temporada_val = grupo_row.get(season_col)
        fecha_val = grupo_row.get(fecha_col) if fecha_col else np.nan

        mask = pd.Series(True, index=datos_variedades.index)
        for col in match_columns:
            if col in datos_variedades.columns and col in grupo_row.index:
                valor = grupo_row.get(col)
                if pd.isna(valor):
                    mask &= datos_variedades[col].isna()
                else:
                    mask &= datos_variedades[col] == valor

        datos_grupo = datos_variedades[mask]
        metric_cols = [col for col in [COL_BRIX, COL_ACIDEZ, 'Firmeza punto valor'] if col in datos_grupo.columns]
        if len(metric_cols) < 3:
            continue

        datos_grupo_validos = datos_grupo.dropna(subset=metric_cols)
        if datos_grupo_validos.empty:
            continue


        muestras = len(datos_grupo_validos)
        frutos = datos_grupo_validos[FRUTO_COLUMN].nunique() if FRUTO_COLUMN in datos_grupo_validos.columns else 0
        localidades_count = datos_grupo_validos[LOCALIDAD_COLUMN].nunique() if LOCALIDAD_COLUMN in datos_grupo_validos.columns else 0
        campos_count = datos_grupo_validos[CAMPO_COLUMN].nunique() if CAMPO_COLUMN in datos_grupo_validos.columns else 0

        band_values = [grupo_row.get(col) for col in ['banda_brix', 'banda_firmeza', 'banda_firmeza_punto', 'banda_acidez'] if col in grupo_row.index]
        if band_values and any(pd.isna(val) for val in band_values):
            continue

        timeline_rows.append({
            'Variedad': var_clean_lookup,
            'Variedad_Nombre': var_original,
            'Temporada': temporada_val,
            'Fecha': fecha_val,
            'GrupoID': grupo_row.get('grupo_id'),
            'GrupoKey': grupo_row.get('grupo_key'),
            'GrupoKeyDetalle': grupo_row.get('grupo_key_detalle'),
            'Cluster': grupo_row.get('cluster_grp'),
            'Suma_Bandas': grupo_row.get('suma_bandas'),
            'Banda_BRIX': grupo_row.get('banda_brix'),
            'Banda_Firmeza_Punto': grupo_row.get('banda_firmeza_punto'),
            'Banda_Mejillas': grupo_row.get('banda_mejillas'),
            'Banda_Acidez': grupo_row.get('banda_acidez'),
            'Muestras': muestras,
            'Frutos': frutos,
            'Localidades': localidades_count,
            'Campos': campos_count,
            'Campo': grupo_row.get('Campo'),
            'Portainjerto': grupo_row.get(COL_PORTAINJERTO),
            'BRIX_Promedio': grupo_row.get('brix_promedio'),
            'Acidez_Promedio': grupo_row.get('acidez_primer_fruto'),
            'Firmeza_Punto_Debil': grupo_row.get('firmeza_punto_debil'),
            'Mejillas_Promedio': grupo_row.get('mejillas_promedio'),
        })

timeline_df = pd.DataFrame(timeline_rows)

brix_combo_data = {}
brix_combo_export_frames = []
combo_label_columns = {}

if not timeline_df.empty:
    if 'Fecha' in timeline_df.columns:
        timeline_df['Fecha_dt'] = pd.to_datetime(timeline_df['Fecha'], errors='coerce')
    sort_columns = ['Variedad', 'Temporada']
    if 'Fecha_dt' in timeline_df.columns:
        sort_columns.append('Fecha_dt')
    timeline_df = timeline_df.sort_values(sort_columns)

combo_candidates = [
    (ESPECIE_COLUMN, 'Especie'),
    (VAR_COLUMN, 'Variedad'),
    (CAMPO_COLUMN, 'Campo'),
    (LOCALIDAD_COLUMN, 'Localidad'),
    (COL_PORTAINJERTO, 'Portainjerto')
]
if 'Color de pulpa' in datos_variedades.columns:
    combo_candidates.append(('Color de pulpa', 'Color de pulpa'))

available_combo_cols = [col for col, _ in combo_candidates if col in datos_variedades.columns]
combo_label_map = {col: label for col, label in combo_candidates if col in datos_variedades.columns}

if COL_BRIX in datos_variedades.columns and available_combo_cols:
    for r in range(1, len(available_combo_cols) + 1):
        for combo in combinations(available_combo_cols, r):
            grouping_cols = list(combo)
            if season_col and season_col not in grouping_cols:
                grouping_cols.append(season_col)
            if fecha_col and fecha_col not in grouping_cols:
                grouping_cols.append(fecha_col)

            grouped = (
                datos_variedades
                .groupby(grouping_cols, dropna=False)[COL_BRIX]
                .agg(['mean', 'count', 'min', 'max', 'std'])
                .reset_index()
            )

            if grouped.empty:
                continue

            grouped = grouped.rename(columns={
                'mean': 'BRIX_Promedio',
                'count': 'Registros',
                'min': 'BRIX_Min',
                'max': 'BRIX_Max',
                'std': 'BRIX_Std'
            })

            combo_label = ' + '.join(combo_label_map[c] for c in combo)
            grouped['Llave'] = combo_label

            if season_col and season_col in grouped.columns:
                grouped = grouped.rename(columns={season_col: 'Temporada'})
            if fecha_col and fecha_col in grouped.columns:
                grouped = grouped.rename(columns={fecha_col: 'Fecha'})

            brix_combo_data[combo_label] = grouped
            combo_label_columns[combo_label] = list(combo)
            brix_combo_export_frames.append(grouped.assign(Llave=combo_label))
else:
    brix_combo_data = {}
    brix_combo_export_frames = []
    combo_label_columns = {}

brix_combo_export_df = pd.concat(brix_combo_export_frames, ignore_index=True) if brix_combo_export_frames else pd.DataFrame()


if brix_combo_data:
    st.markdown("### ?? Promedios de BRIX por combinatoria de llaves")
    combo_options = sorted(brix_combo_data.keys())
    selected_combo = st.selectbox(
        "Selecciona la combinaci?n de llaves:",
        options=combo_options,
        index=0,
        key="combo_brix_llave"
    )

    selected_df = brix_combo_data[selected_combo].copy()
    base_key_cols = combo_label_columns.get(selected_combo, [])
    season_column_name = 'Temporada' if 'Temporada' in selected_df.columns else None
    fecha_column_name = 'Fecha' if 'Fecha' in selected_df.columns else None

    group_key_cols = [col for col in base_key_cols if col in selected_df.columns]
    if not group_key_cols:
        group_key_cols = ['Llave'] if 'Llave' in selected_df.columns else []

    display_columns = [col for col in [
        *group_key_cols,
        season_column_name,
        fecha_column_name,
        'BRIX_Promedio',
        'BRIX_Min',
        'BRIX_Max',
        'BRIX_Std',
        'Registros'
    ] if col and col in selected_df.columns]

    if group_key_cols:
        etiqueta_cols = selected_df[group_key_cols].fillna('NA').astype(str)
        selected_df['Grupo Etiqueta'] = etiqueta_cols.apply(lambda row: ' | '.join(row.values), axis=1)
    else:
        selected_df['Grupo Etiqueta'] = selected_combo

    st.dataframe(selected_df[display_columns + ['Grupo Etiqueta']] if 'Grupo Etiqueta' in selected_df.columns else selected_df[display_columns], use_container_width=True)

    if season_column_name and 'BRIX_Promedio' in selected_df.columns:
        fig_combo_temp = px.line(
            selected_df.sort_values(season_column_name),
            x=season_column_name,
            y='BRIX_Promedio',
            color='Grupo Etiqueta',
            markers=True,
            title=f'Evoluci?n de BRIX por temporada - {selected_combo}'
        )
        fig_combo_temp.update_xaxes(title_text='Temporada')
        fig_combo_temp.update_yaxes(title_text='BRIX Promedio')
        st.plotly_chart(fig_combo_temp, use_container_width=True)

    if fecha_column_name and selected_df[fecha_column_name].notna().any():
        combo_fecha_df = selected_df.copy()
        combo_fecha_df['Fecha_dt'] = pd.to_datetime(combo_fecha_df[fecha_column_name], errors='coerce')
        combo_fecha_df = combo_fecha_df.dropna(subset=['Fecha_dt'])
        if not combo_fecha_df.empty:
            fig_combo_fecha = px.line(
                combo_fecha_df.sort_values('Fecha_dt'),
                x='Fecha_dt',
                y='BRIX_Promedio',
                color='Grupo Etiqueta',
                markers=True,
                title=f'Variaci?n de BRIX por fecha - {selected_combo}'
            )
            fig_combo_fecha.update_xaxes(title_text='Fecha')
            fig_combo_fecha.update_yaxes(title_text='BRIX Promedio')
            st.plotly_chart(fig_combo_fecha, use_container_width=True)
else:
    st.info("No se pudieron calcular combinaciones de BRIX con las llaves disponibles.")

st.markdown("---")
if len(variedades_seleccionadas) == 1:
    st.markdown(f"## 📊 Análisis evolutivo: **{variedades_seleccionadas[0]}**")
else:
    st.markdown(f"## 📊 Comparación evolutiva: **{len(variedades_seleccionadas)} variedades**")
    st.markdown(f"**Variedades seleccionadas:** {', '.join(variedades_seleccionadas)}")

# Información general de las variedades
temporadas = sorted(agg_variedades['harvest_period'].unique()) if 'harvest_period' in agg_variedades.columns else []

# Información adicional sobre localidades y campos
localidades = datos_variedades[LOCALIDAD_COLUMN].nunique() if LOCALIDAD_COLUMN in datos_variedades.columns else 0
campos = datos_variedades[CAMPO_COLUMN].nunique() if CAMPO_COLUMN in datos_variedades.columns else 0

# Resumen general para todas las variedades seleccionadas
st.markdown("### 📈 **Resumen General**")
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_muestras = len(datos_variedades)
    st.metric("Total Muestras", total_muestras)
with col2:
    frutos_unicos = datos_variedades[FRUTO_COLUMN].nunique() if FRUTO_COLUMN in datos_variedades.columns else 0
    st.metric("Frutos Únicos", frutos_unicos)
with col3:
    st.metric("Temporadas", len(temporadas))
with col4:
    años_únicos = datos_variedades['harvest_period'].str.extract(r'(\d{4})').dropna().nunique() if 'harvest_period' in datos_variedades.columns else 0
    st.metric("Años de Datos", años_únicos)

# Segunda fila con información de localidades y campos
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Variedades", len(variedades_seleccionadas))
with col2:
    st.metric("Localidades", localidades)
with col3:
    st.metric("Campos", campos)
with col4:
    # Promedio de BRIX total
    brix_general = datos_variedades[COL_BRIX].mean() if COL_BRIX in datos_variedades.columns else 0
    st.metric("BRIX General", f"{brix_general:.2f}")

# Tercera fila con rangos
col1, col2, col3, col4 = st.columns(4)
with col1:
    # Rango de valores BRIX
    if COL_BRIX in datos_variedades.columns:
        brix_min = datos_variedades[COL_BRIX].min()
        brix_max = datos_variedades[COL_BRIX].max()
        st.metric("Rango BRIX", f"{brix_min:.1f} - {brix_max:.1f}")
    else:
        st.metric("Rango BRIX", "N/A")
with col2:
    if COL_ACIDEZ in datos_variedades.columns:
        acidez_prom = datos_variedades[COL_ACIDEZ].mean()
        st.metric("Acidez Promedio", f"{acidez_prom:.2f}")
    else:
        st.metric("Acidez Promedio", "N/A")
with col3:
    if 'Firmeza punto valor' in datos_variedades.columns:
        firmeza_prom = datos_variedades['Firmeza punto valor'].mean()
        st.metric("Firmeza Promedio", f"{firmeza_prom:.2f}")
    else:
        st.metric("Firmeza Promedio", "N/A")
with col4:
    clusters_unicos = agg_variedades['cluster_grp'].nunique() if 'cluster_grp' in agg_variedades.columns else 0
    st.metric("Clusters Únicos", clusters_unicos)

# Seguimiento detallado de grupos por fecha
if not timeline_df.empty:
    st.markdown("### Seguimiento por grupo y fecha")
    timeline_display = timeline_df.copy()
    if 'Fecha_dt' in timeline_display.columns:
        timeline_display['Fecha'] = timeline_display['Fecha_dt'].dt.date
    if 'Variedad' in timeline_display.columns and 'Variedad_Nombre' in timeline_display.columns:
        timeline_display = timeline_display.drop(columns=['Variedad'])
    if 'Variedad_Nombre' in timeline_display.columns:
        timeline_display = timeline_display.rename(columns={'Variedad_Nombre': 'Variedad'})
    timeline_display = timeline_display.reset_index(drop=True)
    timeline_display = timeline_display.loc[:, ~timeline_display.columns.duplicated()]
    display_columns = [
        col for col in [
            'Variedad',
            'Temporada',
            'Fecha',
            'GrupoID',
            'Cluster',
            'Suma_Bandas',
            'Banda_BRIX',
            'Banda_Firmeza_Punto',
            'Banda_Mejillas',
            'Banda_Acidez',
            'Muestras',
            'Frutos',
            'Localidades',
            'Campos',
            'Campo',
            'Portainjerto',
        ] if col in timeline_display.columns
    ]
    color_cluster_func = get_cluster_style_function()
    data_to_show = timeline_display[display_columns]
    if 'Cluster' in data_to_show.columns:
        st.dataframe(
            data_to_show.style.map(color_cluster_func, subset=['Cluster']),
            use_container_width=True,
        )
    else:
        st.dataframe(data_to_show, use_container_width=True)

    cluster_summary = timeline_df.groupby(['Variedad_Nombre', 'Cluster'], dropna=False).agg(
        Grupos=('GrupoID', 'nunique'),
        Temporadas=('Temporada', 'nunique'),
        Fechas=('Fecha', 'nunique'),
        Promedio_Bandas=('Suma_Bandas', 'mean'),
    ).reset_index()
    cluster_summary = cluster_summary.rename(columns={'Variedad_Nombre': 'Variedad'})
    st.markdown("#### Resumen por cluster")
    st.dataframe(cluster_summary.round(2), use_container_width=True)

    if 'Fecha_dt' in timeline_df.columns and timeline_df['Fecha_dt'].notna().any():
        timeline_chart_df = timeline_df.dropna(subset=['Fecha_dt']).copy()
        timeline_chart_df['Variedad Label'] = timeline_chart_df['Variedad_Nombre'].astype(str)
        fig_timeline = px.line(
            timeline_chart_df,
            x='Fecha_dt',
            y='Cluster',
            color='Variedad Label',
            markers=True,
            hover_data=['Temporada', 'GrupoID', 'GrupoKey', 'Campo', 'Portainjerto'],
            title='Evoluci?n de clusters por fecha',
        )
        fig_timeline.update_yaxes(dtick=1, title_text='Cluster')
        fig_timeline.update_xaxes(title_text='Fecha')
        st.plotly_chart(fig_timeline, use_container_width=True)
else:
    st.info("No se encontraron grupos agregados para las variedades seleccionadas.")

# Tabla resumen por variedad
if len(variedades_seleccionadas) > 1:
    st.markdown("### 📋 **Resumen por Variedad**")
    variedad_summary = []
    for var_clean in variedades_seleccionadas:
        var_original = variedades_mapping[var_clean]
        datos_var = datos_variedades[datos_variedades[VAR_COLUMN] == var_original]
        if not datos_var.empty:
            variedad_summary.append({
                'Variedad': var_clean,
                'Muestras': len(datos_var),
                'Temporadas': datos_var['harvest_period'].nunique() if 'harvest_period' in datos_var.columns else 0,
                'Localidades': datos_var[LOCALIDAD_COLUMN].nunique() if LOCALIDAD_COLUMN in datos_var.columns else 0,
                'Campos': datos_var[CAMPO_COLUMN].nunique() if CAMPO_COLUMN in datos_var.columns else 0,
                'BRIX_Promedio': datos_var[COL_BRIX].mean() if COL_BRIX in datos_var.columns else np.nan,
                'BRIX_Min': datos_var[COL_BRIX].min() if COL_BRIX in datos_var.columns else np.nan,
                'BRIX_Max': datos_var[COL_BRIX].max() if COL_BRIX in datos_var.columns else np.nan,
                'Acidez_Promedio': datos_var[COL_ACIDEZ].mean() if COL_ACIDEZ in datos_var.columns else np.nan,
                'Firmeza_Promedio': datos_var['Firmeza punto valor'].mean() if 'Firmeza punto valor' in datos_var.columns else np.nan
            })
    
    if variedad_summary:
        df_var_summary = pd.DataFrame(variedad_summary).round(2)
        st.dataframe(df_var_summary, use_container_width=True)

# Evolución de métricas por temporada
st.markdown("---")
st.markdown("### 📈 Evolución de Métricas por Temporada")

if len(temporadas) > 1:
    # Usar colores unificados del sistema
    # Usar paleta de colores pastel estándar
    cluster_color_map = get_plotly_color_map()
    colors = get_cluster_colors()
    cluster_colors = colors['plotly']  # Usar colores para Plotly
    
    # Preparar datos para gráficos - ahora para múltiples variedades
    evolution_data = []
    incomplete_evolution = []
    for var_clean in variedades_seleccionadas:
        var_original = variedades_mapping[var_clean]
        datos_var = datos_variedades[datos_variedades[VAR_COLUMN] == var_original]
        agg_var = agg_variedades[agg_variedades[VAR_COLUMN] == var_original]
        
        for temporada in temporadas:
            temp_data = datos_var[datos_var['harvest_period'] == temporada] if 'harvest_period' in datos_var.columns else datos_var
            agg_temp = agg_var[agg_var['harvest_period'] == temporada] if 'harvest_period' in agg_var.columns else agg_var
            metric_columns = [COL_BRIX, COL_ACIDEZ, 'Firmeza punto valor']
            if not all(col in temp_data.columns for col in metric_columns):
                incomplete_evolution.append((var_clean, temporada, 'Faltan columnas de metricas'))
                continue
            temp_metrics = temp_data.dropna(subset=metric_columns)
            if temp_metrics.empty:
                incomplete_evolution.append((var_clean, temporada, 'Registros sin metricas completas'))
                continue

        
        if not temp_metrics.empty:
            # Contar localidades y campos por temporada
            localidades_temp = temp_metrics[LOCALIDAD_COLUMN].nunique() if LOCALIDAD_COLUMN in temp_metrics.columns else 0
            campos_temp = temp_metrics[CAMPO_COLUMN].nunique() if CAMPO_COLUMN in temp_metrics.columns else 0
            
            row = {
                'Variedad': var_clean,
                'Variedad_Nombre': var_original,
                'Temporada': temporada,
                'A??o': temporada.split('_')[-1] if '_' in temporada else temporada,
                'Muestras': len(temp_metrics),
                'Frutos': temp_metrics[FRUTO_COLUMN].nunique() if FRUTO_COLUMN in temp_metrics.columns else 0,
                'Localidades': localidades_temp,
                'Campos': campos_temp,
                'BRIX_Promedio': temp_metrics[COL_BRIX].mean() if COL_BRIX in temp_metrics.columns else np.nan,
                'BRIX_Std': temp_metrics[COL_BRIX].std() if COL_BRIX in temp_metrics.columns else np.nan,
                'BRIX_Min': temp_metrics[COL_BRIX].min() if COL_BRIX in temp_metrics.columns else np.nan,
                'BRIX_Max': temp_metrics[COL_BRIX].max() if COL_BRIX in temp_metrics.columns else np.nan,
                'Acidez_Promedio': temp_metrics[COL_ACIDEZ].mean() if COL_ACIDEZ in temp_metrics.columns else np.nan,
                'Acidez_Std': temp_metrics[COL_ACIDEZ].std() if COL_ACIDEZ in temp_metrics.columns else np.nan,
                'Firmeza_Promedio': temp_metrics['Firmeza punto valor'].mean() if 'Firmeza punto valor' in temp_metrics.columns else np.nan,
                'Firmeza_Min': temp_metrics['Firmeza punto valor'].min() if 'Firmeza punto valor' in temp_metrics.columns else np.nan,
                'Cluster': agg_temp['cluster_grp'].iloc[0] if 'cluster_grp' in agg_temp.columns and not agg_temp.empty else np.nan,
                'Banda_BRIX': agg_temp['banda_brix'].iloc[0] if 'banda_brix' in agg_temp.columns and not agg_temp.empty else np.nan,
                'Banda_Firmeza': (
                    agg_temp['banda_firmeza_punto'].iloc[0]
                    if 'banda_firmeza_punto' in agg_temp.columns and not agg_temp.empty
                    else (agg_temp['banda_firmeza'].iloc[0] if 'banda_firmeza' in agg_temp.columns and not agg_temp.empty else np.nan)
                ),
                'Banda_Mejillas': agg_temp['banda_mejillas'].iloc[0] if 'banda_mejillas' in agg_temp.columns and not agg_temp.empty else np.nan,
                'Banda_Acidez': agg_temp['banda_acidez'].iloc[0] if 'banda_acidez' in agg_temp.columns and not agg_temp.empty else np.nan,
                'Suma_Bandas': agg_temp['suma_bandas'].iloc[0] if 'suma_bandas' in agg_temp.columns and not agg_temp.empty else np.nan,
                'GrupoID': agg_temp['grupo_id'].iloc[0] if 'grupo_id' in agg_temp.columns and not agg_temp.empty else np.nan,
                'GrupoKey': agg_temp['grupo_key'].iloc[0] if 'grupo_key' in agg_temp.columns and not agg_temp.empty else np.nan,
                'GrupoKeyDetalle': agg_temp['grupo_key_detalle'].iloc[0] if 'grupo_key_detalle' in agg_temp.columns and not agg_temp.empty else np.nan,
                'Fecha_Grupo': agg_temp[fecha_col].iloc[0] if fecha_col and fecha_col in agg_temp.columns and not agg_temp.empty else np.nan,
            }
            evolution_data.append(row)
    
    if incomplete_evolution:
        missing_df = pd.DataFrame(incomplete_evolution, columns=['Variedad', 'Temporada', 'Motivo']).drop_duplicates()
        st.warning('Se omitieron combinaciones sin metricas completas. Revisa la tabla para validar los datos faltantes.')
        st.dataframe(missing_df.sort_values(['Variedad', 'Temporada']), use_container_width=True)

    if evolution_data:
        df_evolution = pd.DataFrame(evolution_data).sort_values('Temporada')
        
        # Gráfico principal de evolución - mejorado para múltiples variedades
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['BRIX por Temporada y Variedad', 'Acidez por Temporada y Variedad', 'Firmeza por Temporada y Variedad', 'Evolución de Cluster'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Colores para cada variedad
        # Usar secuencia de colores pastel para variedades
        base_colors = get_plotly_color_sequence()
        colors_variedades = (base_colors * ((len(variedades_seleccionadas) // 4) + 1))[:len(variedades_seleccionadas)]
        
        # BRIX - una línea por variedad
        for i, variedad in enumerate(variedades_seleccionadas):
            df_var = df_evolution[df_evolution['Variedad'] == variedad]
            fig.add_trace(
                go.Scatter(x=df_var['Temporada'], y=df_var['BRIX_Promedio'],
                          mode='lines+markers', name=f'{variedad} - BRIX',
                          line=dict(color=colors_variedades[i], width=3),
                          error_y=dict(type='data', array=df_var['BRIX_Std'], visible=True)),
                row=1, col=1
            )
        
        # Acidez - una línea por variedad
        for i, variedad in enumerate(variedades_seleccionadas):
            df_var = df_evolution[df_evolution['Variedad'] == variedad]
            fig.add_trace(
                go.Scatter(x=df_var['Temporada'], y=df_var['Acidez_Promedio'],
                          mode='lines+markers', name=f'{variedad} - Acidez',
                          line=dict(color=colors_variedades[i], width=3, dash='dot'),
                          error_y=dict(type='data', array=df_var['Acidez_Std'], visible=True)),
                row=1, col=2
            )
        
        # Firmeza - una línea por variedad
        for i, variedad in enumerate(variedades_seleccionadas):
            df_var = df_evolution[df_evolution['Variedad'] == variedad]
            fig.add_trace(
                go.Scatter(x=df_var['Temporada'], y=df_var['Firmeza_Promedio'],
                          mode='lines+markers', name=f'{variedad} - Firmeza Prom',
                          line=dict(color=colors_variedades[i], width=3, dash='dashdot')),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df_var['Temporada'], y=df_var['Firmeza_Min'],
                          mode='lines+markers', name=f'{variedad} - Firmeza Min',
                          line=dict(color=colors_variedades[i], width=2, dash='dash'),
                          showlegend=False),
                row=2, col=1
            )
        
        # Cluster - barras agrupadas por variedad
        for i, variedad in enumerate(variedades_seleccionadas):
            df_var = df_evolution[df_evolution['Variedad'] == variedad]
            colors = [cluster_colors.get(int(c), '#cccccc') if not pd.isna(c) else '#cccccc' for c in df_var['Cluster']]
            
            fig.add_trace(
                go.Bar(x=df_var['Temporada'], y=df_var['Cluster'],
                       name=f'{variedad} - Cluster', 
                       marker_color=colors_variedades[i],
                       opacity=0.7,
                       offsetgroup=i),
                row=2, col=2
            )
        
        if len(variedades_seleccionadas) == 1:
            title_text = f"Evolución de {variedades_seleccionadas[0]}"
        else:
            title_text = f"Comparación Evolutiva: {len(variedades_seleccionadas)} Variedades"
        
        fig.update_layout(height=900, showlegend=True, title_text=title_text, barmode='group')
        fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de evolución
        st.markdown("#### 📋 Tabla de Evolución por Temporada y Variedad")
        display_columns = [
            'Variedad_Nombre', 'Variedad', 'Temporada', 'Fecha_Grupo', 'GrupoID', 'GrupoKey', 'GrupoKeyDetalle',
            'Muestras', 'Frutos', 'Localidades', 'Campos',
            'BRIX_Promedio', 'BRIX_Min', 'BRIX_Max', 'Acidez_Promedio',
            'Firmeza_Promedio', 'Banda_BRIX', 'Banda_Firmeza', 'Banda_Mejillas', 'Banda_Acidez', 'Suma_Bandas', 'Cluster'
        ]
        display_columns = [col for col in display_columns if col in df_evolution.columns]
        display_df = df_evolution[display_columns].round(2)
        display_df = display_df.rename(columns={
            'Variedad_Nombre': 'Variedad Nombre',
            'Fecha_Grupo': 'Fecha Grupo'
        })

        # Usar colores unificados para colorear clusters
        color_cluster_func = get_cluster_style_function()
        styled_df = display_df.style.map(
            color_cluster_func, subset=['Cluster'] if 'Cluster' in display_df.columns else []
        )
        st.dataframe(styled_df, use_container_width=True)
        
        # Gráfico de comparación directa entre variedades (si hay múltiples)
        if len(variedades_seleccionadas) > 1:
            st.markdown("#### 🔄 Comparación Directa entre Variedades")
            
            # Crear gráfico de barras comparativo
            comparison_fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=['BRIX Promedio', 'Acidez Promedio', 'Firmeza Promedio']
            )
            
            for i, variedad in enumerate(variedades_seleccionadas):
                df_var = df_evolution[df_evolution['Variedad'] == variedad]
                
                # BRIX
                comparison_fig.add_trace(
                    go.Box(y=df_var['BRIX_Promedio'], name=f'{variedad}', 
                           marker_color=colors_variedades[i], showlegend=i==0),
                    row=1, col=1
                )
                
                # Acidez
                comparison_fig.add_trace(
                    go.Box(y=df_var['Acidez_Promedio'], name=f'{variedad}', 
                           marker_color=colors_variedades[i], showlegend=False),
                    row=1, col=2
                )
                
                # Firmeza
                comparison_fig.add_trace(
                    go.Box(y=df_var['Firmeza_Promedio'], name=f'{variedad}', 
                           marker_color=colors_variedades[i], showlegend=False),
                    row=1, col=3
                )
            
            comparison_fig.update_layout(height=500, title_text="Distribución de Métricas por Variedad")
            st.plotly_chart(comparison_fig, use_container_width=True)

        # Análisis de tendencias
        st.markdown("---")
        st.markdown("### 🔍 Análisis de Tendencias por Variedad")
        
        # Calcular tendencias
        def calculate_trend(values):
            """Calcula la tendencia (mejora, empeora, estable)"""
            valid_values = [v for v in values if not pd.isna(v)]
            if len(valid_values) < 2:
                return "Sin datos suficientes", 0
            
            first_val = valid_values[0]
            last_val = valid_values[-1]
            change = last_val - first_val
            change_pct = (change / first_val) * 100 if first_val != 0 else 0
            
            if abs(change_pct) < 5:
                return "Estable", change_pct
            elif change_pct > 0:
                return "Mejora", change_pct
            else:
                return "Empeora", change_pct
        
        # Análisis para cada variedad
        for variedad in variedades_seleccionadas:
            st.markdown(f"#### 🌱 **{variedad}**")
            
            df_var = df_evolution[df_evolution['Variedad'] == variedad]
            
            col1, col2, col3, col4 = st.columns(4)
        
            with col1:
                brix_trend, brix_change = calculate_trend(df_var['BRIX_Promedio'].tolist())
                if brix_trend == "Mejora":
                    st.markdown(f'<div class="improvement-box"><h4>🍇 BRIX</h4><p><strong>{brix_trend}</strong><br>Cambio: +{brix_change:.1f}%</p></div>', unsafe_allow_html=True)
                elif brix_trend == "Empeora":
                    st.markdown(f'<div class="degradation-box"><h4>🍇 BRIX</h4><p><strong>{brix_trend}</strong><br>Cambio: {brix_change:.1f}%</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="stable-box"><h4>🍇 BRIX</h4><p><strong>{brix_trend}</strong><br>Cambio: {brix_change:.1f}%</p></div>', unsafe_allow_html=True)
        
            with col2:
                acidez_trend, acidez_change = calculate_trend(df_var['Acidez_Promedio'].tolist())
                if acidez_trend == "Mejora":
                    st.markdown(f'<div class="improvement-box"><h4>🧪 Acidez</h4><p><strong>{acidez_trend}</strong><br>Cambio: +{acidez_change:.1f}%</p></div>', unsafe_allow_html=True)
                elif acidez_trend == "Empeora":
                    st.markdown(f'<div class="degradation-box"><h4>🧪 Acidez</h4><p><strong>{acidez_trend}</strong><br>Cambio: {acidez_change:.1f}%</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="stable-box"><h4>🧪 Acidez</h4><p><strong>{acidez_trend}</strong><br>Cambio: {acidez_change:.1f}%</p></div>', unsafe_allow_html=True)
        
            with col3:
                firmeza_trend, firmeza_change = calculate_trend(df_var['Firmeza_Promedio'].tolist())
                if firmeza_trend == "Mejora":
                    st.markdown(f'<div class="improvement-box"><h4>💪 Firmeza</h4><p><strong>{firmeza_trend}</strong><br>Cambio: +{firmeza_change:.1f}%</p></div>', unsafe_allow_html=True)
                elif firmeza_trend == "Empeora":
                    st.markdown(f'<div class="degradation-box"><h4>💪 Firmeza</h4><p><strong>{firmeza_trend}</strong><br>Cambio: {firmeza_change:.1f}%</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="stable-box"><h4>💪 Firmeza</h4><p><strong>{firmeza_trend}</strong><br>Cambio: {firmeza_change:.1f}%</p></div>', unsafe_allow_html=True)
        
            with col4:
                cluster_trend, cluster_change = calculate_trend(df_var['Cluster'].tolist())
                if cluster_trend == "Mejora":
                    st.markdown(f'<div class="improvement-box"><h4>🏆 Cluster</h4><p><strong>Mejor calidad</strong><br>Cambio: {cluster_change:.1f}</p></div>', unsafe_allow_html=True)
                elif cluster_trend == "Empeora":
                    st.markdown(f'<div class="degradation-box"><h4>🏆 Cluster</h4><p><strong>Peor calidad</strong><br>Cambio: {cluster_change:.1f}</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="stable-box"><h4>🏆 Cluster</h4><p><strong>Calidad estable</strong><br>Cambio: {cluster_change:.1f}</p></div>', unsafe_allow_html=True)
            
            st.markdown("---")  # Separador entre variedades

    # Desglose por localidad y campo
    st.markdown("---")
    st.markdown("### 🌍 Desglose por Localidad y Campo")
    
    if LOCALIDAD_COLUMN in datos_variedad.columns and CAMPO_COLUMN in datos_variedad.columns:
        # Tabla resumen por temporada-localidad-campo
        location_summary = []
        for temporada in temporadas:
            temp_metrics = datos_variedad[datos_variedad['harvest_period'] == temporada]
            for localidad in temp_metrics[LOCALIDAD_COLUMN].unique():
                loc_data = temp_metrics[temp_metrics[LOCALIDAD_COLUMN] == localidad]
                for campo in loc_data[CAMPO_COLUMN].unique():
                    campo_data = loc_data[loc_data[CAMPO_COLUMN] == campo]
                    if not campo_data.empty:
                        location_summary.append({
                            'Temporada': temporada,
                            'Localidad': localidad,
                            'Campo': campo,
                            'Muestras': len(campo_data),
                            'Frutos': campo_data[FRUTO_COLUMN].nunique() if FRUTO_COLUMN in campo_data.columns else 0,
                            'BRIX_Promedio': campo_data[COL_BRIX].mean() if COL_BRIX in campo_data.columns else np.nan,
                            'BRIX_Min': campo_data[COL_BRIX].min() if COL_BRIX in campo_data.columns else np.nan,
                            'BRIX_Max': campo_data[COL_BRIX].max() if COL_BRIX in campo_data.columns else np.nan,
                            'Acidez_Promedio': campo_data[COL_ACIDEZ].mean() if COL_ACIDEZ in campo_data.columns else np.nan,
                            'Firmeza_Promedio': campo_data['Firmeza punto valor'].mean() if 'Firmeza punto valor' in campo_data.columns else np.nan
                        })
        
        if location_summary:
            df_locations = pd.DataFrame(location_summary)
            st.markdown("#### 📋 Resumen por Temporada-Localidad-Campo")
            st.dataframe(df_locations.round(2), use_container_width=True, height=400)
            
            # Gráfico de BRIX por localidad
            if COL_BRIX in datos_variedad.columns:
                fig_brix_loc = px.box(
                    datos_variedad, 
                    x=LOCALIDAD_COLUMN, 
                    y=COL_BRIX,
                    color='harvest_period',
                    title=f'Distribución de BRIX por Localidad - {variedad_seleccionada}',
                    labels={COL_BRIX: 'BRIX (%)', LOCALIDAD_COLUMN: 'Localidad'}
                )
                fig_brix_loc.update_layout(height=500)
                st.plotly_chart(fig_brix_loc, use_container_width=True)
    else:
        st.info("📍 No hay información de localidad y campo disponible en los datos.")
        
else:
    st.warning("⚠️ Esta variedad solo tiene datos de una temporada. No se puede mostrar evolución temporal.")
    
    # Aún así, mostrar información de localidades si está disponible
    if LOCALIDAD_COLUMN in datos_variedad.columns and CAMPO_COLUMN in datos_variedad.columns:
        st.markdown("### 🌍 Información de Localidades y Campos")
        location_summary = []
        for localidad in datos_variedad[LOCALIDAD_COLUMN].unique():
            loc_data = datos_variedad[datos_variedad[LOCALIDAD_COLUMN] == localidad]
            for campo in loc_data[CAMPO_COLUMN].unique():
                campo_data = loc_data[loc_data[CAMPO_COLUMN] == campo]
                if not campo_data.empty:
                    location_summary.append({
                        'Localidad': localidad,
                        'Campo': campo,
                        'Muestras': len(campo_data),
                        'Frutos': campo_data[FRUTO_COLUMN].nunique() if FRUTO_COLUMN in campo_data.columns else 0,
                        'BRIX_Promedio': campo_data[COL_BRIX].mean() if COL_BRIX in campo_data.columns else np.nan,
                        'BRIX_Min': campo_data[COL_BRIX].min() if COL_BRIX in campo_data.columns else np.nan,
                        'BRIX_Max': campo_data[COL_BRIX].max() if COL_BRIX in campo_data.columns else np.nan
                    })
        
        if location_summary:
            df_locations = pd.DataFrame(location_summary)
            st.dataframe(df_locations.round(2), use_container_width=True)

# Valores individuales por temporada
st.markdown("---")
st.markdown("### 🔬 Valores Individuales por Temporada")

temporada_seleccionada = st.selectbox(
    "Selecciona una temporada para ver valores individuales:",
    options=temporadas,
    key="temporada_individual"
)

datos_temporada = datos_variedad[datos_variedad['harvest_period'] == temporada_seleccionada] if 'harvest_period' in datos_variedad.columns else datos_variedad

if not datos_temporada.empty:
    st.markdown(f"#### 📊 Muestras individuales - {temporada_seleccionada}")
    
    # Información de la temporada
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Muestras", len(datos_temporada))
    with col2:
        st.metric("Frutos", datos_temporada[FRUTO_COLUMN].nunique() if FRUTO_COLUMN in datos_temporada.columns else 0)
    with col3:
        brix_prom = datos_temporada[COL_BRIX].mean() if COL_BRIX in datos_temporada.columns else 0
        st.metric("BRIX Promedio", f"{brix_prom:.2f}")
    with col4:
        cluster_temp = agg_variedad[agg_variedad['harvest_period'] == temporada_seleccionada]['cluster_grp'].iloc[0] if 'harvest_period' in agg_variedad.columns and 'cluster_grp' in agg_variedad.columns else "N/A"
        st.metric("Cluster Asignado", cluster_temp)
    
    # Segunda fila con información de localidades y campos
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        localidades_temp = datos_temporada[LOCALIDAD_COLUMN].nunique() if LOCALIDAD_COLUMN in datos_temporada.columns else 0
        st.metric("Localidades", localidades_temp)
    with col2:
        campos_temp = datos_temporada[CAMPO_COLUMN].nunique() if CAMPO_COLUMN in datos_temporada.columns else 0
        st.metric("Campos", campos_temp)
    with col3:
        if COL_BRIX in datos_temporada.columns:
            brix_min = datos_temporada[COL_BRIX].min()
            brix_max = datos_temporada[COL_BRIX].max()
            st.metric("Rango BRIX", f"{brix_min:.1f} - {brix_max:.1f}")
        else:
            st.metric("Rango BRIX", "N/A")
    with col4:
        if COL_ACIDEZ in datos_temporada.columns:
            acidez_prom = datos_temporada[COL_ACIDEZ].mean()
            st.metric("Acidez Promedio", f"{acidez_prom:.2f}")
        else:
            st.metric("Acidez Promedio", "N/A")
    
    # Gráfico de dispersión de valores individuales
    if COL_BRIX in datos_temporada.columns and 'Firmeza punto valor' in datos_temporada.columns:
        fig_scatter = px.scatter(
            datos_temporada, 
            x=COL_BRIX, 
            y='Firmeza punto valor',
            color=FRUTO_COLUMN,
            title=f'BRIX vs Firmeza - {temporada_seleccionada}',
            labels={COL_BRIX: 'BRIX (%)', 'Firmeza punto valor': 'Firmeza'},
            hover_data=[FRUTO_COLUMN]
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Tabla de valores individuales con localidad y campo
    columnas_mostrar = [col for col in [FRUTO_COLUMN, LOCALIDAD_COLUMN, CAMPO_COLUMN, COL_BRIX, COL_ACIDEZ, 'Firmeza punto valor'] 
                       if col in datos_temporada.columns]
    
    if columnas_mostrar:
        st.markdown("#### 📋 Tabla de valores individuales")
        datos_display = datos_temporada[columnas_mostrar].round(2)
        st.dataframe(datos_display, use_container_width=True, height=400)
        
        # Estadísticas descriptivas
        st.markdown("#### 📊 Estadísticas descriptivas")
        stats_cols = [col for col in [COL_BRIX, COL_ACIDEZ, 'Firmeza punto valor'] if col in datos_temporada.columns]
        if stats_cols:
            stats_df = datos_temporada[stats_cols].describe().round(2)
            st.dataframe(stats_df, use_container_width=True)
        
        # Estadísticas por localidad y campo para la temporada
        if LOCALIDAD_COLUMN in datos_temporada.columns and CAMPO_COLUMN in datos_temporada.columns:
            st.markdown("#### 🌍 Estadísticas por Localidad y Campo")
            
            # Agrupar por localidad
            localidad_stats = datos_temporada.groupby(LOCALIDAD_COLUMN).agg({
                FRUTO_COLUMN: 'nunique',
                COL_BRIX: ['count', 'mean', 'min', 'max', 'std'] if COL_BRIX in datos_temporada.columns else 'count',
                COL_ACIDEZ: ['mean', 'std'] if COL_ACIDEZ in datos_temporada.columns else 'count',
                'Firmeza punto valor': ['mean', 'min', 'max'] if 'Firmeza punto valor' in datos_temporada.columns else 'count'
            }).round(2)
            
            localidad_stats.columns = ['_'.join(col).strip() for col in localidad_stats.columns.values]
            localidad_stats = localidad_stats.reset_index()
            
            st.dataframe(localidad_stats, use_container_width=True)
            
            # Gráfico comparativo de BRIX por localidad
            if COL_BRIX in datos_temporada.columns and len(datos_temporada[LOCALIDAD_COLUMN].unique()) > 1:
                fig_comparison = px.violin(
                    datos_temporada, 
                    x=LOCALIDAD_COLUMN, 
                    y=COL_BRIX,
                    title=f'Distribución de BRIX por Localidad - {temporada_seleccionada}',
                    labels={COL_BRIX: 'BRIX (%)', LOCALIDAD_COLUMN: 'Localidad'}
                )
                fig_comparison.update_layout(height=400)
                st.plotly_chart(fig_comparison, use_container_width=True)

# Comparación entre temporadas
if len(temporadas) > 1:
    st.markdown("---")
    st.markdown("### ⚖️ Comparación Directa entre Temporadas")
    
    col1, col2 = st.columns(2)
    with col1:
        temp1 = st.selectbox("Primera temporada:", options=temporadas, key="temp1")
    with col2:
        temp2 = st.selectbox("Segunda temporada:", options=[t for t in temporadas if t != temp1], key="temp2")
    
    if temp1 and temp2:
        datos_temp1 = datos_variedad[datos_variedad['harvest_period'] == temp1]
        datos_temp2 = datos_variedad[datos_variedad['harvest_period'] == temp2]
        
        # Comparación lado a lado para múltiples variedades
        st.markdown("#### 🔄 Comparación por Variedad")
        
        for variedad in variedades_seleccionadas:
            var_original = variedades_mapping[variedad]
            var_temp1 = datos_temp1[datos_temp1[VAR_COLUMN] == var_original]
            var_temp2 = datos_temp2[datos_temp2[VAR_COLUMN] == var_original]
            
            if not var_temp1.empty and not var_temp2.empty:
                st.markdown(f"##### 🌱 {variedad}")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**{temp1}**")
                    st.metric("Muestras", len(var_temp1))
                    st.metric("BRIX Promedio", f"{var_temp1[COL_BRIX].mean():.2f}" if COL_BRIX in var_temp1.columns else "N/A")
                    st.metric("Firmeza Promedio", f"{var_temp1['Firmeza punto valor'].mean():.2f}" if 'Firmeza punto valor' in var_temp1.columns else "N/A")
        
                with col2:
                    st.markdown(f"**{temp2}**")
                    st.metric("Muestras", len(var_temp2))
                    
                    brix_t2 = var_temp2[COL_BRIX].mean() if COL_BRIX in var_temp2.columns else 0
                    brix_t1 = var_temp1[COL_BRIX].mean() if COL_BRIX in var_temp1.columns else 0
                    delta_brix = brix_t2 - brix_t1
                    st.metric("BRIX Promedio", f"{brix_t2:.2f}", f"{delta_brix:+.2f}" if abs(delta_brix) > 0.01 else None)
                    
                    firm_t2 = var_temp2['Firmeza punto valor'].mean() if 'Firmeza punto valor' in var_temp2.columns else 0
                    firm_t1 = var_temp1['Firmeza punto valor'].mean() if 'Firmeza punto valor' in var_temp1.columns else 0
                    delta_firm = firm_t2 - firm_t1
                    st.metric("Firmeza Promedio", f"{firm_t2:.2f}", f"{delta_firm:+.2f}" if abs(delta_firm) > 0.01 else None)
                    
                st.markdown("---")  # Separador entre variedades
        
        # Tabla comparativa detallada por variedad y temporada
        st.markdown("#### 📋 Comparación detallada entre temporadas")
        
        comparison_data = []
        for variedad in variedades_seleccionadas:
            var_original = variedades_mapping[variedad]
            for temp_name, temp_metrics in [(temp1, datos_temp1), (temp2, datos_temp2)]:
                var_temp_metrics = temp_metrics[temp_metrics[VAR_COLUMN] == var_original]
                if not var_temp_metrics.empty:
                    row = {
                        'Variedad': variedad,
                        'Temporada': temp_name,
                        'Muestras': len(var_temp_metrics),
                        'Frutos': var_temp_data[FRUTO_COLUMN].nunique() if FRUTO_COLUMN in var_temp_data.columns else 0,
                        'Localidades': var_temp_data[LOCALIDAD_COLUMN].nunique() if LOCALIDAD_COLUMN in var_temp_data.columns else 0,
                        'Campos': var_temp_data[CAMPO_COLUMN].nunique() if CAMPO_COLUMN in var_temp_data.columns else 0,
                        'BRIX_Promedio': var_temp_data[COL_BRIX].mean() if COL_BRIX in var_temp_data.columns else np.nan,
                        'BRIX_Min': var_temp_data[COL_BRIX].min() if COL_BRIX in var_temp_data.columns else np.nan,
                        'BRIX_Max': var_temp_data[COL_BRIX].max() if COL_BRIX in var_temp_data.columns else np.nan,
                        'Acidez_Promedio': var_temp_data[COL_ACIDEZ].mean() if COL_ACIDEZ in var_temp_data.columns else np.nan,
                        'Firmeza_Promedio': var_temp_data['Firmeza punto valor'].mean() if 'Firmeza punto valor' in var_temp_data.columns else np.nan
                    }
                    comparison_data.append(row)
        
        if len(comparison_data) == 2:
            df_comparison = pd.DataFrame(comparison_data).round(2)
            st.dataframe(df_comparison, use_container_width=True)

# Exportar datos
st.markdown("---")
if st.button("📥 Exportar análisis de evolución a Excel"):
    # Crear archivo Excel con múltiples hojas
    output = pd.ExcelWriter('evolucion_variedad.xlsx', engine='xlsxwriter')
    
    # Hoja 1: Datos de evolución por temporada
    if 'df_evolution' in locals():
        df_evolution.to_excel(output, sheet_name='Evolucion_Temporadas', index=False)
    
    # Hoja 2: Datos agregados por variedad
    agg_variedades.to_excel(output, sheet_name='Datos_Agregados', index=False)
    
    # Hoja 3: Detalle por grupo y fecha
    if 'timeline_df' in locals() and not timeline_df.empty:
        timeline_export = timeline_df.copy()
        if 'Fecha_dt' in timeline_export.columns:
            timeline_export['Fecha_dt'] = timeline_export['Fecha_dt'].astype(str)
        timeline_export.to_excel(output, sheet_name='Detalle_Grupos', index=False)

    # Hoja 4: Datos crudos de las variedades
    datos_variedades.to_excel(output, sheet_name='Datos_Crudos', index=False)
    # Hoja 5: Brix combinatoria por llaves
    if 'brix_combo_export_df' in locals() and not brix_combo_export_df.empty:
        export_brix_combo = brix_combo_export_df.copy()
        if 'Fecha' in export_brix_combo.columns:
            export_brix_combo['Fecha'] = pd.to_datetime(export_brix_combo['Fecha'], errors='coerce').astype(str)
        export_brix_combo.to_excel(output, sheet_name='Brix_Combinatoria', index=False)

    
    # Hoja 6: Desglose por localidad y campo (si existe)
    if 'df_locations' in locals():
        df_locations.to_excel(output, sheet_name='Localidad_Campo', index=False)
    
    output.close()
    
    with open('evolucion_variedad.xlsx', 'rb') as f:
        st.download_button(
            label="Descargar archivo Excel",
            data=f.read(),
            file_name=f'evolucion_{'_'.join(variedades_seleccionadas[:3])}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
