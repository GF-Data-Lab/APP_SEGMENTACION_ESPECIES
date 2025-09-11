import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import show_logo

# Configuración de página
st.set_page_config(
    page_title="📈 Evolución de Variedad", 
    page_icon="📈", 
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
      
      .metric-evolution-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 5px solid #2196f3;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      }
      
      .season-comparison-box {
        background-color: #e8f5e8;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4caf50;
      }
      
      .improvement-box {
        background-color: #e8f5e8;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4caf50;
      }
      
      .degradation-box {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #f44336;
      }
      
      .stable-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #ff9800;
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
        if st.button('Evolución Variedad 📈'):
            st.switch_page('pages/evolucion_variedad.py')

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
ESPECIE_COLUMN = "Especie"
VAR_COLUMN = "Variedad"
FRUTO_COLUMN = "N° Muestra"
COL_BRIX = "Sólidos Solubles (%)"
COL_ACIDEZ = "Acidez (%)"
LOCALIDAD_COLUMN = "Localidad"
CAMPO_COLUMN = "Campo"

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
    años_únicos = datos_variedades['harvest_period'].str.extract('(\d{4})').dropna().nunique() if 'harvest_period' in datos_variedades.columns else 0
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
    # Colores para clusters
    cluster_colors = {
        1: '#a8e6cf',  # verde claro - Excelente
        2: '#ffd3b6',  # naranja claro - Bueno
        3: '#ffaaa5',  # coral - Regular
        4: '#ff8b94',  # rojo rosado - Deficiente
    }
    
    # Preparar datos para gráficos - ahora para múltiples variedades
    evolution_data = []
    for var_clean in variedades_seleccionadas:
        var_original = variedades_mapping[var_clean]
        datos_var = datos_variedades[datos_variedades[VAR_COLUMN] == var_original]
        agg_var = agg_variedades[agg_variedades[VAR_COLUMN] == var_original]
        
        for temporada in temporadas:
            temp_data = datos_var[datos_var['harvest_period'] == temporada] if 'harvest_period' in datos_var.columns else datos_var
            agg_temp = agg_var[agg_var['harvest_period'] == temporada] if 'harvest_period' in agg_var.columns else agg_var
        
        if not temp_data.empty:
            # Contar localidades y campos por temporada
            localidades_temp = temp_data[LOCALIDAD_COLUMN].nunique() if LOCALIDAD_COLUMN in temp_data.columns else 0
            campos_temp = temp_data[CAMPO_COLUMN].nunique() if CAMPO_COLUMN in temp_data.columns else 0
            
            row = {
                'Variedad': var_clean,
                'Temporada': temporada,
                'Año': temporada.split('_')[-1] if '_' in temporada else temporada,
                'Muestras': len(temp_data),
                'Frutos': temp_data[FRUTO_COLUMN].nunique() if FRUTO_COLUMN in temp_data.columns else 0,
                'Localidades': localidades_temp,
                'Campos': campos_temp,
                'BRIX_Promedio': temp_data[COL_BRIX].mean() if COL_BRIX in temp_data.columns else np.nan,
                'BRIX_Std': temp_data[COL_BRIX].std() if COL_BRIX in temp_data.columns else np.nan,
                'BRIX_Min': temp_data[COL_BRIX].min() if COL_BRIX in temp_data.columns else np.nan,
                'BRIX_Max': temp_data[COL_BRIX].max() if COL_BRIX in temp_data.columns else np.nan,
                'Acidez_Promedio': temp_data[COL_ACIDEZ].mean() if COL_ACIDEZ in temp_data.columns else np.nan,
                'Acidez_Std': temp_data[COL_ACIDEZ].std() if COL_ACIDEZ in temp_data.columns else np.nan,
                'Firmeza_Promedio': temp_data['Firmeza punto valor'].mean() if 'Firmeza punto valor' in temp_data.columns else np.nan,
                'Firmeza_Min': temp_data['Firmeza punto valor'].min() if 'Firmeza punto valor' in temp_data.columns else np.nan,
                'Cluster': agg_temp['cluster_grp'].iloc[0] if 'cluster_grp' in agg_temp.columns and not agg_temp.empty else np.nan,
                'Banda_BRIX': agg_temp['banda_brix'].iloc[0] if 'banda_brix' in agg_temp.columns and not agg_temp.empty else np.nan,
                'Banda_Firmeza': agg_temp['banda_firmeza'].iloc[0] if 'banda_firmeza' in agg_temp.columns and not agg_temp.empty else np.nan,
                'Suma_Bandas': agg_temp['suma_bandas'].iloc[0] if 'suma_bandas' in agg_temp.columns and not agg_temp.empty else np.nan,
            }
            evolution_data.append(row)
    
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
        colors_variedades = px.colors.qualitative.Set1[:len(variedades_seleccionadas)]
        
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
        display_df = df_evolution[['Variedad', 'Temporada', 'Muestras', 'Frutos', 'Localidades', 'Campos',
                                 'BRIX_Promedio', 'BRIX_Min', 'BRIX_Max', 'Acidez_Promedio', 
                                 'Firmeza_Promedio', 'Banda_BRIX', 'Banda_Firmeza', 'Suma_Bandas', 'Cluster']].round(2)
        
        # Colorear la tabla por cluster y variedad
        def color_cluster_table(row):
            cluster = row['Cluster']
            variedad = row['Variedad']
            colors_row = ['']*len(row)
            
            # Color de fondo para el cluster
            if not pd.isna(cluster):
                cluster_color = cluster_colors.get(int(cluster), '')
                if 'Cluster' in row.index:
                    colors_row[list(row.index).index('Cluster')] = f'background-color: {cluster_color}'
            
            # Color de borde para la variedad
            if 'Variedad' in row.index and variedad in variedades_seleccionadas:
                var_idx = variedades_seleccionadas.index(variedad)
                var_color = colors_variedades[var_idx] if var_idx < len(colors_variedades) else '#cccccc'
                colors_row[list(row.index).index('Variedad')] = f'background-color: {var_color}; opacity: 0.3'
            
            return colors_row
        
        styled_df = display_df.style.apply(color_cluster_table, axis=1)
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
            temp_data = datos_variedad[datos_variedad['harvest_period'] == temporada]
            for localidad in temp_data[LOCALIDAD_COLUMN].unique():
                loc_data = temp_data[temp_data[LOCALIDAD_COLUMN] == localidad]
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
            for temp_name, temp_data in [(temp1, datos_temp1), (temp2, datos_temp2)]:
                var_temp_data = temp_data[temp_data[VAR_COLUMN] == var_original]
                if not var_temp_data.empty:
                    row = {
                        'Variedad': variedad,
                        'Temporada': temp_name,
                        'Muestras': len(var_temp_data),
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
    
    # Hoja 3: Datos crudos de las variedades
    datos_variedades.to_excel(output, sheet_name='Datos_Crudos', index=False)
    
    # Hoja 4: Desglose por localidad y campo (si existe)
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