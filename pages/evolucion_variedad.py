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

# Obtener lista de variedades únicas
variedades = agg_data[VAR_COLUMN].unique() if VAR_COLUMN in agg_data.columns else []

if len(variedades) == 0:
    st.error("❌ No se encontraron variedades en los datos.")
    st.stop()

with col2:
    variedad_seleccionada = st.selectbox(
        "Selecciona la variedad a analizar:",
        options=sorted(variedades),
        key="variedad_evolucion"
    )

# Filtrar datos por variedad
datos_variedad = raw_data[raw_data[VAR_COLUMN] == variedad_seleccionada]
agg_variedad = agg_data[agg_data[VAR_COLUMN] == variedad_seleccionada]

st.markdown("---")
st.markdown(f"## 📊 Análisis evolutivo: **{variedad_seleccionada}**")

# Información general de la variedad
temporadas = sorted(agg_variedad['harvest_period'].unique()) if 'harvest_period' in agg_variedad.columns else []

col1, col2, col3, col4 = st.columns(4)
with col1:
    total_muestras = len(datos_variedad)
    st.metric("Total Muestras", total_muestras)
with col2:
    frutos_unicos = datos_variedad[FRUTO_COLUMN].nunique() if FRUTO_COLUMN in datos_variedad.columns else 0
    st.metric("Frutos Únicos", frutos_unicos)
with col3:
    st.metric("Temporadas", len(temporadas))
with col4:
    años_únicos = datos_variedad['harvest_period'].str.extract('(\d{4})').dropna().nunique() if 'harvest_period' in datos_variedad.columns else 0
    st.metric("Años de Datos", años_únicos)

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
    
    # Preparar datos para gráficos
    evolution_data = []
    for temporada in temporadas:
        temp_data = datos_variedad[datos_variedad['harvest_period'] == temporada] if 'harvest_period' in datos_variedad.columns else datos_variedad
        agg_temp = agg_variedad[agg_variedad['harvest_period'] == temporada] if 'harvest_period' in agg_variedad.columns else agg_variedad
        
        if not temp_data.empty:
            row = {
                'Temporada': temporada,
                'Año': temporada.split('_')[-1] if '_' in temporada else temporada,
                'Muestras': len(temp_data),
                'Frutos': temp_data[FRUTO_COLUMN].nunique() if FRUTO_COLUMN in temp_data.columns else 0,
                'BRIX_Promedio': temp_data[COL_BRIX].mean() if COL_BRIX in temp_data.columns else np.nan,
                'BRIX_Std': temp_data[COL_BRIX].std() if COL_BRIX in temp_data.columns else np.nan,
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
        
        # Gráfico principal de evolución
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['BRIX por Temporada', 'Acidez por Temporada', 'Firmeza por Temporada', 'Evolución de Cluster'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # BRIX
        fig.add_trace(
            go.Scatter(x=df_evolution['Temporada'], y=df_evolution['BRIX_Promedio'],
                      mode='lines+markers', name='BRIX Promedio',
                      line=dict(color='#2E8B57', width=3),
                      error_y=dict(type='data', array=df_evolution['BRIX_Std'], visible=True)),
            row=1, col=1
        )
        
        # Acidez
        fig.add_trace(
            go.Scatter(x=df_evolution['Temporada'], y=df_evolution['Acidez_Promedio'],
                      mode='lines+markers', name='Acidez Promedio',
                      line=dict(color='#FF6347', width=3),
                      error_y=dict(type='data', array=df_evolution['Acidez_Std'], visible=True)),
            row=1, col=2
        )
        
        # Firmeza
        fig.add_trace(
            go.Scatter(x=df_evolution['Temporada'], y=df_evolution['Firmeza_Promedio'],
                      mode='lines+markers', name='Firmeza Promedio',
                      line=dict(color='#4169E1', width=3)),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df_evolution['Temporada'], y=df_evolution['Firmeza_Min'],
                      mode='lines+markers', name='Firmeza Mínima',
                      line=dict(color='#9370DB', width=2, dash='dash')),
            row=2, col=1
        )
        
        # Cluster (como barras coloreadas)
        colors = [cluster_colors.get(int(c), '#cccccc') if not pd.isna(c) else '#cccccc' for c in df_evolution['Cluster']]
        fig.add_trace(
            go.Bar(x=df_evolution['Temporada'], y=df_evolution['Cluster'],
                   name='Cluster Asignado', marker_color=colors),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, title_text=f"Evolución de {variedad_seleccionada}")
        fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de evolución
        st.markdown("#### 📋 Tabla de Evolución por Temporada")
        display_df = df_evolution[['Temporada', 'Muestras', 'Frutos', 'BRIX_Promedio', 'Acidez_Promedio', 
                                 'Firmeza_Promedio', 'Banda_BRIX', 'Banda_Firmeza', 'Suma_Bandas', 'Cluster']].round(2)
        
        # Colorear la tabla por cluster
        def color_cluster_table(row):
            cluster = row['Cluster']
            if pd.isna(cluster):
                return [''] * len(row)
            color = cluster_colors.get(int(cluster), '')
            return [f'background-color: {color}' if col == 'Cluster' else '' for col in row.index]
        
        styled_df = display_df.style.apply(color_cluster_table, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Análisis de tendencias
        st.markdown("---")
        st.markdown("### 🔍 Análisis de Tendencias")
        
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
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            brix_trend, brix_change = calculate_trend(df_evolution['BRIX_Promedio'].tolist())
            if brix_trend == "Mejora":
                st.markdown(f'<div class="improvement-box"><h4>🍇 BRIX</h4><p><strong>{brix_trend}</strong><br>Cambio: +{brix_change:.1f}%</p></div>', unsafe_allow_html=True)
            elif brix_trend == "Empeora":
                st.markdown(f'<div class="degradation-box"><h4>🍇 BRIX</h4><p><strong>{brix_trend}</strong><br>Cambio: {brix_change:.1f}%</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="stable-box"><h4>🍇 BRIX</h4><p><strong>{brix_trend}</strong><br>Cambio: {brix_change:.1f}%</p></div>', unsafe_allow_html=True)
        
        with col2:
            acidez_trend, acidez_change = calculate_trend(df_evolution['Acidez_Promedio'].tolist())
            if acidez_trend == "Mejora":
                st.markdown(f'<div class="improvement-box"><h4>🧪 Acidez</h4><p><strong>{acidez_trend}</strong><br>Cambio: +{acidez_change:.1f}%</p></div>', unsafe_allow_html=True)
            elif acidez_trend == "Empeora":
                st.markdown(f'<div class="degradation-box"><h4>🧪 Acidez</h4><p><strong>{acidez_trend}</strong><br>Cambio: {acidez_change:.1f}%</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="stable-box"><h4>🧪 Acidez</h4><p><strong>{acidez_trend}</strong><br>Cambio: {acidez_change:.1f}%</p></div>', unsafe_allow_html=True)
        
        with col3:
            firmeza_trend, firmeza_change = calculate_trend(df_evolution['Firmeza_Promedio'].tolist())
            if firmeza_trend == "Mejora":
                st.markdown(f'<div class="improvement-box"><h4>💪 Firmeza</h4><p><strong>{firmeza_trend}</strong><br>Cambio: +{firmeza_change:.1f}%</p></div>', unsafe_allow_html=True)
            elif firmeza_trend == "Empeora":
                st.markdown(f'<div class="degradation-box"><h4>💪 Firmeza</h4><p><strong>{firmeza_trend}</strong><br>Cambio: {firmeza_change:.1f}%</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="stable-box"><h4>💪 Firmeza</h4><p><strong>{firmeza_trend}</strong><br>Cambio: {firmeza_change:.1f}%</p></div>', unsafe_allow_html=True)
        
        with col4:
            cluster_trend, cluster_change = calculate_trend(df_evolution['Cluster'].tolist())
            if cluster_trend == "Mejora":
                st.markdown(f'<div class="improvement-box"><h4>🏆 Cluster</h4><p><strong>Mejor calidad</strong><br>Cambio: {cluster_change:.1f}</p></div>', unsafe_allow_html=True)
            elif cluster_trend == "Empeora":
                st.markdown(f'<div class="degradation-box"><h4>🏆 Cluster</h4><p><strong>Peor calidad</strong><br>Cambio: {cluster_change:.1f}</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="stable-box"><h4>🏆 Cluster</h4><p><strong>Calidad estable</strong><br>Cambio: {cluster_change:.1f}</p></div>', unsafe_allow_html=True)

else:
    st.warning("⚠️ Esta variedad solo tiene datos de una temporada. No se puede mostrar evolución temporal.")

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
    
    # Tabla de valores individuales
    columnas_mostrar = [col for col in [FRUTO_COLUMN, COL_BRIX, COL_ACIDEZ, 'Firmeza punto valor'] 
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
        
        # Comparación lado a lado
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {temp1}")
            if not datos_temp1.empty:
                st.metric("Muestras", len(datos_temp1))
                st.metric("BRIX Promedio", f"{datos_temp1[COL_BRIX].mean():.2f}" if COL_BRIX in datos_temp1.columns else "N/A")
                st.metric("Firmeza Promedio", f"{datos_temp1['Firmeza punto valor'].mean():.2f}" if 'Firmeza punto valor' in datos_temp1.columns else "N/A")
        
        with col2:
            st.markdown(f"#### {temp2}")
            if not datos_temp2.empty:
                st.metric("Muestras", len(datos_temp2))
                brix_t2 = datos_temp2[COL_BRIX].mean() if COL_BRIX in datos_temp2.columns else 0
                brix_t1 = datos_temp1[COL_BRIX].mean() if COL_BRIX in datos_temp1.columns else 0
                delta_brix = brix_t2 - brix_t1
                st.metric("BRIX Promedio", f"{brix_t2:.2f}", f"{delta_brix:+.2f}" if abs(delta_brix) > 0.01 else None)
                
                firm_t2 = datos_temp2['Firmeza punto valor'].mean() if 'Firmeza punto valor' in datos_temp2.columns else 0
                firm_t1 = datos_temp1['Firmeza punto valor'].mean() if 'Firmeza punto valor' in datos_temp1.columns else 0
                delta_firm = firm_t2 - firm_t1
                st.metric("Firmeza Promedio", f"{firm_t2:.2f}", f"{delta_firm:+.2f}" if abs(delta_firm) > 0.01 else None)

# Exportar datos
st.markdown("---")
if st.button("📥 Exportar análisis de evolución a Excel"):
    # Crear archivo Excel con múltiples hojas
    output = pd.ExcelWriter('evolucion_variedad.xlsx', engine='xlsxwriter')
    
    # Hoja 1: Datos de evolución por temporada
    if 'df_evolution' in locals():
        df_evolution.to_excel(output, sheet_name='Evolucion_Temporadas', index=False)
    
    # Hoja 2: Datos agregados por variedad
    agg_variedad.to_excel(output, sheet_name='Datos_Agregados', index=False)
    
    # Hoja 3: Datos crudos de la variedad
    datos_variedad.to_excel(output, sheet_name='Datos_Crudos', index=False)
    
    output.close()
    
    with open('evolucion_variedad.xlsx', 'rb') as f:
        st.download_button(
            label="Descargar archivo Excel",
            data=f.read(),
            file_name=f'evolucion_{variedad_seleccionada}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )