import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from utils import show_logo

# Configuración de la página
st.set_page_config(
    page_title="Detección de Outliers", 
    page_icon="🎯", 
    layout="wide"
)

st.markdown(
    """
    <style>
      [data-testid="stSidebar"] div.stButton > button {
        background-color: #D32F2F !important;
        color: white !important;
        border: none !important;
      }
      [data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #B71C1C !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

def generar_menu():
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
        if st.button('Detección Outliers 🎯', type="primary"):
            st.switch_page('pages/outliers.py')

def detectar_outliers_zscore(df, columns, threshold=2.0):
    """Detecta outliers usando Z-Score"""
    outliers_info = {}
    df_outliers = df.copy()
    
    for col in columns:
        if col in df.columns and df[col].notna().sum() > 0:
            # Convertir a numérico
            values = pd.to_numeric(df[col], errors='coerce')
            z_scores = np.abs(stats.zscore(values, nan_policy='omit'))
            outliers = z_scores > threshold
            
            df_outliers[f'{col}_outlier'] = outliers
            df_outliers[f'{col}_zscore'] = z_scores
            
            outliers_info[col] = {
                'count': outliers.sum(),
                'percentage': (outliers.sum() / len(df)) * 100,
                'threshold': threshold
            }
    
    return df_outliers, outliers_info

def detectar_outliers_iqr(df, columns, factor=1.5):
    """Detecta outliers usando método IQR"""
    outliers_info = {}
    df_outliers = df.copy()
    
    for col in columns:
        if col in df.columns and df[col].notna().sum() > 0:
            # Convertir a numérico
            values = pd.to_numeric(df[col], errors='coerce')
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            outliers = (values < lower_bound) | (values > upper_bound)
            
            df_outliers[f'{col}_outlier'] = outliers
            df_outliers[f'{col}_lower_bound'] = lower_bound
            df_outliers[f'{col}_upper_bound'] = upper_bound
            
            outliers_info[col] = {
                'count': outliers.sum(),
                'percentage': (outliers.sum() / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR
            }
    
    return df_outliers, outliers_info

def crear_grafico_outliers(df, column, outlier_method='zscore'):
    """Crea gráficos para visualizar outliers"""
    if column not in df.columns:
        return None
    
    values = pd.to_numeric(df[column], errors='coerce').dropna()
    
    if outlier_method == 'zscore':
        outlier_col = f'{column}_outlier'
        if outlier_col in df.columns:
            outliers = df[df[outlier_col] == True][column]
            normal = df[df[outlier_col] == False][column]
        else:
            return None
    else:  # IQR
        outlier_col = f'{column}_outlier'
        if outlier_col in df.columns:
            outliers = df[df[outlier_col] == True][column]
            normal = df[df[outlier_col] == False][column]
        else:
            return None
    
    # Crear subplot con boxplot e histograma
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[f'Boxplot - {column}', f'Distribución - {column}'],
        vertical_spacing=0.12
    )
    
    # Boxplot
    fig.add_trace(
        go.Box(y=values, name=column, boxpoints="outliers"),
        row=1, col=1
    )
    
    # Histograma
    fig.add_trace(
        go.Histogram(x=normal, name="Normal", opacity=0.7, nbinsx=30),
        row=2, col=1
    )
    
    if len(outliers) > 0:
        fig.add_trace(
            go.Histogram(x=outliers, name="Outliers", opacity=0.7, nbinsx=30),
            row=2, col=1
        )
    
    fig.update_layout(height=600, showlegend=True)
    return fig

def main():
    generar_menu()
    
    st.title("🎯 Detección de Outliers y Filtrado de Datos")
    
    st.markdown("""
    Esta página te permite:
    1. **Detectar outliers** en tus datos usando diferentes métodos
    2. **Visualizar** la distribución de outliers por variable
    3. **Filtrar datos** para excluir outliers del análisis
    4. **Configurar parámetros** de detección personalizados
    """)
    
    # Verificar si hay datos cargados
    if "carozos_df" not in st.session_state:
        st.warning("⚠️ No hay datos cargados. Por favor, carga primero un archivo en la página de **Carga de archivos**.")
        if st.button("📁 Ir a Carga de archivos"):
            st.switch_page('pages/carga_datos.py')
        return
    
    df = st.session_state["carozos_df"].copy()
    st.success(f"✅ Datos cargados: {len(df)} registros, {len(df.columns)} columnas")
    
    # Configuración de detección de outliers
    st.header("⚙️ Configuración de Detección")
    
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.selectbox(
            "Método de detección:",
            ["Z-Score", "IQR (Rango Intercuartílico)"],
            help="Z-Score: identifica valores > N desviaciones estándar. IQR: identifica valores fuera del rango Q1-1.5*IQR, Q3+1.5*IQR"
        )
        
        if method == "Z-Score":
            threshold = st.slider(
                "Umbral Z-Score:",
                min_value=1.0, max_value=4.0, value=2.0, step=0.1,
                help="Valores con |Z-Score| > umbral se consideran outliers"
            )
        else:
            factor = st.slider(
                "Factor IQR:",
                min_value=1.0, max_value=3.0, value=1.5, step=0.1,
                help="Outliers = valores < Q1-factor*IQR o > Q3+factor*IQR"
            )
    
    with col2:
        # Selección de columnas numéricas para análisis
        numeric_columns = []
        for col in df.columns:
            try:
                pd.to_numeric(df[col], errors='coerce')
                if df[col].notna().sum() > 10:  # Al menos 10 valores no nulos
                    numeric_columns.append(col)
            except:
                pass
        
        selected_columns = st.multiselect(
            "Columnas a analizar:",
            numeric_columns,
            default=numeric_columns[:5] if len(numeric_columns) > 5 else numeric_columns,
            help="Selecciona las variables numéricas para detectar outliers"
        )
    
    if not selected_columns:
        st.warning("⚠️ Selecciona al menos una columna para analizar.")
        return
    
    # Ejecutar detección de outliers
    st.header("🔍 Resultados de Detección")
    
    with st.spinner("Detectando outliers..."):
        if method == "Z-Score":
            df_outliers, outliers_info = detectar_outliers_zscore(df, selected_columns, threshold)
        else:
            df_outliers, outliers_info = detectar_outliers_iqr(df, selected_columns, factor)
    
    # Mostrar resumen de outliers
    st.subheader("📊 Resumen de Outliers Detectados")
    
    summary_data = []
    for col, info in outliers_info.items():
        summary_data.append({
            "Variable": col,
            "Total Outliers": info['count'],
            "% Outliers": f"{info['percentage']:.2f}%",
            "Registros Válidos": df[col].notna().sum(),
            "Método": method
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Crear pestañas para visualización y filtrado
    tab1, tab2, tab3 = st.tabs(["📈 Visualización", "🔧 Filtrado de Datos", "📋 Datos Detallados"])
    
    with tab1:
        st.subheader("Visualización de Outliers")
        
        col_to_plot = st.selectbox("Selecciona variable para visualizar:", selected_columns)
        
        if col_to_plot:
            fig = crear_grafico_outliers(df_outliers, col_to_plot, 
                                       'zscore' if method == "Z-Score" else 'iqr')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Estadísticas de la variable
            st.write("**Estadísticas:**")
            col_stats = df[col_to_plot].describe()
            st.write(col_stats)
    
    with tab2:
        st.subheader("Filtrado de Datos para Modelado")
        
        st.markdown("""
        **Configura qué datos excluir del análisis:**
        - Puedes excluir outliers por variable individual
        - O aplicar filtros combinados
        """)
        
        # Crear filtros por variable
        filters = {}
        for col in selected_columns:
            outlier_col = f'{col}_outlier'
            if outlier_col in df_outliers.columns:
                exclude_outliers = st.checkbox(
                    f"Excluir outliers de **{col}** ({outliers_info[col]['count']} registros)",
                    value=False,
                    key=f"exclude_{col}"
                )
                filters[col] = exclude_outliers
        
        # Filtros adicionales
        st.write("**Filtros Adicionales:**")
        col1, col2 = st.columns(2)
        
        with col1:
            # Filtro por especie
            if 'Especie' in df.columns:
                species_options = ['Todas'] + list(df['Especie'].unique())
                selected_species = st.multiselect(
                    "Incluir especies:",
                    species_options,
                    default=species_options,
                    key="species_filter"
                )
        
        with col2:
            # Filtro por rango de fechas
            if 'Fecha evaluación' in df.columns:
                date_filter = st.checkbox("Aplicar filtro de fechas", key="date_filter")
                if date_filter:
                    # Convertir fechas para filtro
                    try:
                        df['fecha_parsed'] = pd.to_datetime(df['Fecha evaluación'], errors='coerce')
                        date_range = st.date_input(
                            "Rango de fechas:",
                            value=[df['fecha_parsed'].min(), df['fecha_parsed'].max()],
                            key="date_range"
                        )
                    except:
                        st.warning("No se pudo parsear las fechas")
        
        # Aplicar filtros y mostrar resultado
        if st.button("🔄 Aplicar Filtros", key="apply_filters"):
            df_filtered = df_outliers.copy()
            excluded_count = 0
            
            # Aplicar filtros de outliers
            for col, exclude in filters.items():
                if exclude:
                    outlier_col = f'{col}_outlier'
                    mask = df_filtered[outlier_col] == False
                    excluded_count += (~mask).sum()
                    df_filtered = df_filtered[mask]
            
            # Aplicar filtro de especies
            if 'Especie' in df.columns and 'Todas' not in selected_species:
                df_filtered = df_filtered[df_filtered['Especie'].isin(selected_species)]
            
            # Aplicar filtro de fechas
            if date_filter and 'fecha_parsed' in df_filtered.columns:
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    df_filtered = df_filtered[
                        (df_filtered['fecha_parsed'] >= pd.Timestamp(start_date)) &
                        (df_filtered['fecha_parsed'] <= pd.Timestamp(end_date))
                    ]
            
            # Guardar datos filtrados en session_state
            st.session_state["carozos_df_filtered"] = df_filtered
            
            st.success(f"""
            ✅ **Filtros aplicados:**
            - Registros originales: {len(df)}
            - Registros después de filtros: {len(df_filtered)}
            - Registros excluidos: {len(df) - len(df_filtered)}
            """)
            
            # Mostrar estadísticas del dataset filtrado
            if len(df_filtered) > 0:
                st.write("**Distribución por especie (datos filtrados):**")
                if 'Especie' in df_filtered.columns:
                    species_counts = df_filtered['Especie'].value_counts()
                    st.write(species_counts)
                
                # Opción para descargar datos filtrados
                @st.cache_data
                def convert_df_to_csv(dataframe):
                    return dataframe.to_csv(index=False).encode('utf-8')
                
                csv_filtered = convert_df_to_csv(df_filtered)
                st.download_button(
                    label="📥 Descargar datos filtrados (CSV)",
                    data=csv_filtered,
                    file_name="datos_filtrados_sin_outliers.csv",
                    mime="text/csv"
                )
        
        # Mostrar estado actual de filtros
        if "carozos_df_filtered" in st.session_state:
            filtered_df = st.session_state["carozos_df_filtered"]
            st.info(f"📊 **Datos filtrados activos:** {len(filtered_df)} registros (de {len(df)} originales)")
            
            if st.button("🗑️ Limpiar filtros", key="clear_filters"):
                if "carozos_df_filtered" in st.session_state:
                    del st.session_state["carozos_df_filtered"]
                st.success("✅ Filtros eliminados. Se usarán todos los datos originales.")
                st.rerun()
    
    with tab3:
        st.subheader("Vista Detallada de Outliers")
        
        # Tabla con outliers marcados
        variable_detail = st.selectbox("Variable para ver detalles:", selected_columns, key="detail_var")
        
        if variable_detail:
            outlier_col = f'{variable_detail}_outlier'
            if outlier_col in df_outliers.columns:
                # Mostrar solo registros con outliers
                outliers_only = df_outliers[df_outliers[outlier_col] == True]
                
                if len(outliers_only) > 0:
                    st.write(f"**Registros con outliers en {variable_detail}:** {len(outliers_only)}")
                    
                    # Seleccionar columnas relevantes para mostrar
                    cols_to_show = ['Especie', 'Variedad', variable_detail]
                    if method == "Z-Score":
                        zscore_col = f'{variable_detail}_zscore'
                        if zscore_col in outliers_only.columns:
                            cols_to_show.append(zscore_col)
                    else:
                        bound_cols = [f'{variable_detail}_lower_bound', f'{variable_detail}_upper_bound']
                        cols_to_show.extend([col for col in bound_cols if col in outliers_only.columns])
                    
                    # Filtrar columnas que existen
                    available_cols = [col for col in cols_to_show if col in outliers_only.columns]
                    
                    st.dataframe(
                        outliers_only[available_cols].head(100),
                        use_container_width=True
                    )
                    
                    if len(outliers_only) > 100:
                        st.info(f"Mostrando los primeros 100 de {len(outliers_only)} outliers")
                else:
                    st.info("No se encontraron outliers para esta variable.")

if __name__ == "__main__":
    main()