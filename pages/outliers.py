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

def crear_boxplot_outliers(df, column, title):
    """Crea un boxplot simple para visualizar outliers"""
    if column not in df.columns:
        return None
    
    values = pd.to_numeric(df[column], errors='coerce').dropna()
    
    fig = go.Figure()
    fig.add_trace(go.Box(y=values, name=column, boxpoints="outliers"))
    fig.update_layout(
        title=title,
        yaxis_title=column,
        height=400
    )
    return fig

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

def detectar_outliers_por_especie(df_especie, especie_nombre, method, threshold_z=2.0, factor_iqr=1.5):
    """Función auxiliar para detectar outliers por especie"""
    
    # Variables numéricas relevantes para cada especie
    numeric_columns = [
        "BRIX", "Acidez (%)", "Peso (g)",
        "Punta", "Quilla", "Hombro", "Mejilla 1", "Mejilla 2"
    ]
    
    # Filtrar solo columnas que existan
    available_columns = [col for col in numeric_columns if col in df_especie.columns]
    
    if not available_columns:
        st.warning(f"No se encontraron columnas numéricas para {especie_nombre}")
        return df_especie, {}
    
    st.markdown(f"#### Variables analizadas para {especie_nombre}:")
    st.write(", ".join(available_columns))
    
    # Configuración de parámetros
    col1, col2 = st.columns(2)
    
    with col1:
        if method == "Z-Score":
            threshold = st.slider(f"Umbral Z-Score para {especie_nombre}", 1.0, 4.0, threshold_z, 0.1, key=f"zscore_{especie_nombre}")
        else:
            factor = st.slider(f"Factor IQR para {especie_nombre}", 1.0, 3.0, factor_iqr, 0.1, key=f"iqr_{especie_nombre}")
    
    with col2:
        st.markdown("**Método seleccionado:**")
        if method == "Z-Score":
            st.info(f"Z-Score con umbral {threshold}")
        else:
            st.info(f"IQR con factor {factor}")
    
    # Detectar outliers
    if method == "Z-Score":
        df_outliers, outliers_info = detectar_outliers_zscore(df_especie, available_columns, threshold)
    else:
        df_outliers, outliers_info = detectar_outliers_iqr(df_especie, available_columns, factor)
    
    return df_outliers, outliers_info, available_columns

def main():
    generar_menu()
    
    st.title("🎯 Detección de Outliers por Especie")
    
    st.markdown("""
    Esta página te permite:
    1. **Detectar outliers por especie** usando diferentes métodos (Z-Score o IQR)
    2. **Visualizar** distribuciones y outliers por cada especie
    3. **Filtrar y guardar** datos sin outliers para cada especie
    4. **Configurar parámetros** específicos por especie
    """)
    
    # Verificar si hay datos cargados
    if "carozos_df" not in st.session_state:
        st.warning("⚠️ No hay datos cargados. Por favor, carga primero un archivo en la página de **Carga de archivos**.")
        if st.button("📁 Ir a Carga de archivos"):
            st.switch_page('pages/carga_datos.py')
        return
    
    df = st.session_state["carozos_df"].copy()
    st.success(f"✅ Datos cargados: {len(df)} registros, {len(df.columns)} columnas")
    
    # Verificar que existe la columna de especie
    if "Especie" not in df.columns:
        st.error("❌ No se encontró la columna 'Especie' en los datos.")
        return
    
    # Mostrar resumen de especies
    especies_disponibles = df["Especie"].value_counts()
    st.markdown("### 📊 Resumen por especie:")
    
    col_summary1, col_summary2 = st.columns(2)
    with col_summary1:
        st.dataframe(especies_disponibles.to_frame("Registros"), use_container_width=True)
    
    # Configuración global
    st.header("⚙️ Configuración de Detección")
    
    col_config1, col_config2 = st.columns(2)
    with col_config1:
        method = st.selectbox("Método de detección", ["Z-Score", "IQR"])
    
    with col_config2:
        if method == "Z-Score":
            default_threshold = st.number_input("Umbral base Z-Score", 1.0, 4.0, 2.0, 0.1)
        else:
            default_factor = st.number_input("Factor base IQR", 1.0, 3.0, 1.5, 0.1)
    
    # Pestañas por especie
    st.header("📋 Detección por Especie")
    
    # Crear pestañas dinámicamente según las especies disponibles
    especies = df["Especie"].unique()
    tabs = st.tabs([f"🍑 {especie}" for especie in especies])
    
    # Diccionario para almacenar datos filtrados
    filtered_data = {}
    
    for i, especie in enumerate(especies):
        with tabs[i]:
            st.markdown(f"### Análisis de Outliers - {especie}")
            
            # Filtrar datos por especie
            df_especie = df[df["Especie"] == especie].copy()
            st.info(f"📊 **{especie}**: {len(df_especie)} registros")
            
            if len(df_especie) == 0:
                st.warning(f"No hay datos para {especie}")
                continue
            
            # Detectar outliers para esta especie
            try:
                if method == "Z-Score":
                    result = detectar_outliers_por_especie(df_especie, especie, method, default_threshold)
                else:
                    result = detectar_outliers_por_especie(df_especie, especie, method, factor_iqr=default_factor)
                
                df_outliers, outliers_info, available_columns = result
                
                # Mostrar resumen de outliers
                st.markdown("#### 📈 Resumen de Outliers:")
                
                if outliers_info:
                    summary_data = []
                    for var, info in outliers_info.items():
                        summary_data.append({
                            'Variable': var,
                            'Outliers': info['count'],
                            '% Outliers': f"{info['percentage']:.1f}%"
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Visualización
                    st.markdown("#### 📊 Visualización de Outliers:")
                    
                    # Selector de variable para visualizar
                    var_to_plot = st.selectbox(
                        f"Variable a visualizar para {especie}",
                        available_columns,
                        key=f"var_plot_{especie}"
                    )
                    
                    if var_to_plot:
                        # Crear gráfico de boxplot
                        fig = crear_boxplot_outliers(df_especie, var_to_plot, f"{var_to_plot} - {especie}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Opción para filtrar outliers
                    st.markdown("#### 🗂️ Filtrado de Datos:")
                    
                    if st.checkbox(f"Aplicar filtro de outliers para {especie}", key=f"filter_{especie}"):
                        # Crear máscara de outliers
                        outlier_mask = pd.Series(False, index=df_especie.index)
                        
                        for col in available_columns:
                            outlier_col = f"{col}_outlier"
                            if outlier_col in df_outliers.columns:
                                outlier_mask |= df_outliers[outlier_col]
                        
                        # Datos sin outliers
                        df_clean = df_especie[~outlier_mask]
                        total_outliers_removed = outlier_mask.sum()
                        
                        st.success(f"✅ **{especie}**: Removidos {total_outliers_removed} outliers. Datos limpios: {len(df_clean)} registros")
                        
                        # Guardar datos filtrados
                        filtered_data[especie] = df_clean
                        
                        # Mostrar estadísticas comparativas
                        col_stats1, col_stats2 = st.columns(2)
                        
                        with col_stats1:
                            st.markdown("**Datos originales:**")
                            st.dataframe(df_especie[available_columns].describe(), use_container_width=True)
                        
                        with col_stats2:
                            st.markdown("**Datos filtrados:**")
                            st.dataframe(df_clean[available_columns].describe(), use_container_width=True)
                    
                else:
                    st.info("No se detectaron outliers o no hay suficientes datos.")
                    
            except Exception as e:
                st.error(f"Error procesando {especie}: {e}")
    
    # Opción para guardar datos filtrados
    st.header("💾 Guardar Datos Filtrados")
    
    if filtered_data:
        st.markdown("### Resumen de datos filtrados:")
        
        summary_filtered = []
        total_original = len(df)
        total_filtered = 0
        
        for especie, df_clean in filtered_data.items():
            original_count = len(df[df["Especie"] == especie])
            filtered_count = len(df_clean)
            removed_count = original_count - filtered_count
            total_filtered += filtered_count
            
            summary_filtered.append({
                'Especie': especie,
                'Registros Originales': original_count,
                'Registros Filtrados': filtered_count,
                'Outliers Removidos': removed_count,
                '% Removido': f"{(removed_count/original_count*100):.1f}%"
            })
        
        summary_df_filtered = pd.DataFrame(summary_filtered)
        st.dataframe(summary_df_filtered, use_container_width=True)
        
        # Combinar datos filtrados
        df_combined_filtered = pd.concat(list(filtered_data.values()), ignore_index=True)
        
        # Agregar especies no filtradas
        especies_no_filtradas = [esp for esp in especies if esp not in filtered_data.keys()]
        if especies_no_filtradas:
            df_no_filtradas = df[df["Especie"].isin(especies_no_filtradas)]
            df_combined_filtered = pd.concat([df_combined_filtered, df_no_filtradas], ignore_index=True)
        
        st.info(f"📊 **Total**: {total_original} → {len(df_combined_filtered)} registros ({total_original - len(df_combined_filtered)} outliers removidos)")
        
        # Botón para guardar
        if st.button("💾 Guardar datos filtrados", key="save_filtered"):
            st.session_state["carozos_df_filtered"] = df_combined_filtered
            st.success("✅ Datos filtrados guardados exitosamente. Se usarán en las páginas de segmentación.")
            st.balloons()
        
    else:
        st.info("No hay datos filtrados para guardar. Aplica filtros en las pestañas de especies.")
    
    # Mostrar estado actual
    if "carozos_df_filtered" in st.session_state:
        filtered_df = st.session_state["carozos_df_filtered"]
        st.success(f"✅ **Estado actual**: Datos filtrados activos con {len(filtered_df)} registros")
        
        if st.button("🗑️ Limpiar todos los filtros"):
            if "carozos_df_filtered" in st.session_state:
                del st.session_state["carozos_df_filtered"]
            st.success("✅ Filtros eliminados. Se usarán todos los datos originales.")
            st.rerun()

if __name__ == "__main__":
    main()
