"""
Módulo Streamlit simplificado para la segmentación de carozos usando nuevas reglas.

Este módulo implementa ÚNICAMENTE las nuevas reglas de clustering del script de validación:
* Agrupación por temporada (no harvest_period)
* Inclusión de portainjerto en claves de agrupación
* Cálculo de bandas según las reglas específicas por especie
* Comparativas por temporada entre especies/variedades
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from new_clustering_rules import aplicar_nuevas_reglas_clustering
from pathlib import Path
from typing import Dict, List, Tuple, Union, Sequence
from collections.abc import Iterable
import streamlit as st
import unicodedata
import io
import plotly.express as px

from utils import show_logo


def load_excel_with_headers_detection(file_path: Union[str, Path], sheet_name: str, usecols: str = None) -> pd.DataFrame:
    """
    Carga un archivo Excel detectando automáticamente dónde están los encabezados.
    
    Args:
        file_path: Ruta al archivo Excel o objeto de archivo
        sheet_name: Nombre de la hoja
        usecols: Columnas a leer (ej: "A:AP")
    
    Returns:
        DataFrame con los encabezados correctamente detectados
    """
    try:
        # Primero, leer las primeras filas sin skiprows para encontrar encabezados
        try:
            preview_df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=usecols, nrows=10, dtype=str)
        except UnicodeDecodeError:
            # Si hay problemas de codificación, intentar con diferentes engines
            try:
                preview_df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=usecols, nrows=10, dtype=str, engine='openpyxl')
            except:
                preview_df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=usecols, nrows=10, dtype=str, engine='xlrd')
        
        # Buscar la fila que probablemente contenga los encabezados
        # Los encabezados usualmente tienen más texto y menos nulos
        header_row = 0
        max_non_null = 0
        
        for i in range(min(5, len(preview_df))):  # Revisar las primeras 5 filas
            non_null_count = preview_df.iloc[i].notna().sum()
            # También verificar que no sean solo números (que serían datos, no encabezados)
            text_count = sum(1 for val in preview_df.iloc[i] if isinstance(str(val), str) and len(str(val)) > 2)
            
            if non_null_count > max_non_null and text_count > non_null_count * 0.3:
                max_non_null = non_null_count
                header_row = i
        
        # Si no encontramos encabezados convincentes en las primeras filas, usar la primera
        if max_non_null < 3:
            header_row = 0
            
        # Ahora cargar el archivo completo usando la fila de encabezados detectada
        try:
            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                usecols=usecols,
                header=header_row,
                dtype=str
            )
        except UnicodeDecodeError:
            # Si hay problemas de codificación, intentar con diferentes engines
            try:
                df = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                    usecols=usecols,
                    header=header_row,
                    dtype=str,
                    engine='openpyxl'
                )
            except:
                # Último intento con xlrd
                df = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                    usecols=usecols,
                    header=header_row,
                    dtype=str,
                    engine='xlrd'
                )
        
        # Limpiar encabezados: eliminar espacios extra, convertir a string, manejar encoding
        def safe_str_clean(col, i):
            try:
                if col is None:
                    return f"Column_{i}"
                # Convertir a string de forma segura
                if isinstance(col, bytes):
                    try:
                        col_str = col.decode('utf-8')
                    except UnicodeDecodeError:
                        col_str = col.decode('latin-1', errors='ignore')
                else:
                    col_str = str(col)
                return col_str.strip()
            except:
                return f"Column_{i}"
        
        cleaned_columns = [safe_str_clean(col, i) for i, col in enumerate(df.columns)]
        
        # Manejar columnas duplicadas
        seen_columns = {}
        final_columns = []
        
        for col in cleaned_columns:
            if col in seen_columns:
                seen_columns[col] += 1
                final_columns.append(f"{col}_{seen_columns[col]}")
            else:
                seen_columns[col] = 0
                final_columns.append(col)
        
        df.columns = final_columns
        
        # Filtrar filas vacías después de los encabezados
        df = df.dropna(how='all')
        
        # Limpiar valores de celdas con problemas de encoding
        def safe_cell_clean(val):
            if val is None or pd.isna(val):
                return val
            try:
                if isinstance(val, bytes):
                    try:
                        return val.decode('utf-8')
                    except UnicodeDecodeError:
                        return val.decode('latin-1', errors='ignore')
                return str(val)
            except:
                return str(val) if val is not None else val
        
        # Aplicar limpieza a todas las celdas de texto
        for col in df.columns:
            try:
                df[col] = df[col].apply(safe_cell_clean)
            except:
                continue
        
        return df
        
    except Exception as e:
        st.error(f"Error al cargar el archivo Excel: {e}")
        # Fallback: intentar con la configuración original
        return pd.read_excel(file_path, sheet_name=sheet_name, usecols=usecols, skiprows=2, dtype=str)


def segmentacion_app(especie: str):
    """Aplicación simplificada de segmentación usando SOLO las nuevas reglas."""
    
    # Configuración general de la página
    st.set_page_config(
        page_title="Segmentaciones",
        page_icon="G.png",
        layout="wide"
    )
    
    # Estilos para los botones del sidebar
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
        unsafe_allow_html=True
    )

    # Normalizar el nombre de la especie
    especie_key = "Nectarin" if especie.lower().startswith("nect") else "Ciruela"
    titulo_especie = "Nectarina" if especie_key == "Nectarin" else "Ciruela"

    # Parámetros y constantes
    ESPECIE_COLUMN = "Especie"
    VAR_COLUMN = "Variedad"
    FRUTO_COLUMN = "Fruto (n°)"
    COL_BRIX = "Solidos solubles (%)"
    COL_ACIDEZ = "Acidez (%)"
    
    # Sidebar con menú
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

    generar_menu()

    st.title(f"🎯 Segmentación {titulo_especie} - Nuevas Reglas")
    st.markdown("""
    Esta página implementa **únicamente las nuevas reglas de clustering** basadas en el script de validación.
    
    **Características:**
    - ✅ Agrupación por temporada (en lugar de harvest_period)
    - ✅ Inclusión de portainjerto en las claves de agrupación  
    - ✅ Bandas específicas por especie y subtipo
    - ✅ Comparativas por temporada entre variedades
    """)

    # Verificar archivo cargado
    if especie_key in ("Nectarin", "Ciruela"):
        df_upload = st.session_state.get("carozos_df")
        file_label = "carozos"
    else:
        df_upload = st.session_state.get("cerezas_df")
        file_label = "cerezas"

    if df_upload is None:
        st.info(f"No se encontró el archivo de {file_label}. Primero súbelo en la página 'Carga de archivos'.")
        if st.button("📁 Ir a Carga de archivos"):
            st.switch_page('pages/carga_datos.py')
        return

    # Aplicar procesamiento básico
    df_processed = df_upload.copy()
    
    # Filtrar por especie
    if ESPECIE_COLUMN in df_processed.columns:
        df_processed = df_processed[df_processed[ESPECIE_COLUMN] == especie_key].copy()
    
    # Conversión básica a numérico para columnas necesarias
    numeric_cols = [COL_BRIX, COL_ACIDEZ, "Quilla", "Hombro", "Punta", "Peso (g)"]
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(
                df_processed[col].astype(str).str.replace(',', '.'), 
                errors='coerce'
            )
    
    # Agregar columnas necesarias para las nuevas reglas
    if 'avg_mejillas' not in df_processed.columns:
        mejilla_cols = ["Firmeza quilla", "Firmeza hombro", "Firmeza punta"]
        available_mejilla_cols = [col for col in mejilla_cols if col in df_processed.columns]
        if available_mejilla_cols:
            df_processed['avg_mejillas'] = df_processed[available_mejilla_cols].mean(axis=1)
        else:
            df_processed['avg_mejillas'] = np.nan
    
    # Agregar temporada si no existe (usar año de fecha o valor por defecto)
    if 'temporada' not in df_processed.columns:
        if 'Fecha de evaluación' in df_processed.columns:
            df_processed['Fecha de evaluación'] = pd.to_datetime(df_processed['Fecha de evaluación'], errors='coerce')
            df_processed['temporada'] = df_processed['Fecha de evaluación'].dt.year.astype(str)
            df_processed['temporada'] = df_processed['temporada'].fillna('2023')
        else:
            df_processed['temporada'] = '2023'
    
    # Mostrar datos procesados
    st.markdown("### 📋 Datos Procesados")
    st.write(f"Total de registros para {titulo_especie}: {len(df_processed)}")
    
    # Mostrar información básica
    if len(df_processed) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Variedades", df_processed[VAR_COLUMN].nunique() if VAR_COLUMN in df_processed.columns else 0)
        with col2:
            st.metric("Temporadas", df_processed['temporada'].nunique())
        with col3:
            st.metric("Campos", df_processed['Campo'].nunique() if 'Campo' in df_processed.columns else 0)
        
        st.dataframe(df_processed.head(), use_container_width=True)
    
        # APLICAR NUEVAS REGLAS DE CLUSTERING
        st.markdown("---")
        st.markdown("### 🎯 Aplicar Nuevas Reglas de Clustering")
        
        if st.button("🚀 Procesar con Nuevas Reglas"):
            try:
                with st.spinner("Aplicando nuevas reglas de clustering..."):
                    agg_groups = aplicar_nuevas_reglas_clustering(
                        df_processed, COL_BRIX, COL_ACIDEZ, ESPECIE_COLUMN, VAR_COLUMN, FRUTO_COLUMN
                    )
                    
                st.success(f"✅ Nuevas reglas aplicadas exitosamente: {len(agg_groups)} grupos creados")
                
                # Guardar en session_state para otras páginas
                st.session_state["df_seg_especies"] = agg_groups
                
                # Mostrar distribución de clusters
                if 'cluster_grp' in agg_groups.columns:
                    st.markdown("#### 📊 Distribución de Clusters")
                    cluster_counts = agg_groups['cluster_grp'].value_counts().sort_index()
                    cluster_names = {1: "Excelente", 2: "Bueno", 3: "Regular", 4: "Deficiente"}
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        for cluster, count in cluster_counts.items():
                            if not pd.isna(cluster):
                                st.metric(
                                    f"Cluster {int(cluster)} - {cluster_names.get(int(cluster), 'Desconocido')}", 
                                    count
                                )
                    
                    with col2:
                        # Gráfico de distribución
                        cluster_data = pd.DataFrame({
                            'Cluster': [f"C{int(c)} - {cluster_names.get(int(c), 'Desc')}" for c in cluster_counts.index if not pd.isna(c)],
                            'Cantidad': [cluster_counts[c] for c in cluster_counts.index if not pd.isna(c)]
                        })
                        fig = px.pie(cluster_data, values='Cantidad', names='Cluster', title="Distribución de Clusters")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Tabla de resultados
                st.markdown("#### 📋 Resultados por Grupo")
                display_cols = []
                for col in [ESPECIE_COLUMN, VAR_COLUMN, 'temporada', 'Campo', 'portainjerto', 
                           'brix_promedio', 'banda_brix', 'acidez_primer_fruto', 'banda_acidez',
                           'mejillas_promedio', 'banda_mejillas', 'firmeza_punto_debil', 'banda_firmeza_punto',
                           'suma_bandas', 'cluster_grp', 'n_registros']:
                    if col in agg_groups.columns:
                        display_cols.append(col)
                
                if display_cols:
                    # Función para colorear clusters
                    def color_cluster(val):
                        if pd.isna(val):
                            return ''
                        colors = {1: 'background-color: #d4edda', 2: 'background-color: #fff3cd', 
                                 3: 'background-color: #f8d7da', 4: 'background-color: #f5c6cb'}
                        return colors.get(int(val), '')
                    
                    styled_df = agg_groups[display_cols].style.applymap(
                        color_cluster, subset=['cluster_grp'] if 'cluster_grp' in display_cols else []
                    )
                    
                    st.dataframe(styled_df, use_container_width=True)
                
                # Comparativa por temporada
                st.markdown("#### 📈 Comparativa por Temporada")
                if 'temporada' in agg_groups.columns and len(agg_groups['temporada'].unique()) > 1:
                    # Seleccionar variedad para comparar
                    variedades_disponibles = sorted(agg_groups[VAR_COLUMN].dropna().unique())
                    variedad_sel = st.selectbox("Selecciona variedad para comparar por temporada:", variedades_disponibles)
                    
                    if variedad_sel:
                        datos_var = agg_groups[agg_groups[VAR_COLUMN] == variedad_sel].copy()
                        
                        if len(datos_var) > 0:
                            # Gráfico de evolución
                            metricas_graf = ['brix_promedio', 'acidez_primer_fruto', 'mejillas_promedio', 'firmeza_punto_debil']
                            metricas_disponibles = [m for m in metricas_graf if m in datos_var.columns]
                            
                            if metricas_disponibles:
                                datos_graf = datos_var.groupby('temporada')[metricas_disponibles].mean().reset_index()
                                
                                fig = px.line(
                                    datos_graf.melt(id_vars=['temporada'], value_vars=metricas_disponibles),
                                    x='temporada', y='value', color='variable',
                                    title=f"Evolución de {variedad_sel} por temporada",
                                    labels={'value': 'Valor', 'variable': 'Métrica', 'temporada': 'Temporada'}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Tabla detallada
                            st.dataframe(datos_var[display_cols], use_container_width=True)
                        else:
                            st.warning("No hay datos para la variedad seleccionada.")
                else:
                    st.info("No hay múltiples temporadas para comparar.")
                
                # Exportar resultados
                st.markdown("#### 💾 Exportar Resultados")
                if st.button("📥 Descargar Excel"):
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                        agg_groups.to_excel(writer, sheet_name='Resultados_Nuevas_Reglas', index=False)
                        
                        # Hoja de resumen por clusters
                        if 'cluster_grp' in agg_groups.columns:
                            resumen = agg_groups.groupby('cluster_grp').agg({
                                'n_registros': 'sum',
                                'brix_promedio': 'mean',
                                'acidez_primer_fruto': 'mean',
                                'mejillas_promedio': 'mean',
                                'firmeza_punto_debil': 'mean'
                            }).reset_index()
                            resumen.to_excel(writer, sheet_name='Resumen_Clusters', index=False)
                    
                    st.download_button(
                        label="📥 Descargar",
                        data=buf.getvalue(),
                        file_name=f"segmentacion_{titulo_especie.lower()}_nuevas_reglas.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
            except Exception as e:
                st.error(f"❌ Error aplicando las nuevas reglas: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
    else:
        st.warning("No hay datos disponibles para procesar.")