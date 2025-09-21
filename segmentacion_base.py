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
from common_styles import configure_page, get_cluster_colors, get_cluster_style_function, get_plotly_color_map, get_plotly_color_sequence, generarMenu
from data_columns import (
    COL_ESPECIE,
    COL_VARIEDAD,
    COL_FRUTO,
    COL_BRIX as BRIX_COLUMN,
    COL_ACIDEZ as ACIDEZ_COLUMN,
    COL_PORTAINJERTO,
    COL_CAMPO,
    standardize_columns,
)


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
        df = standardize_columns(df)
        
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
        fallback_df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=usecols, skiprows=2, dtype=str)
        return standardize_columns(fallback_df)


def segmentacion_app(especie: str):
    """Aplicación simplificada de segmentación usando SOLO las nuevas reglas."""
    
    # Configuración con estilos unificados
    configure_page("Segmentaciones", "🍑")

    # Normalizar el nombre de la especie
    especie_key = "Nectarin" if especie.lower().startswith("nect") else "Ciruela"
    titulo_especie = "Nectarina" if especie_key == "Nectarin" else "Ciruela"

    # Parámetros y constantes
    ESPECIE_COLUMN = COL_ESPECIE
    VAR_COLUMN = COL_VARIEDAD
    FRUTO_COLUMN = COL_FRUTO
    COL_BRIX = BRIX_COLUMN
    COL_ACIDEZ = ACIDEZ_COLUMN
    
    # Sidebar con menú unificado

    generarMenu()

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
        # Normalizar nombres de columnas para buscar mejillas
        column_mapping = {}
        for col in df_processed.columns:
            normalized_col = str(col).strip().upper()
            column_mapping[normalized_col] = col
        
        # Buscar columnas de mejillas con diferentes nombres normalizados
        mejilla_cols = []
        mejilla_patterns = ["MEJILLA", "CHEEK", "FIRMEZA MEJILLA", "FIRMEZA CHEEK"]
        
        for normalized_col, original_col in column_mapping.items():
            for pattern in mejilla_patterns:
                if pattern in normalized_col:
                    mejilla_cols.append(original_col)
                    break
        
        # Si no encuentra mejillas, usar columnas de firmeza alternativas
        if not mejilla_cols:
            fallback_patterns = ["QUILLA", "HOMBRO", "PUNTA", "FIRMEZA QUILLA", "FIRMEZA HOMBRO", "FIRMEZA PUNTA"]
            for normalized_col, original_col in column_mapping.items():
                for pattern in fallback_patterns:
                    if pattern in normalized_col:
                        mejilla_cols.append(original_col)
                        break
        
        if mejilla_cols:
            st.info(f"📊 Calculando mejillas promedio usando columnas: {mejilla_cols}")
            # Convertir columnas a numérico, errores a NaN
            try:
                for col in mejilla_cols:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                
                # Verificar si alguna columna tiene datos numéricos válidos
                valid_cols = []
                for col in mejilla_cols:
                    if not df_processed[col].isna().all():
                        valid_cols.append(col)
                
                if valid_cols:
                    df_processed['avg_mejillas'] = df_processed[valid_cols].mean(axis=1)
                else:
                    st.warning(f"⚠️ Las columnas {mejilla_cols} no contienen datos numéricos válidos")
                    df_processed['avg_mejillas'] = np.nan
                    
            except Exception as e:
                st.error(f"❌ Error al calcular mejillas: {str(e)}")
                df_processed['avg_mejillas'] = np.nan
        else:
            st.warning("⚠️ No se encontraron columnas de mejillas o firmeza para calcular promedio")
            df_processed['avg_mejillas'] = np.nan
    
    # La columna temporada debe venir por defecto en los datos
    # Normalizar nombres de columnas para buscar TEMPORADA
    column_mapping = {}
    for col in df_processed.columns:
        normalized_col = str(col).strip().upper()
        column_mapping[normalized_col] = col
    
    # Buscar TEMPORADA en diferentes formas
    temporada_col = None
    for search_term in ['TEMPORADA', 'SEASON', 'PERIODO', 'PERIOD']:
        if search_term in column_mapping:
            temporada_col = column_mapping[search_term]
            break
    
    if temporada_col and temporada_col != 'temporada':
        st.info(f"📅 Encontrada columna de temporada: '{temporada_col}' → renombrada a 'temporada'")
        df_processed['temporada'] = df_processed[temporada_col]
    elif 'temporada' not in df_processed.columns:
        st.warning("⚠️ La columna 'temporada' no existe en los datos. Se debe cargar desde el archivo.")
        df_processed['temporada'] = 'Unknown'
    
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
            st.metric("Campos", df_processed[COL_CAMPO].nunique() if COL_CAMPO in df_processed.columns else 0)
        
        st.dataframe(df_processed.head(), use_container_width=True)
    
        # APLICAR NUEVAS REGLAS DE CLUSTERING
        st.markdown("---")
        st.markdown("### 🎯 Aplicar Nuevas Reglas de Clustering")
        
        # Crear clave única para esta especie y datos
        data_key = f"processed_{titulo_especie.lower()}_{len(df_processed)}"
        
        # Verificar si ya se procesaron datos para esta especie
        if data_key not in st.session_state:
            if st.button("🚀 Procesar con Nuevas Reglas"):
                try:
                    with st.spinner("Aplicando nuevas reglas de clustering..."):
                        agg_groups = aplicar_nuevas_reglas_clustering(
                            df_processed, COL_BRIX, COL_ACIDEZ, ESPECIE_COLUMN, VAR_COLUMN, FRUTO_COLUMN
                        )
                        
                    st.success(f"✅ Nuevas reglas aplicadas exitosamente: {len(agg_groups)} grupos creados")
                    
                    # Guardar en session_state con clave específica
                    st.session_state[data_key] = agg_groups
                    st.session_state[f"{titulo_especie.lower()}_processed"] = df_processed
                    
                    # También guardar en las claves generales para compatibilidad
                    if titulo_especie == "Ciruela":
                        st.session_state["agg_groups_plum"] = agg_groups
                        st.session_state["df_processed_plum"] = df_processed
                    elif titulo_especie == "Nectarina":
                        st.session_state["agg_groups_nect"] = agg_groups
                        st.session_state["df_processed_nect"] = df_processed
                        
                    st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error al aplicar las reglas: {str(e)}")
                    return
        else:
            # Datos ya procesados, mostrar información
            agg_groups = st.session_state[data_key]
            st.info(f"✅ Datos ya procesados: {len(agg_groups)} grupos disponibles")
            
            if st.button("🔄 Reprocesar con Nuevas Reglas"):
                # Limpiar cache y reprocesar
                del st.session_state[data_key]
                st.rerun()
        
        # Si tenemos datos procesados, mostrarlos
        if data_key in st.session_state:
            agg_groups = st.session_state[data_key]
            
            # Mostrar distribución de clusters
            if 'cluster_grp' in agg_groups.columns:
                st.markdown("#### 📊 Distribución de Clusters")
                cluster_counts = agg_groups['cluster_grp'].value_counts().sort_index()
                colors = get_cluster_colors()
                cluster_names = colors['names']
                
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
                for col in [ESPECIE_COLUMN, VAR_COLUMN, 'temporada', COL_CAMPO, COL_PORTAINJERTO, 
                           'brix_promedio', 'banda_brix', 'acidez_primer_fruto', 'banda_acidez',
                           'mejillas_promedio', 'banda_mejillas', 'firmeza_punto_debil', 'banda_firmeza_punto',
                           'grupo_id', 'grupo_key', 'grupo_key_detalle', 'suma_bandas', 'cluster_grp', 'n_registros']:
                    if col in agg_groups.columns:
                        display_cols.append(col)
                
                if display_cols:
                    # Usar colores unificados del sistema
                    color_cluster = get_cluster_style_function()
                    
                    styled_df = agg_groups[display_cols].style.map(
                        color_cluster, subset=['cluster_grp'] if 'cluster_grp' in display_cols else []
                    )
                    
                    st.dataframe(styled_df, use_container_width=True)
                
                # Gráfico PCA por Cluster agrupado por Variedad
                st.markdown("#### 🎯 Análisis PCA por Cluster y Variedad")
                
                # Preparar datos para PCA
                numeric_cols = ['brix_promedio', 'acidez_primer_fruto', 'mejillas_promedio', 'firmeza_punto_debil']
                available_numeric = [col for col in numeric_cols if col in agg_groups.columns]
                
                if len(available_numeric) >= 2 and 'cluster_grp' in agg_groups.columns:
                    try:
                        # Verificar si xgboost está instalado, si no, usar sklearn
                        try:
                            from sklearn.decomposition import PCA
                            from sklearn.preprocessing import StandardScaler
                            sklearn_available = True
                        except ImportError:
                            sklearn_available = False
                            st.warning("⚠️ Instalando librerías necesarias para PCA...")
                        
                        if sklearn_available:
                            # Filtrar datos válidos para PCA
                            pca_data = agg_groups[available_numeric + ['cluster_grp', VAR_COLUMN]].dropna()
                            
                            if len(pca_data) > 3:
                                # Standardizar datos
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(pca_data[available_numeric])
                                
                                # Aplicar PCA
                                pca = PCA(n_components=2, random_state=42)
                                pca_coords = pca.fit_transform(X_scaled)
                                
                                # Crear DataFrame con resultados PCA
                                pca_df = pca_data.copy()
                                pca_df['PCA1'] = pca_coords[:, 0]
                                pca_df['PCA2'] = pca_coords[:, 1]
                                
                                # Usar paleta de colores pastel estándar
                                cluster_color_map = get_plotly_color_map()

                                # Gráfico PCA por cluster con colores pastel
                                fig_pca = px.scatter(
                                    pca_df, x='PCA1', y='PCA2',
                                    color='cluster_grp',
                                    symbol=VAR_COLUMN,
                                    title=f"Análisis PCA por Cluster y Variedad - {titulo_especie}",
                                    labels={
                                        'PCA1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)',
                                        'PCA2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)',
                                        'cluster_grp': 'Cluster',
                                        VAR_COLUMN: 'Variedad'
                                    },
                                    color_discrete_map=cluster_color_map,
                                    hover_data=[VAR_COLUMN, 'brix_promedio', 'mejillas_promedio'] if 'brix_promedio' in pca_df.columns else None
                                )
                                
                                fig_pca.update_layout(
                                    height=600,
                                    showlegend=True,
                                    legend=dict(
                                        title="Cluster (Calidad)",
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )
                                
                                st.plotly_chart(fig_pca, use_container_width=True)
                                
                                # Mostrar información de componentes principales
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Varianza explicada PC1", f"{pca.explained_variance_ratio_[0]:.1%}")
                                    st.metric("Varianza total explicada", f"{pca.explained_variance_ratio_.sum():.1%}")
                                
                                with col2:
                                    st.metric("Varianza explicada PC2", f"{pca.explained_variance_ratio_[1]:.1%}")
                                    st.metric("Muestras analizadas", len(pca_df))
                                
                                # Tabla de contribuciones de variables
                                components_df = pd.DataFrame(
                                    pca.components_.T,
                                    columns=['PC1', 'PC2'],
                                    index=available_numeric
                                )
                                components_df['Importancia_Total'] = np.sqrt(components_df['PC1']**2 + components_df['PC2']**2)
                                components_df = components_df.sort_values('Importancia_Total', ascending=False)
                                
                                st.markdown("##### 📊 Contribución de Variables a los Componentes Principales")
                                st.dataframe(components_df.round(3), use_container_width=True)
                                
                            else:
                                st.warning("⚠️ No hay suficientes datos válidos para realizar análisis PCA.")
                        else:
                            st.error("❌ Scikit-learn no está disponible. PCA no se puede generar.")
                            
                    except Exception as e:
                        st.error(f"❌ Error al generar PCA: {str(e)}")
                else:
                    st.warning("⚠️ No hay suficientes columnas numéricas o datos de cluster para PCA.")
                
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
    else:
        st.warning("No hay datos disponibles para procesar.")