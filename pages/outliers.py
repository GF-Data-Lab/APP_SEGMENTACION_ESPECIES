"""
Página de Detección de Outliers - Análisis por Especie

Esta página permite detectar valores atípicos en los datos de carozos,
con la capacidad de analizar cada especie por separado para mayor precisión.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from pandas.api import types as ptypes

from common_styles import configure_page, generarMenu, get_cluster_colors
from data_columns import COL_ESPECIE, COL_VARIEDAD


def _coerce_numeric_columns(df: pd.DataFrame):
    """Convert object columns that look numeric into numeric dtype."""
    df_numeric = df.copy()
    numeric_cols = list(df_numeric.select_dtypes(include=[np.number]).columns)
    converted_cols = []

    for col in df_numeric.columns:
        if col in numeric_cols:
            continue

        series = df_numeric[col]
        if not (ptypes.is_object_dtype(series) or ptypes.is_string_dtype(series)):
            continue

        min_required = min(len(series), max(3, int(0.2 * len(series))))

        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() < min_required:
            cleaned = series.astype(str)
            cleaned = cleaned.str.replace(r"\s+", "", regex=True)
            cleaned = cleaned.str.replace(',', '.', regex=False)
            numeric = pd.to_numeric(cleaned, errors="coerce")

        if numeric.notna().sum() >= min_required:
            df_numeric[col] = numeric
            converted_cols.append(col)

    numeric_candidates = sorted(set(numeric_cols + converted_cols))
    return df_numeric, numeric_candidates, converted_cols


def detectar_outliers_serie(serie, metodo, z_threshold=3.0, iqr_factor=1.5):
    """
    Detecta outliers en una serie usando el método especificado.

    Args:
        serie: Serie de pandas con datos numéricos
        metodo: 'Z-Score', 'IQR', o 'Ambos métodos'
        z_threshold: Umbral para Z-Score
        iqr_factor: Factor para IQR

    Returns:
        Serie booleana indicando outliers
    """
    outliers = pd.Series(False, index=serie.index)

    if metodo in ["Z-Score", "Ambos métodos"]:
        valid = serie.dropna()
        if len(valid) >= 3:  # Mínimo para Z-score confiable
            z_scores = pd.Series(stats.zscore(valid, nan_policy='omit'), index=valid.index)
            outliers_z = z_scores.abs() > z_threshold
            outliers.loc[outliers_z.index] |= outliers_z

    if metodo in ["IQR (Rango Intercuartílico)", "Ambos métodos"]:
        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:  # Evitar división por cero
            lower = q1 - iqr_factor * iqr
            upper = q3 + iqr_factor * iqr
            outliers_iqr = (serie < lower) | (serie > upper)
            outliers |= outliers_iqr.fillna(False)

    return outliers.fillna(False)


# Configuración de la página
configure_page("Detección de Outliers", "🎯")

# Generar el menú
generarMenu()

# Título principal
st.title("🎯 Detección de Outliers")

st.markdown("""
## 📊 Análisis de Datos Atípicos por Especie

Esta herramienta permite identificar y filtrar **valores atípicos** en los datos de calidad,
con la capacidad de analizar cada especie por separado para mayor precisión estadística.

### 🎯 Importancia del Análisis por Especie:
- **Precisión estadística:** Cada especie tiene rangos de valores naturales diferentes
- **Evita falsos positivos:** Un valor normal para una especie puede ser atípico para otra
- **Mejor limpieza de datos:** Filtros específicos por tipo de fruto
""")

# Verificar si hay datos cargados
if "carozos_df" not in st.session_state:
    st.warning("⚠️ **No hay datos cargados**. Por favor, ve a la página de 'Carga de archivos' primero.")
    st.stop()

# Preparar dataframes
df_original = st.session_state["carozos_df"].copy()
df_numeric, numeric_candidates, converted_columns = _coerce_numeric_columns(df_original)

# Verificar si existe columna de especie
especies_disponibles = []
if COL_ESPECIE in df_original.columns:
    especies_disponibles = df_original[COL_ESPECIE].dropna().unique().tolist()

if converted_columns:
    st.info(f"📊 **Columnas convertidas automáticamente:** {', '.join(converted_columns)}")

if not numeric_candidates:
    st.error("❌ No se encontraron columnas numéricas disponibles para analizar.")
    st.stop()

# Configuración principal en la página
st.markdown("## ⚙️ Configuración de Análisis")

# Crear columnas para organizar los controles
col_config1, col_config2, col_config3 = st.columns([1, 1, 1])

with col_config1:
    # Selector de especie para análisis separado
    if especies_disponibles:
        st.markdown("### 🍑 Filtro por Especie")
        analisis_por_especie = st.checkbox(
            "📊 Analizar por especie separadamente",
            value=True,
            help="Recomendado: Cada especie tiene características diferentes"
        )

        if analisis_por_especie:
            especie_seleccionada = st.selectbox(
                "🔍 Seleccionar Especie:",
                ["Todas"] + especies_disponibles,
                help="Selecciona una especie específica o 'Todas' para comparar"
            )
        else:
            especie_seleccionada = "Todas"
            st.warning("⚠️ Análisis conjunto puede ser menos preciso")
    else:
        analisis_por_especie = False
        especie_seleccionada = "Todas"
        st.info("ℹ️ No se detectó columna de especie")

with col_config2:
    st.markdown("### 🔧 Método de Detección")
    metodo = st.selectbox(
        "Método de detección:",
        ["Z-Score", "IQR (Rango Intercuartílico)", "Ambos métodos"]
    )

    # Inicializar variables con valores por defecto
    z_threshold = 3.0
    iqr_factor = 1.5

    if metodo in ["Z-Score", "Ambos métodos"]:
        z_threshold = st.slider(
            "Umbral Z-Score:",
            min_value=2.0,
            max_value=4.0,
            value=3.0,
            step=0.1,
            help="Valores con Z-Score > umbral se consideran outliers"
        )

    if metodo in ["IQR (Rango Intercuartílico)", "Ambos métodos"]:
        iqr_factor = st.slider(
            "Factor IQR:",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Factor para calcular los límites (Q1 - factor*IQR, Q3 + factor*IQR)"
        )

with col_config3:
    st.markdown("### 📋 Columnas a Analizar")
    default_candidates = [col for col in ["Firmeza (lb)", "Acidez", "Brix"] if col in numeric_candidates]
    if not default_candidates:
        default_candidates = numeric_candidates[:3]

    columnas_analizar = st.multiselect(
        "Columnas a analizar:",
        options=numeric_candidates,
        default=default_candidates,
        help="Selecciona las métricas numéricas que deseas revisar."
    )

# Información en sidebar (solo informativa)
with st.sidebar:
    st.markdown("### 📊 Información del Análisis")
    if especies_disponibles:
        st.info(f"🍑 **Especies disponibles:** {len(especies_disponibles)}")
        for esp in especies_disponibles:
            count = len(df_numeric[df_numeric[COL_ESPECIE] == esp]) if COL_ESPECIE in df_numeric.columns else 0
            st.metric(f"📊 {esp}", f"{count:,} registros")

    st.markdown("### 🔧 Configuración Actual")
    st.info(f"**Método:** {metodo}")
    if metodo in ["Z-Score", "Ambos métodos"]:
        st.info(f"**Z-Score:** {z_threshold}")
    if metodo in ["IQR (Rango Intercuartílico)", "Ambos métodos"]:
        st.info(f"**Factor IQR:** {iqr_factor}")

    if columnas_analizar:
        st.info(f"**Columnas:** {len(columnas_analizar)} seleccionadas")

# Verificar que se hayan seleccionado columnas
if not columnas_analizar:
    st.warning("⚠️ **Selecciona al menos una columna** para analizar en la configuración anterior.")
    st.stop()

# Separador visual
st.markdown("---")

# Aplicar filtro de especie si está seleccionado
if especies_disponibles and analisis_por_especie and especie_seleccionada != "Todas":
    df_analisis = df_numeric[df_numeric[COL_ESPECIE] == especie_seleccionada].copy()
    titulo_especie = f" - {especie_seleccionada}"
    st.success(f"📊 **Analizando outliers específicamente para:** {especie_seleccionada}")
else:
    df_analisis = df_numeric.copy()
    titulo_especie = " - Todas las especies" if especies_disponibles else ""
    if especies_disponibles and not analisis_por_especie:
        st.warning("⚠️ **Análisis conjunto de especies:** Los resultados pueden ser menos precisos debido a las diferencias naturales entre especies.")

# Tabs principales para el análisis
st.markdown("## 📊 Análisis de Outliers")
tab1, tab2, tab3, tab4 = st.tabs(["📈 Visualización", "🧮 Detección", "📊 Estadísticas", "💾 Exportar"])

with tab1:
    st.markdown(f"### 📈 Distribución de Datos{titulo_especie}")

    # Mostrar información del dataset actual
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("📊 Registros Total", len(df_analisis))
    with col_info2:
        st.metric("🔢 Variables Seleccionadas", len(columnas_analizar))
    with col_info3:
        if especies_disponibles and especie_seleccionada != "Todas":
            st.metric("🍑 Especie", especie_seleccionada)
        else:
            st.metric("🍑 Especies", len(especies_disponibles) if especies_disponibles else "N/A")

    # Si estamos analizando todas las especies, mostrar comparación
    if columnas_analizar:
        if especies_disponibles and especie_seleccionada == "Todas" and analisis_por_especie:
            st.markdown("#### 🔍 Comparación entre Especies")

            st.info("💡 **Observa las diferencias:** Nota cómo cada especie tiene rangos naturales diferentes. Esto justifica el análisis separado.")

            for col in columnas_analizar[:2]:  # Limitar a 2 columnas para no sobrecargar
                col1, col2 = st.columns(2)

                with col1:
                    # Histograma por especie
                    fig_hist = px.histogram(
                        df_analisis,
                        x=col,
                        color=COL_ESPECIE,
                        title=f"Distribución de {col} por Especie",
                        nbins=20,
                        barmode='overlay',
                        opacity=0.7
                    )
                    fig_hist.update_layout(height=350)
                    st.plotly_chart(fig_hist, use_container_width=True)

                with col2:
                    # Box plot por especie
                    fig_box = px.box(
                        df_analisis,
                        x=COL_ESPECIE,
                        y=col,
                        title=f"Box Plot de {col} por Especie"
                    )
                    fig_box.update_layout(height=350)
                    st.plotly_chart(fig_box, use_container_width=True)

            # Estadísticas comparativas
            st.markdown("#### 📊 Estadísticas por Especie")
            stats_df = df_analisis.groupby(COL_ESPECIE)[columnas_analizar].agg(['mean', 'std', 'min', 'max']).round(3)
            st.dataframe(stats_df, use_container_width=True)

        else:
            # Análisis para especie específica o sin separación
            for col in columnas_analizar:
                col1, col2 = st.columns(2)

                with col1:
                    fig_hist = px.histogram(
                        df_analisis,
                        x=col,
                        title=f"Distribución de {col}{titulo_especie}",
                        nbins=30,
                        color_discrete_sequence=["#FF6B6B"]  # Usar color coral pastel
                    )
                    fig_hist.update_layout(height=300)
                    st.plotly_chart(fig_hist, use_container_width=True)

                with col2:
                    fig_box = px.box(
                        df_analisis,
                        y=col,
                        title=f"Box Plot de {col}{titulo_especie}",
                        color_discrete_sequence=["#FF6B6B"]  # Usar color coral pastel
                    )
                    fig_box.update_layout(height=300)
                    st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Selecciona columnas para analizar en el panel lateral.")

with tab2:
    st.markdown(f"### 🧮 Identificación de Outliers{titulo_especie}")

    if columnas_analizar:
        # Si analizamos por especie, mostrar estadísticas de cada especie
        if especies_disponibles and analisis_por_especie and especie_seleccionada == "Todas":
            st.markdown("#### 📊 Análisis por Especie Individual")

            outliers_totales = pd.Series(False, index=df_numeric.index)
            outliers_detalles = {}
            resultados_especies = {}

            for especie in especies_disponibles:
                st.markdown(f"##### 🍑 {especie}")
                df_especie = df_numeric[df_numeric[COL_ESPECIE] == especie]

                if len(df_especie) < 5:
                    st.warning(f"⚠️ Muy pocos datos para {especie} ({len(df_especie)} registros). Se recomienda mínimo 5.")
                    continue

                outliers_especie = pd.Series(False, index=df_especie.index)
                outliers_detalle_especie = {}

                for col in columnas_analizar:
                    serie = df_especie[col]
                    outliers_col = detectar_outliers_serie(serie, metodo,
                                                          z_threshold if metodo in ["Z-Score", "Ambos métodos"] else 3.0,
                                                          iqr_factor if metodo in ["IQR (Rango Intercuartílico)", "Ambos métodos"] else 1.5)
                    outliers_especie |= outliers_col
                    outliers_detalle_especie[col] = outliers_col

                # Métricas por especie
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(f"📊 Registros", len(df_especie))
                with col2:
                    num_outliers_esp = int(outliers_especie.sum())
                    porcentaje_esp = (num_outliers_esp / len(df_especie)) * 100 if len(df_especie) > 0 else 0
                    st.metric(f"🎯 Outliers", f"{num_outliers_esp}")
                with col3:
                    st.metric(f"📈 % Outliers", f"{porcentaje_esp:.1f}%")
                with col4:
                    st.metric(f"✅ Datos limpios", len(df_especie) - num_outliers_esp)

                # Guardar resultados
                resultados_especies[especie] = {
                    'total': len(df_especie),
                    'outliers': num_outliers_esp,
                    'porcentaje': porcentaje_esp
                }

                # Acumular outliers totales
                outliers_totales = outliers_totales.reindex(
                    outliers_totales.index.union(outliers_especie.index),
                    fill_value=False
                )
                outliers_totales.loc[outliers_especie.index] |= outliers_especie

                for col in columnas_analizar:
                    if col not in outliers_detalles:
                        outliers_detalles[col] = pd.Series(False, index=df_numeric.index)
                    outliers_detalles[col] = outliers_detalles[col].reindex(
                        outliers_detalles[col].index.union(outliers_detalle_especie[col].index),
                        fill_value=False
                    )
                    outliers_detalles[col].loc[outliers_detalle_especie[col].index] |= outliers_detalle_especie[col]

            # Resumen comparativo
            st.markdown("#### 📊 Resumen Comparativo por Especie")
            if resultados_especies:
                df_resumen = pd.DataFrame(resultados_especies).T
                df_resumen.columns = ['Total Registros', 'Outliers Detectados', '% Outliers']
                st.dataframe(df_resumen.round(1), use_container_width=True)

        else:
            # Análisis para especie específica
            outliers_totales = pd.Series(False, index=df_analisis.index)
            outliers_detalles = {}

            for col in columnas_analizar:
                serie = df_analisis[col]
                outliers_col = detectar_outliers_serie(serie, metodo,
                                                      z_threshold if metodo in ["Z-Score", "Ambos métodos"] else 3.0,
                                                      iqr_factor if metodo in ["IQR (Rango Intercuartílico)", "Ambos métodos"] else 1.5)
                outliers_totales |= outliers_col
                outliers_detalles[col] = outliers_col

        # Resumen general
        st.markdown("#### 📊 Resumen General")
        total_registros = len(df_analisis)
        num_outliers = int(outliers_totales.sum())
        porcentaje = (num_outliers / total_registros) * 100 if total_registros > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Registros analizados", total_registros)
        with col2:
            st.metric(
                "🎯 Outliers detectados",
                f"{num_outliers}",
                f"{porcentaje:.1f}%",
                delta_color="inverse"
            )
        with col3:
            st.metric(
                "✅ Datos limpios",
                total_registros - num_outliers,
                f"{((total_registros - num_outliers) / total_registros * 100):.1f}%"
            )

        # Detalle por columna
        st.markdown("#### 📋 Detalle por Variable")
        for col in columnas_analizar:
            outliers_col = outliers_detalles[col]
            num_outliers_col = int(outliers_col.sum())
            porcentaje_col = (num_outliers_col / len(df_analisis)) * 100 if len(df_analisis) > 0 else 0

            with st.expander(f"📊 {col} - {num_outliers_col} outliers ({porcentaje_col:.1f}%)"):
                if num_outliers_col > 0:
                    # Mostrar outliers de esta columna
                    outliers_data = df_analisis[outliers_col]

                    if especies_disponibles and COL_ESPECIE in outliers_data.columns:
                        outliers_display = outliers_data[[COL_ESPECIE, col]].copy()
                    else:
                        outliers_display = outliers_data[[col]].copy()

                    st.dataframe(outliers_display, use_container_width=True)

                    # Estadísticas de la columna
                    serie = df_analisis[col].dropna()
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    with col_stats1:
                        st.metric("📊 Media", f"{serie.mean():.3f}")
                    with col_stats2:
                        st.metric("📈 Desv. Estándar", f"{serie.std():.3f}")
                    with col_stats3:
                        st.metric("🔢 Rango", f"{serie.min():.2f} - {serie.max():.2f}")
                else:
                    st.info("✅ No se detectaron outliers en esta variable.")

        # Guardar resultados en session state
        st.session_state["outliers_results"] = {
            'outliers_mask': outliers_totales,
            'outliers_detalles': outliers_detalles,
            'df_limpio': df_analisis[~outliers_totales],
            'metodo': metodo,
            'especie': especie_seleccionada,
            'columnas': columnas_analizar
        }

    else:
        st.info("Selecciona columnas para analizar en el panel lateral.")

with tab3:
    st.markdown("### 📊 Estadísticas Comparativas")

    if "outliers_results" in st.session_state and columnas_analizar:
        results = st.session_state["outliers_results"]
        df_limpio = results['df_limpio']

        st.markdown("#### 📋 Comparación: Antes vs Después")

        for col in columnas_analizar:
            st.markdown(f"##### 📊 {col}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📊 Datos Originales**")
                serie_original = df_analisis[col].dropna()
                if len(serie_original) > 0:
                    st.metric("📈 Media", f"{serie_original.mean():.3f}")
                    st.metric("📊 Mediana", f"{serie_original.median():.3f}")
                    st.metric("📉 Desv. Estándar", f"{serie_original.std():.3f}")
                    st.metric("🔢 Registros", len(serie_original))

            with col2:
                st.markdown("**✅ Datos Limpios**")
                serie_limpia = df_limpio[col].dropna()
                if len(serie_limpia) > 0:
                    st.metric("📈 Media", f"{serie_limpia.mean():.3f}")
                    st.metric("📊 Mediana", f"{serie_limpia.median():.3f}")
                    st.metric("📉 Desv. Estándar", f"{serie_limpia.std():.3f}")
                    st.metric("🔢 Registros", len(serie_limpia))

            # Gráfico comparativo
            fig_comp = go.Figure()

            fig_comp.add_trace(go.Histogram(
                x=serie_original,
                name="Original",
                opacity=0.7,
                marker_color='#FF6B6B'
            ))

            fig_comp.add_trace(go.Histogram(
                x=serie_limpia,
                name="Limpio",
                opacity=0.7,
                marker_color='#7FB069'
            ))

            fig_comp.update_layout(
                title=f"Comparación de Distribuciones - {col}",
                barmode='overlay',
                height=300
            )

            st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Ejecuta primero la detección de outliers para ver las estadísticas.")

with tab4:
    st.markdown("### 💾 Exportar Resultados")

    if "outliers_results" in st.session_state:
        results = st.session_state["outliers_results"]
        df_limpio = results['df_limpio']
        metodo_usado = results['metodo']
        especie_analizada = results['especie']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📥 Descargar Datos Limpios")

            # CSV de datos limpios
            csv_limpio = df_limpio.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Datos Limpios (CSV)",
                data=csv_limpio,
                file_name=f"datos_limpios_{especie_analizada}_{metodo_usado}.csv",
                mime="text/csv"
            )

            st.metric("📊 Registros en datos limpios", len(df_limpio))
            st.metric("✅ Datos removidos", len(df_analisis) - len(df_limpio))

        with col2:
            st.markdown("#### 🎯 Descargar Solo Outliers")

            # CSV de outliers
            outliers_mask = results['outliers_mask']
            df_outliers = df_analisis[outliers_mask]

            if len(df_outliers) > 0:
                csv_outliers = df_outliers.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Outliers (CSV)",
                    data=csv_outliers,
                    file_name=f"outliers_detectados_{especie_analizada}_{metodo_usado}.csv",
                    mime="text/csv"
                )

                st.metric("🎯 Outliers detectados", len(df_outliers))
                st.metric("📈 % del total", f"{(len(df_outliers)/len(df_analisis)*100):.1f}%")
            else:
                st.info("✅ No se detectaron outliers para exportar.")

        # Guardar datos limpios en session state para uso en otras páginas
        if st.button("💾 Guardar Datos Limpios para Análisis Posterior", type="primary"):
            st.session_state["carozos_df_filtered"] = df_limpio
            st.success("✅ Datos limpios guardados para uso en otras páginas de la aplicación.")

            # Mostrar información de lo que se guardó
            st.info(f"""
            **📊 Datos guardados:**
            - **Especie analizada:** {especie_analizada}
            - **Método usado:** {metodo_usado}
            - **Registros originales:** {len(df_analisis):,}
            - **Registros limpios:** {len(df_limpio):,}
            - **Outliers removidos:** {len(df_analisis) - len(df_limpio):,}
            """)
    else:
        st.info("Ejecuta primero la detección de outliers para generar datos para exportar.")

# Información adicional
with st.expander("ℹ️ Información sobre los Métodos"):
    st.markdown("""
    ### 🔍 Métodos de Detección de Outliers:

    **Z-Score:**
    - Mide cuántas desviaciones estándar se aleja un valor de la media
    - Outlier si |Z-Score| > umbral (típicamente 3.0)
    - Asume distribución normal
    - Sensible a outliers extremos

    **IQR (Rango Intercuartílico):**
    - Basado en cuartiles (Q1, Q3)
    - Outlier si valor < Q1 - 1.5*IQR o valor > Q3 + 1.5*IQR
    - Más robusto que Z-Score
    - No asume distribución específica

    ### 🍑 Importancia del Análisis por Especie:

    **¿Por qué separar por especie?**
    - Cada especie tiene rangos naturales diferentes
    - Un valor normal para ciruelas puede ser atípico para nectarinas
    - Evita clasificación errónea de datos normales como outliers

    **Ejemplo:**
    - Firmeza normal para ciruela: 4-8 lb
    - Firmeza normal para nectarina: 6-12 lb
    - Un valor de 5 lb sería normal para ciruela pero bajo para nectarina
    """)

# Tips de uso
st.markdown("---")
st.markdown("""
### 💡 **Recomendaciones de Uso:**

1. **✅ Usar análisis por especie** cuando tengas múltiples especies
2. **📊 Revisar distribuciones** antes de aplicar filtros
3. **🎯 Comenzar con parámetros conservadores** (Z-Score 3.0, IQR 1.5)
4. **📋 Validar outliers manualmente** antes de remover
5. **💾 Guardar datos limpios** para análisis posterior
""")