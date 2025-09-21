"""
Página de Análisis Exploratorio - Visualizaciones y Patrones

Esta página proporciona herramientas avanzadas de análisis exploratorio
para descubrir patrones, correlaciones y insights en los datos de carozos.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

from common_styles import configure_page, generarMenu, get_cluster_colors, get_plotly_color_map, get_plotly_color_sequence
from data_columns import COL_ESPECIE, COL_VARIEDAD, COL_BRIX, COL_ACIDEZ, COL_FECHA_EVALUACION

# Configuración de página
configure_page("🔍 Análisis Exploratorio", "🔬")
generarMenu()

st.title("🔍 Análisis Exploratorio Avanzado")

st.markdown("""
## 🎯 Descubrimiento de Patrones y Insights

Explora tus datos de carozos con herramientas avanzadas de análisis estadístico
y visualización interactiva para descubrir patrones ocultos y relaciones importantes.

### ✨ Funcionalidades:
- **Análisis Estadístico:** Descriptivos, correlaciones, distribuciones
- **Visualizaciones Interactivas:** Scatter plots, heatmaps, box plots
- **Clustering Exploratorio:** t-SNE, PCA, clustering jerárquico
- **Análisis Temporal:** Tendencias y patrones estacionales
- **Detección de Outliers:** Identificación de valores atípicos
""")

# Verificar datos
if "carozos_df" not in st.session_state:
    st.warning("⚠️ No hay datos cargados. Por favor, ve a 'Carga de Archivos' primero.")
    st.info("📋 **Para usar esta página:**\n1. Carga datos en 'Carga de Archivos'\n2. Opcionalmente ejecuta segmentación\n3. Explora patrones aquí")
    st.stop()

df = st.session_state["carozos_df"]

# Información general de los datos
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Total Registros", len(df))
with col2:
    especies = df[COL_ESPECIE].nunique() if COL_ESPECIE in df.columns else 0
    st.metric("🍑 Especies", especies)
with col3:
    variedades = df[COL_VARIEDAD].nunique() if COL_VARIEDAD in df.columns else 0
    st.metric("🌱 Variedades", variedades)
with col4:
    columnas_numericas = len(df.select_dtypes(include=[np.number]).columns)
    st.metric("🔢 Variables Numéricas", columnas_numericas)

# Sidebar para filtros
st.sidebar.markdown("## 🔍 Filtros de Análisis")

# Filtro por especie
if COL_ESPECIE in df.columns:
    especies_disponibles = ['Todas'] + list(df[COL_ESPECIE].dropna().unique())
    especie_filtro = st.sidebar.selectbox("🍑 Filtrar por Especie", especies_disponibles)

    if especie_filtro != 'Todas':
        df = df[df[COL_ESPECIE] == especie_filtro]

# Filtro por variedad
if COL_VARIEDAD in df.columns and len(df) > 0:
    variedades_disponibles = ['Todas'] + list(df[COL_VARIEDAD].dropna().unique())
    variedad_filtro = st.sidebar.selectbox("🌱 Filtrar por Variedad", variedades_disponibles)

    if variedad_filtro != 'Todas':
        df = df[df[COL_VARIEDAD] == variedad_filtro]

# Obtener columnas numéricas
columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
columnas_numericas = [col for col in columnas_numericas if not col.startswith('Unnamed')]

if len(columnas_numericas) < 2:
    st.error("❌ No hay suficientes columnas numéricas para el análisis.")
    st.stop()

# Tabs principales
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Estadísticas Descriptivas",
    "🔗 Correlaciones",
    "📈 Distribuciones",
    "🎯 Clustering Exploratorio",
    "⏰ Análisis Temporal"
])

with tab1:
    st.markdown("## 📊 Estadísticas Descriptivas")

    # Estadísticas generales
    st.markdown("### 📋 Resumen Estadístico")

    df_numerico = df[columnas_numericas].dropna()
    estadisticas = df_numerico.describe()

    # Formatear estadísticas
    st.dataframe(estadisticas.round(3), use_container_width=True)

    # Métricas destacadas
    st.markdown("### 🎯 Métricas Clave")

    cols = st.columns(min(4, len(columnas_numericas)))
    for i, col in enumerate(columnas_numericas[:4]):
        with cols[i]:
            media = df_numerico[col].mean()
            std = df_numerico[col].std()
            cv = (std / media * 100) if media != 0 else 0

            st.metric(
                label=f"📊 {col}",
                value=f"{media:.2f}",
                delta=f"CV: {cv:.1f}%",
                help=f"Coeficiente de Variación: {cv:.1f}%"
            )

    # Histogramas por variable
    st.markdown("### 📈 Distribución por Variable")

    col_seleccionada = st.selectbox(
        "🔍 Seleccionar Variable",
        columnas_numericas,
        key="hist_var"
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_hist = px.histogram(
            df_numerico,
            x=col_seleccionada,
            nbins=30,
            title=f"📊 Distribución de {col_seleccionada}",
            color_discrete_sequence=['#7FB069']
        )

        fig_hist.update_layout(
            showlegend=False,
            height=400,
            bargap=0.1
        )

        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        # Estadísticas de la variable seleccionada
        serie = df_numerico[col_seleccionada]

        st.markdown(f"#### 📊 {col_seleccionada}")
        st.metric("📈 Media", f"{serie.mean():.3f}")
        st.metric("📊 Mediana", f"{serie.median():.3f}")
        st.metric("📉 Desv. Estándar", f"{serie.std():.3f}")
        st.metric("🔢 Valores Únicos", len(serie.unique()))

        # Test de normalidad
        _, p_value = stats.normaltest(serie.dropna())
        if p_value > 0.05:
            st.success("✅ Distribución Normal")
        else:
            st.warning("⚠️ No Normal")

with tab2:
    st.markdown("## 🔗 Análisis de Correlaciones")

    # Matriz de correlación
    df_corr = df[columnas_numericas].corr()

    # Heatmap de correlaciones
    fig_corr = px.imshow(
        df_corr,
        color_continuous_scale='RdBu',
        aspect='auto',
        title="🔗 Matriz de Correlaciones",
        labels=dict(color="Correlación")
    )

    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Correlaciones más fuertes
    st.markdown("### 🎯 Correlaciones Más Significativas")

    # Obtener correlaciones sin la diagonal
    corr_values = []
    for i in range(len(df_corr.columns)):
        for j in range(i+1, len(df_corr.columns)):
            corr_values.append({
                'Variable 1': df_corr.columns[i],
                'Variable 2': df_corr.columns[j],
                'Correlación': df_corr.iloc[i, j],
                'Abs_Correlación': abs(df_corr.iloc[i, j])
            })

    df_corr_sorted = pd.DataFrame(corr_values).sort_values('Abs_Correlación', ascending=False)

    # Mostrar top correlaciones
    st.dataframe(
        df_corr_sorted.head(10)[['Variable 1', 'Variable 2', 'Correlación']].round(3),
        use_container_width=True
    )

    # Scatter plot de correlación seleccionada
    st.markdown("### 📊 Explorar Correlación")

    col1, col2 = st.columns(2)
    with col1:
        var_x = st.selectbox("📊 Variable X", columnas_numericas, key="corr_x")
    with col2:
        var_y = st.selectbox("📈 Variable Y", columnas_numericas, index=1, key="corr_y")

    if var_x != var_y:
        # Calcular correlación
        correlacion = df[var_x].corr(df[var_y])

        # Scatter plot
        fig_scatter = px.scatter(
            df,
            x=var_x,
            y=var_y,
            title=f"📊 {var_x} vs {var_y} (r = {correlacion:.3f})",
            trendline="ols",
            color_discrete_sequence=['#FF9F40']
        )

        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Interpretación de la correlación
        if abs(correlacion) > 0.7:
            st.success(f"🌟 Correlación fuerte: {correlacion:.3f}")
        elif abs(correlacion) > 0.3:
            st.info(f"👍 Correlación moderada: {correlacion:.3f}")
        else:
            st.warning(f"⚠️ Correlación débil: {correlacion:.3f}")

with tab3:
    st.markdown("## 📈 Análisis de Distribuciones")

    # Box plots comparativos
    st.markdown("### 📦 Box Plots por Especie/Variedad")

    variable_boxplot = st.selectbox(
        "🔍 Variable para Box Plot",
        columnas_numericas,
        key="boxplot_var"
    )

    # Selector de agrupación
    opciones_grupo = []
    if COL_ESPECIE in df.columns:
        opciones_grupo.append(COL_ESPECIE)
    if COL_VARIEDAD in df.columns:
        opciones_grupo.append(COL_VARIEDAD)

    if opciones_grupo:
        grupo_boxplot = st.selectbox(
            "🎯 Agrupar por",
            opciones_grupo,
            key="boxplot_grupo"
        )

        fig_box = px.box(
            df,
            x=grupo_boxplot,
            y=variable_boxplot,
            title=f"📦 Distribución de {variable_boxplot} por {grupo_boxplot}",
            color_discrete_sequence=get_plotly_color_sequence()
        )

        fig_box.update_layout(height=500)
        st.plotly_chart(fig_box, use_container_width=True)

    # Violin plots
    st.markdown("### 🎻 Violin Plots - Distribución Detallada")

    if opciones_grupo:
        fig_violin = px.violin(
            df,
            x=grupo_boxplot,
            y=variable_boxplot,
            title=f"🎻 Distribución Detallada de {variable_boxplot}",
            color_discrete_sequence=['#FF6B6B']
        )

        fig_violin.update_layout(height=500)
        st.plotly_chart(fig_violin, use_container_width=True)

    # Comparación de distribuciones
    st.markdown("### 📊 Comparación de Distribuciones")

    col1, col2 = st.columns(2)
    with col1:
        var_dist1 = st.selectbox("📊 Variable 1", columnas_numericas, key="dist1")
    with col2:
        var_dist2 = st.selectbox("📈 Variable 2", columnas_numericas, index=1, key="dist2")

    # Histogramas superpuestos
    fig_dist = go.Figure()

    fig_dist.add_trace(go.Histogram(
        x=df[var_dist1].dropna(),
        name=var_dist1,
        opacity=0.7,
        marker_color='#7FB069'
    ))

    fig_dist.add_trace(go.Histogram(
        x=df[var_dist2].dropna(),
        name=var_dist2,
        opacity=0.7,
        marker_color='#FF9F40'
    ))

    fig_dist.update_layout(
        title="📊 Comparación de Distribuciones",
        barmode='overlay',
        height=400
    )

    st.plotly_chart(fig_dist, use_container_width=True)

with tab4:
    st.markdown("## 🎯 Clustering Exploratorio")

    # Selección de variables para clustering
    st.markdown("### 🔧 Configuración del Análisis")

    variables_clustering = st.multiselect(
        "🎯 Seleccionar Variables para Clustering",
        columnas_numericas,
        default=columnas_numericas[:4] if len(columnas_numericas) >= 4 else columnas_numericas,
        key="clustering_vars"
    )

    if len(variables_clustering) >= 2:
        # Preparar datos
        df_clustering = df[variables_clustering].dropna()

        if len(df_clustering) >= 10:
            # Estandarizar datos
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_clustering)

            # Tabs para diferentes técnicas
            tab_pca, tab_tsne, tab_hier = st.tabs(["📊 PCA", "🎯 t-SNE", "🌳 Clustering Jerárquico"])

            with tab_pca:
                st.markdown("#### 📊 Análisis de Componentes Principales")

                # PCA
                pca = PCA(n_components=min(3, len(variables_clustering)), random_state=42)
                pca_result = pca.fit_transform(X_scaled)

                # Varianza explicada
                var_exp = pca.explained_variance_ratio_
                var_cum = np.cumsum(var_exp)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📊 Varianza PC1", f"{var_exp[0]*100:.1f}%")
                    if len(var_exp) > 1:
                        st.metric("📈 Varianza PC2", f"{var_exp[1]*100:.1f}%")

                with col2:
                    st.metric("🎯 Varianza Total", f"{var_cum[-1]*100:.1f}%")
                    if len(var_exp) > 2:
                        st.metric("📊 Varianza PC3", f"{var_exp[2]*100:.1f}%")

                # Gráfico PCA 2D
                df_pca = pd.DataFrame(pca_result[:, :2], columns=['PC1', 'PC2'])

                # Si hay clusters previos, usarlos para colorear
                if 'cluster' in df.columns or 'cluster_grp' in df.columns:
                    cluster_col = 'cluster_grp' if 'cluster_grp' in df.columns else 'cluster'
                    if len(df[cluster_col].dropna()) == len(df_pca):
                        df_pca['Cluster'] = df[cluster_col].values
                        color_col = 'Cluster'
                        colors = get_plotly_color_map()
                    else:
                        color_col = None
                        colors = None
                else:
                    color_col = None
                    colors = None

                fig_pca = px.scatter(
                    df_pca,
                    x='PC1', y='PC2',
                    color=color_col,
                    color_discrete_map=colors,
                    title="📊 Proyección PCA 2D",
                    labels={'PC1': f'PC1 ({var_exp[0]*100:.1f}%)',
                           'PC2': f'PC2 ({var_exp[1]*100:.1f}%)'}
                )

                st.plotly_chart(fig_pca, use_container_width=True)

                # Contribución de variables
                components = pd.DataFrame(
                    pca.components_[:2].T,
                    columns=['PC1', 'PC2'],
                    index=variables_clustering
                )

                fig_contrib = px.bar(
                    components.reset_index(),
                    x='index',
                    y=['PC1', 'PC2'],
                    title="🔍 Contribución de Variables a PCA",
                    barmode='group'
                )

                st.plotly_chart(fig_contrib, use_container_width=True)

            with tab_tsne:
                st.markdown("#### 🎯 t-SNE (t-Distributed Stochastic Neighbor Embedding)")

                # Parámetros t-SNE
                col1, col2 = st.columns(2)
                with col1:
                    perplexity = st.slider("🎛️ Perplexity", 5, 50, 30)
                with col2:
                    n_iter = st.slider("🔄 Iteraciones", 500, 2000, 1000, step=250)

                if st.button("🚀 Ejecutar t-SNE", key="tsne_run"):
                    with st.spinner("🔄 Ejecutando t-SNE..."):
                        # t-SNE
                        tsne = TSNE(n_components=2, perplexity=perplexity,
                                   n_iter=n_iter, random_state=42)
                        tsne_result = tsne.fit_transform(X_scaled)

                        df_tsne = pd.DataFrame(tsne_result, columns=['t-SNE1', 't-SNE2'])

                        # Colorear por clusters si existen
                        if 'cluster' in df.columns or 'cluster_grp' in df.columns:
                            cluster_col = 'cluster_grp' if 'cluster_grp' in df.columns else 'cluster'
                            if len(df[cluster_col].dropna()) == len(df_tsne):
                                df_tsne['Cluster'] = df[cluster_col].values
                                color_col = 'Cluster'
                                colors = get_plotly_color_map()
                            else:
                                color_col = None
                                colors = None
                        else:
                            color_col = None
                            colors = None

                        fig_tsne = px.scatter(
                            df_tsne,
                            x='t-SNE1', y='t-SNE2',
                            color=color_col,
                            color_discrete_map=colors,
                            title="🎯 Proyección t-SNE 2D"
                        )

                        st.plotly_chart(fig_tsne, use_container_width=True)

                        st.success("✅ t-SNE completado exitosamente!")

            with tab_hier:
                st.markdown("#### 🌳 Clustering Jerárquico")

                if len(df_clustering) <= 100:  # Limitar para performance
                    # Calcular linkage
                    linkage_matrix = linkage(X_scaled, method='ward')

                    # Dendrograma
                    fig_dendro = go.Figure()

                    # Crear dendrograma
                    dendro = dendrogram(linkage_matrix, no_plot=True)

                    # Agregar líneas del dendrograma
                    for i in range(len(dendro['icoord'])):
                        fig_dendro.add_trace(go.Scatter(
                            x=dendro['icoord'][i],
                            y=dendro['dcoord'][i],
                            mode='lines',
                            line=dict(color='#7FB069', width=2),
                            showlegend=False
                        ))

                    fig_dendro.update_layout(
                        title="🌳 Dendrograma - Clustering Jerárquico",
                        xaxis_title="Muestras",
                        yaxis_title="Distancia",
                        height=500
                    )

                    st.plotly_chart(fig_dendro, use_container_width=True)

                    st.info("💡 El dendrograma muestra la jerarquía de agrupamiento. Las líneas más largas indican mayor separación entre clusters.")
                else:
                    st.warning(f"⚠️ Demasiadas muestras ({len(df_clustering)}) para dendrograma. Máximo recomendado: 100.")
        else:
            st.warning("⚠️ Necesitas al menos 10 registros válidos para clustering.")
    else:
        st.warning("⚠️ Selecciona al menos 2 variables para el análisis de clustering.")

with tab5:
    st.markdown("## ⏰ Análisis Temporal")

    if COL_FECHA_EVALUACION in df.columns:
        # Convertir fecha
        df_fecha = df.copy()
        df_fecha[COL_FECHA_EVALUACION] = pd.to_datetime(df_fecha[COL_FECHA_EVALUACION], errors='coerce')
        df_fecha = df_fecha.dropna(subset=[COL_FECHA_EVALUACION])

        if len(df_fecha) > 0:
            # Agregar columnas temporales
            df_fecha['Año'] = df_fecha[COL_FECHA_EVALUACION].dt.year
            df_fecha['Mes'] = df_fecha[COL_FECHA_EVALUACION].dt.month
            df_fecha['Día_Semana'] = df_fecha[COL_FECHA_EVALUACION].dt.day_name()

            # Tendencias temporales
            st.markdown("### 📅 Tendencias Temporales")

            variable_temporal = st.selectbox(
                "🔍 Variable para Análisis Temporal",
                columnas_numericas,
                key="temporal_var"
            )

            # Gráfico temporal
            df_temporal = df_fecha.groupby('Año')[variable_temporal].agg(['mean', 'std', 'count']).reset_index()

            fig_temporal = px.line(
                df_temporal,
                x='Año',
                y='mean',
                title=f"📈 Evolución Temporal de {variable_temporal}",
                labels={'mean': f'Promedio {variable_temporal}'},
                markers=True,
                color_discrete_sequence=['#7FB069']
            )

            # Agregar barras de error
            fig_temporal.add_traces([
                go.Scatter(
                    x=df_temporal['Año'],
                    y=df_temporal['mean'] + df_temporal['std'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                go.Scatter(
                    x=df_temporal['Año'],
                    y=df_temporal['mean'] - df_temporal['std'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(127, 176, 105, 0.3)',
                    showlegend=False,
                    name='Desviación Estándar'
                )
            ])

            st.plotly_chart(fig_temporal, use_container_width=True)

            # Análisis por mes
            st.markdown("### 📅 Patrones Estacionales")

            df_mensual = df_fecha.groupby('Mes')[variable_temporal].mean().reset_index()

            fig_mensual = px.bar(
                df_mensual,
                x='Mes',
                y=variable_temporal,
                title=f"📊 Promedio Mensual de {variable_temporal}",
                color_discrete_sequence=['#FF9F40']
            )

            st.plotly_chart(fig_mensual, use_container_width=True)

            # Calendario de calor
            st.markdown("### 🗓️ Calendario de Actividad")

            # Contar registros por día
            df_fecha['Fecha_Solo'] = df_fecha[COL_FECHA_EVALUACION].dt.date
            conteo_fechas = df_fecha['Fecha_Solo'].value_counts().reset_index()
            conteo_fechas.columns = ['Fecha', 'Registros']

            fig_calendar = px.scatter(
                conteo_fechas,
                x='Fecha',
                y=[1] * len(conteo_fechas),
                size='Registros',
                title="🗓️ Distribución de Registros por Fecha",
                labels={'y': ''},
                color='Registros',
                color_continuous_scale='Viridis'
            )

            fig_calendar.update_layout(
                height=200,
                yaxis=dict(showticklabels=False, showgrid=False),
                showlegend=False
            )

            st.plotly_chart(fig_calendar, use_container_width=True)

        else:
            st.warning("⚠️ No hay fechas válidas en los datos para análisis temporal.")
    else:
        st.info("ℹ️ No se encontró columna de fecha para análisis temporal.")

# Resumen y exportación
st.markdown("---")
st.markdown("## 📥 Exportar Análisis")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Generar Reporte PDF", type="primary"):
        st.info("🚧 Función de reporte PDF en desarrollo.")

with col2:
    # Exportar datos filtrados
    csv_filtered = df.to_csv(index=False)
    st.download_button(
        label="📥 Descargar Datos Filtrados CSV",
        data=csv_filtered,
        file_name="analisis_exploratorio_datos.csv",
        mime="text/csv"
    )

# Tips de interpretación
with st.expander("💡 Tips de Interpretación"):
    st.markdown("""
    ### 🎯 Cómo Interpretar los Análisis:

    **Correlaciones:**
    - |r| > 0.7: Correlación fuerte
    - 0.3 < |r| < 0.7: Correlación moderada
    - |r| < 0.3: Correlación débil

    **PCA:**
    - PC1 y PC2 capturan la mayor varianza
    - Variables con contribuciones altas son más importantes
    - Puntos cercanos son similares

    **t-SNE:**
    - Preserva distancias locales mejor que PCA
    - Útil para visualizar clusters naturales
    - Perplexity afecta el balance local/global

    **Clustering Jerárquico:**
    - Dendrograma muestra jerarquía de agrupamiento
    - Cortes horizontales definen número de clusters
    - Altura indica distancia entre grupos
    """)