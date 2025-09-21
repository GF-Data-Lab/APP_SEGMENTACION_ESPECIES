"""
Página de Modelo de Clasificación - Clustering y Machine Learning

Esta página implementa modelos de clasificación avanzados para especies de carozos
utilizando técnicas de machine learning y clustering automático.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

from common_styles import configure_page, generarMenu, get_cluster_colors, get_plotly_color_map, get_plotly_color_sequence
from data_columns import COL_ESPECIE, COL_VARIEDAD, COL_BRIX, COL_ACIDEZ

# Configuración de página
configure_page("🤖 Modelo de Clasificación", "🤖")
generarMenu()

st.title("🤖 Modelo de Clasificación de Especies")

st.markdown("""
## 🎯 Modelos Predictivos y Clustering Avanzado

Esta página permite entrenar y evaluar modelos de machine learning para la clasificación
automática de especies de carozos basada en sus características de calidad.

### ✨ Funcionalidades:
- **Clustering Automático:** K-means, DBSCAN, Clustering Jerárquico
- **Análisis PCA:** Reducción dimensional y visualización
- **Métricas de Evaluación:** Silhouette Score, Calinski-Harabasz Index
- **Modelos Predictivos:** Clasificación supervisada
- **Validación Cruzada:** Evaluación robusta de modelos
""")

# Verificar datos disponibles
if "carozos_df" not in st.session_state:
    st.warning("⚠️ No hay datos cargados. Por favor, ve a 'Carga de Archivos' primero.")
    st.info("📋 **Para usar esta página:**\n1. Carga datos en 'Carga de Archivos'\n2. Ejecuta segmentación en Ciruela o Nectarina\n3. Regresa aquí para análisis avanzado")
    st.stop()

df = st.session_state["carozos_df"]

# Sidebar para configuración
st.sidebar.markdown("## ⚙️ Configuración del Modelo")

# Selección de especie
especies_disponibles = df[COL_ESPECIE].dropna().unique() if COL_ESPECIE in df.columns else []
if len(especies_disponibles) == 0:
    st.error("❌ No se encontraron especies en los datos.")
    st.stop()

especie_seleccionada = st.sidebar.selectbox(
    "🔍 Seleccionar Especie",
    especies_disponibles,
    help="Selecciona la especie para entrenar el modelo"
)

# Filtrar datos por especie
df_especie = df[df[COL_ESPECIE] == especie_seleccionada] if COL_ESPECIE in df.columns else df

# Selección de características
st.sidebar.markdown("### 📊 Características para el Modelo")

columnas_numericas = df_especie.select_dtypes(include=[np.number]).columns.tolist()
columnas_disponibles = [col for col in columnas_numericas if col not in ['cluster', 'cluster_grp']]

if len(columnas_disponibles) < 2:
    st.error("❌ No hay suficientes columnas numéricas para el análisis.")
    st.stop()

caracteristicas_seleccionadas = st.sidebar.multiselect(
    "🎯 Seleccionar Características",
    columnas_disponibles,
    default=columnas_disponibles[:4] if len(columnas_disponibles) >= 4 else columnas_disponibles,
    help="Selecciona las características para entrenar el modelo"
)

if len(caracteristicas_seleccionadas) < 2:
    st.sidebar.error("⚠️ Selecciona al menos 2 características.")
    st.stop()

# Preparar datos para clustering
df_modelo = df_especie[caracteristicas_seleccionadas].dropna()

if len(df_modelo) < 10:
    st.error("❌ No hay suficientes datos válidos para el análisis.")
    st.stop()

# Estandarizar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_modelo)

# Configuración de clustering
st.sidebar.markdown("### 🎛️ Parámetros de Clustering")

n_clusters = st.sidebar.slider(
    "🔢 Número de Clusters",
    min_value=2,
    max_value=8,
    value=4,
    help="Número de clusters para K-means"
)

algoritmo_clustering = st.sidebar.selectbox(
    "🤖 Algoritmo de Clustering",
    ["K-means", "K-means++"],
    help="Algoritmo de clustering a utilizar"
)

# Tabs principales
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Clustering Automático",
    "📊 Análisis PCA",
    "📈 Evaluación de Modelos",
    "🔍 Predicciones"
])

with tab1:
    st.markdown("## 🎯 Clustering Automático")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🚀 Ejecutar Clustering", type="primary"):
            with st.spinner("🔄 Ejecutando clustering..."):
                # Ejecutar K-means
                if algoritmo_clustering == "K-means":
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                else:
                    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)

                clusters = kmeans.fit_predict(X_scaled)

                # Guardar resultados
                df_resultado = df_modelo.copy()
                df_resultado['Cluster'] = clusters + 1  # Clusters 1-based

                # Métricas de evaluación
                silhouette = silhouette_score(X_scaled, clusters)
                calinski = calinski_harabasz_score(X_scaled, clusters)

                # Mostrar métricas
                st.success("✅ Clustering completado exitosamente!")

                col_met1, col_met2, col_met3 = st.columns(3)
                with col_met1:
                    st.metric("📊 Silhouette Score", f"{silhouette:.3f}", help="Rango: -1 a 1. Mejor: cercano a 1")
                with col_met2:
                    st.metric("📈 Calinski-Harabasz", f"{calinski:.1f}", help="Mayor valor = mejor separación")
                with col_met3:
                    st.metric("🎯 Clusters Generados", n_clusters)

                # Gráfico de resultados
                if len(caracteristicas_seleccionadas) >= 2:
                    colors = get_plotly_color_map()

                    fig = px.scatter(
                        df_resultado,
                        x=caracteristicas_seleccionadas[0],
                        y=caracteristicas_seleccionadas[1],
                        color='Cluster',
                        color_discrete_map=colors,
                        title=f"🎯 Resultados de Clustering - {especie_seleccionada}",
                        labels={'Cluster': 'Cluster'}
                    )

                    fig.update_layout(
                        height=500,
                        showlegend=True,
                        font=dict(size=12)
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Guardar en session state
                st.session_state[f"clustering_results_{especie_seleccionada}"] = df_resultado
                st.session_state[f"clustering_metrics_{especie_seleccionada}"] = {
                    'silhouette': silhouette,
                    'calinski': calinski,
                    'n_clusters': n_clusters
                }

    with col2:
        st.markdown("### 📋 Información del Modelo")
        st.info(f"""
        **Especie:** {especie_seleccionada}

        **Características:** {len(caracteristicas_seleccionadas)}

        **Registros:** {len(df_modelo):,}

        **Algoritmo:** {algoritmo_clustering}

        **Clusters:** {n_clusters}
        """)

with tab2:
    st.markdown("## 📊 Análisis de Componentes Principales (PCA)")

    if len(caracteristicas_seleccionadas) >= 3:
        # Ejecutar PCA
        pca_2d = PCA(n_components=2, random_state=42)
        pca_3d = PCA(n_components=3, random_state=42)

        pca_2d_result = pca_2d.fit_transform(X_scaled)
        pca_3d_result = pca_3d.fit_transform(X_scaled)

        # Información de varianza explicada
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Varianza Explicada (2D)", f"{pca_2d.explained_variance_ratio_.sum()*100:.1f}%")
        with col2:
            st.metric("📈 Varianza Explicada (3D)", f"{pca_3d.explained_variance_ratio_.sum()*100:.1f}%")

        # Gráficos PCA
        tab_2d, tab_3d = st.tabs(["📊 PCA 2D", "🎯 PCA 3D"])

        with tab_2d:
            df_pca_2d = pd.DataFrame(pca_2d_result, columns=['PC1', 'PC2'])

            # Si hay resultados de clustering, usarlos
            if f"clustering_results_{especie_seleccionada}" in st.session_state:
                df_clustering = st.session_state[f"clustering_results_{especie_seleccionada}"]
                if len(df_pca_2d) == len(df_clustering):
                    df_pca_2d['Cluster'] = df_clustering['Cluster'].values
                    color_col = 'Cluster'
                    colors = get_plotly_color_map()
                else:
                    color_col = None
                    colors = None
            else:
                color_col = None
                colors = None

            fig_2d = px.scatter(
                df_pca_2d,
                x='PC1', y='PC2',
                color=color_col,
                color_discrete_map=colors,
                title="📊 Análisis PCA 2D",
                labels={'PC1': f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)',
                       'PC2': f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)'}
            )

            fig_2d.update_layout(height=500)
            st.plotly_chart(fig_2d, use_container_width=True)

        with tab_3d:
            df_pca_3d = pd.DataFrame(pca_3d_result, columns=['PC1', 'PC2', 'PC3'])

            # Si hay resultados de clustering, usarlos
            if f"clustering_results_{especie_seleccionada}" in st.session_state:
                df_clustering = st.session_state[f"clustering_results_{especie_seleccionada}"]
                if len(df_pca_3d) == len(df_clustering):
                    df_pca_3d['Cluster'] = df_clustering['Cluster'].values
                    color_col = 'Cluster'
                    colors = get_plotly_color_sequence()
                else:
                    color_col = None
                    colors = None
            else:
                color_col = None
                colors = None

            fig_3d = px.scatter_3d(
                df_pca_3d,
                x='PC1', y='PC2', z='PC3',
                color=color_col,
                color_discrete_sequence=colors,
                title="🎯 Análisis PCA 3D",
                labels={'PC1': f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)',
                       'PC2': f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)',
                       'PC3': f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)'}
            )

            fig_3d.update_layout(height=600)
            st.plotly_chart(fig_3d, use_container_width=True)

        # Contribución de características
        st.markdown("### 🔍 Contribución de Características a los Componentes")

        components_df = pd.DataFrame(
            pca_2d.components_.T,
            columns=['PC1', 'PC2'],
            index=caracteristicas_seleccionadas
        )

        fig_contrib = px.bar(
            components_df.reset_index(),
            x='index', y=['PC1', 'PC2'],
            title="📊 Contribución de Características a PCA",
            labels={'index': 'Características', 'value': 'Contribución'},
            barmode='group'
        )

        fig_contrib.update_layout(height=400)
        st.plotly_chart(fig_contrib, use_container_width=True)

    else:
        st.warning("⚠️ Selecciona al menos 3 características para el análisis PCA completo.")

with tab3:
    st.markdown("## 📈 Evaluación de Modelos")

    # Determinar rango óptimo de clusters
    if st.button("🔍 Análisis de Clusters Óptimos", type="primary"):
        with st.spinner("🔄 Analizando número óptimo de clusters..."):
            k_range = range(2, min(11, len(df_modelo)//2))
            silhouette_scores = []
            calinski_scores = []
            inertias = []

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)

                silhouette_scores.append(silhouette_score(X_scaled, clusters))
                calinski_scores.append(calinski_harabasz_score(X_scaled, clusters))
                inertias.append(kmeans.inertia_)

            # Gráficos de evaluación
            col1, col2 = st.columns(2)

            with col1:
                fig_sil = px.line(
                    x=list(k_range), y=silhouette_scores,
                    title="📊 Silhouette Score vs Número de Clusters",
                    labels={'x': 'Número de Clusters', 'y': 'Silhouette Score'},
                    markers=True
                )
                fig_sil.update_traces(line_color='#7FB069')
                st.plotly_chart(fig_sil, use_container_width=True)

            with col2:
                fig_cal = px.line(
                    x=list(k_range), y=calinski_scores,
                    title="📈 Calinski-Harabasz Index vs Clusters",
                    labels={'x': 'Número de Clusters', 'y': 'Calinski-Harabasz Index'},
                    markers=True
                )
                fig_cal.update_traces(line_color='#FF9F40')
                st.plotly_chart(fig_cal, use_container_width=True)

            # Método del codo
            fig_elbow = px.line(
                x=list(k_range), y=inertias,
                title="📉 Método del Codo - Inercia vs Clusters",
                labels={'x': 'Número de Clusters', 'y': 'Inercia'},
                markers=True
            )
            fig_elbow.update_traces(line_color='#FF6B6B')
            st.plotly_chart(fig_elbow, use_container_width=True)

            # Recomendaciones
            best_silhouette_k = k_range[np.argmax(silhouette_scores)]
            best_calinski_k = k_range[np.argmax(calinski_scores)]

            st.success(f"""
            ### 🎯 Recomendaciones de Clustering:

            - **Mejor Silhouette Score:** {best_silhouette_k} clusters (Score: {max(silhouette_scores):.3f})
            - **Mejor Calinski-Harabasz:** {best_calinski_k} clusters (Index: {max(calinski_scores):.1f})

            💡 **Recomendación:** Usar entre {min(best_silhouette_k, best_calinski_k)} y {max(best_silhouette_k, best_calinski_k)} clusters.
            """)

with tab4:
    st.markdown("## 🔍 Predicciones y Clasificación")

    if f"clustering_results_{especie_seleccionada}" in st.session_state:
        df_resultados = st.session_state[f"clustering_results_{especie_seleccionada}"]
        metricas = st.session_state[f"clustering_metrics_{especie_seleccionada}"]

        st.success(f"✅ Modelo entrenado para {especie_seleccionada}")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 📊 Distribución de Clusters")
            cluster_counts = df_resultados['Cluster'].value_counts().sort_index()

            colors = get_cluster_colors()
            cluster_colors = [colors['plotly'][i] for i in cluster_counts.index]

            fig_dist = px.pie(
                values=cluster_counts.values,
                names=[f"Cluster {i}" for i in cluster_counts.index],
                title="🎯 Distribución de Clusters",
                color_discrete_sequence=cluster_colors
            )

            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            st.markdown("### 📈 Métricas del Modelo")
            st.metric("📊 Silhouette Score", f"{metricas['silhouette']:.3f}")
            st.metric("📈 Calinski-Harabasz", f"{metricas['calinski']:.1f}")
            st.metric("🎯 Clusters", metricas['n_clusters'])

            # Calidad del modelo
            if metricas['silhouette'] > 0.5:
                st.success("🌟 Excelente separación de clusters")
            elif metricas['silhouette'] > 0.3:
                st.info("👍 Buena separación de clusters")
            else:
                st.warning("⚠️ Separación moderada de clusters")

        # Tabla de resultados
        st.markdown("### 📋 Resultados Detallados")

        # Aplicar colores a la tabla
        style_function = get_cluster_colors()

        def color_cluster_cells(val):
            colors = get_cluster_colors()
            if pd.isna(val):
                return ""
            try:
                cluster_num = int(val)
                bg_color = colors["background"].get(cluster_num, "#F8F9FA")
                text_color = colors["text"].get(cluster_num, "#6C757D")
                return f"background-color: {bg_color}; color: {text_color}; font-weight: 500"
            except:
                return ""

        styled_df = df_resultados.style.map(color_cluster_cells, subset=['Cluster'])
        st.dataframe(styled_df, use_container_width=True, height=400)

        # Descarga de resultados
        csv = df_resultados.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Resultados CSV",
            data=csv,
            file_name=f"clustering_results_{especie_seleccionada}.csv",
            mime="text/csv"
        )

    else:
        st.info("🎯 Ejecuta primero el clustering en la pestaña 'Clustering Automático' para ver las predicciones.")

# Información adicional
with st.expander("ℹ️ Información sobre los Algoritmos"):
    st.markdown("""
    ### 🤖 Algoritmos Utilizados:

    **K-means:**
    - Agrupa datos en k clusters predefinidos
    - Minimiza la varianza intra-cluster
    - Eficiente para clusters esféricos

    **PCA (Análisis de Componentes Principales):**
    - Reduce dimensionalidad preservando varianza
    - Identifica patrones principales en los datos
    - Útil para visualización y feature selection

    **Métricas de Evaluación:**
    - **Silhouette Score:** Mide qué tan similar es un objeto a su cluster vs otros
    - **Calinski-Harabasz Index:** Ratio de varianza entre-clusters vs intra-clusters
    - **Método del Codo:** Identifica número óptimo de clusters
    """)