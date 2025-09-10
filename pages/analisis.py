# pages/04_Analisis.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
from utils import show_logo
import io

# —————— Configuración de página “wide” ——————
st.set_page_config(
    page_title="🔍 Análisis Exploratorio y Clustering",
    layout="wide",
)

# --------------------------------------------------
# Sidebar styling + menu
# --------------------------------------------------
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
generarMenu()

st.title("🔍 Análisis Exploratorio y Clustering de Carozos")

# —————— Cargar datos ——————
if "carozos_df" not in st.session_state:
    st.warning("Carga primero un archivo en 'Carga de archivos'.")
    st.stop()
df_raw = st.session_state["carozos_df"].copy()

# Columnas esperadas para permitir clasificación y detección de outliers
expected_cols = [
    "Especie", "Variedad", "Quilla", "Hombro", "Mejilla 1", "Mejilla 2",
    "BRIX", "Acidez (%)", "Punta", "Peso (g)", "cond_sum_grp"
]
missing = [c for c in expected_cols if c not in df_raw.columns]
if missing:
    st.warning(
        "No se encontraron las columnas: " + ", ".join(missing) +
        ". Se crearán valores nulos para continuar."
    )
    for col in missing:
        df_raw[col] = np.nan

numeric_cols_all = df_raw.select_dtypes(include=np.number).columns.tolist()
mask_outliers = pd.Series(False, index=df_raw.index)
for col in numeric_cols_all:
    q1 = df_raw[col].quantile(0.25)
    q3 = df_raw[col].quantile(0.75)
    iqr = q3 - q1
    mask_outliers |= (df_raw[col] < q1 - 1.5 * iqr) | (df_raw[col] > q3 + 1.5 * iqr)

# —————— Definir rangos y etiquetas ——————
# grp_cod_sum entre 1–4 ➔ Top 1; 5–8 ➔ Top 2; 9–12 ➔ Top 3; 13–16 ➔ Top 4
bins  = [0, 4,  8,   12,  16]                  # límites (0 para incluir 1)
labels = ["Top 1", "Top 2", "Top 3", "Top 4"]   # etiquetas

df_raw["cond_sum_grp"] = pd.to_numeric(df_raw["cond_sum_grp"], errors="coerce")
df_raw["rankid"] = pd.cut(
    df_raw["cond_sum_grp"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

df_no_outliers = df_raw[~mask_outliers].copy()
exclude_out = st.checkbox("Excluir registros atípicos (IQR)", value=False)
df = df_no_outliers if exclude_out else df_raw.copy()
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# Ahora sí son 7 pestañas
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Descriptivas",
    "🔗 Correlaciones",
    "📦 Boxplots",
    "📐 PCA",
    "🔬 Clustering",
    "📥 Exportar",
    "🌱 Subtipos"       # <- ¡Este es el séptimo!
])

# --- 1. Estadísticas descriptivas + por especie ---
with tab1:
    st.subheader("🔍 Vista detalle de cada grupo")

    # 1) Define las opciones de filtro
    opciones_rank = ["Top 1", "Top 2", "Top 3", "Top 4"]
    seleccion = st.multiselect(
        "Selecciona qué Top(s) mostrar:",
        options=opciones_rank,
        default=opciones_rank
    )

    # 2) Filtra el DataFrame según la selección y ordena las columnas
    columnas_detalle = [
        "Especie", "Variedad", "Quilla", "Hombro", "Mejilla 1", "Mejilla 2",
        "BRIX", "Acidez (%)", "Punta", "Peso (g)",
        "cond_sum_grp", "rankid"
    ]
    df_filtrado = df[df["rankid"].isin(seleccion)][columnas_detalle]

    # 4) Divide el layout en 2 columnas
    col1, col2 = st.columns(2)

    # 5) En la primera columna, solo Ciruela
    with col1:
        st.markdown("**Ciruela**")
        df_ciru = df_filtrado[df_filtrado["Especie"] == "Ciruela"]
        st.dataframe(df_ciru, use_container_width=True)

    # 6) En la segunda columna, solo Nectarina
    with col2:
        st.markdown("**Nectarina**")
        df_nec = df_filtrado[df_filtrado["Especie"] == "Nectarin"]
        st.dataframe(df_nec, use_container_width=True)

    # 7) Renderiza la tabla filtrada con esas columnas
    st.dataframe(df_filtrado)
    st.subheader("1. Estadísticas Descriptivas")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.markdown("#### 1.b. Estadísticas Numéricas por Especie")
    st.dataframe(
        df.groupby("Especie")[numeric_cols]
          .describe()
          .T,
        use_container_width=True
    )

    st.markdown("#### 1.c. Boxplots comparativos por Especie")
    for col in numeric_cols:
        fig = px.box(
            df, x="Especie", y=col, points="outliers",
            title=f"{col} por Especie"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 1.d. Registros atípicos (IQR)")
    outlier_frames = []
    for col in numeric_cols_all:
        q1 = df_raw[col].quantile(0.25)
        q3 = df_raw[col].quantile(0.75)
        iqr = q3 - q1
        mask = (df_raw[col] < q1 - 1.5 * iqr) | (df_raw[col] > q3 + 1.5 * iqr)
        if mask.any():
            tmp = df_raw.loc[mask, ["Especie", "Variedad", "rankid", col]].copy()
            tmp["Variable"] = col
            outlier_frames.append(tmp)
    if outlier_frames:
        df_out = pd.concat(outlier_frames, ignore_index=True)
        st.dataframe(df_out, use_container_width=True)
    else:
        st.write("No se detectaron outliers.")

# --- 2. Matriz de correlaciones ---
with tab2:
    st.subheader("2. Heatmap de Correlaciones")
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr, text_auto=True, aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Matriz de Correlaciones"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- 3. Boxplots univariados ---
with tab3:
    st.subheader("3. Boxplots Univariados (optimizado)")
    cols = st.multiselect(
        "Selecciona hasta 4 variables",
        options=numeric_cols,
        default=numeric_cols[:4],
        max_selections=4
    )
    for row in [cols[i:i+2] for i in range(0, len(cols), 2)]:
        c1, c2 = st.columns(2)
        for name, holder in zip(row, (c1, c2)):
            fig = px.box(df, y=name, points="outliers", title=f"Boxplot de {name}")
            holder.plotly_chart(fig, use_container_width=True)

# --- Helpers cacheados ---
@st.cache_data
def run_pca(data, n):
    pca = PCA(n_components=n)
    pcs = pca.fit_transform(data)
    return pcs, pca.explained_variance_ratio_

@st.cache_data
def run_km(pc, k):
    km = KMeans(n_clusters=k, random_state=0)
    labels = km.fit_predict(pc)
    # scikit-learn's silhouette_score requires at least 2 distinct labels.
    # In some datasets KMeans may return a single label (e.g. identical samples),
    # which would normally raise a ValueError.  We guard against that case and
    # return ``None`` for the silhouette instead of crashing the app.
    sil = None
    if len(set(labels)) > 1:
        sil = silhouette_score(pc, labels)
    return labels, km.inertia_, sil

# --- 4. PCA ---
with tab4:
    st.subheader("4. Análisis de Componentes Principales (PCA)")
    
    # Selector de tipo de análisis PCA
    pca_type = st.radio(
        "Tipo de análisis PCA:", 
        ["Registros individuales", "Datos agregados por variedad"], 
        index=1
    )
    
    if pca_type == "Registros individuales":
        # PCA original sobre datos individuales
        n_comp = st.slider("Componentes PCA", 2, 4, 2)
        pc, var_ratio = run_pca(df[numeric_cols].fillna(0), n_comp)
        pca_df = pd.DataFrame(pc, columns=[f"PC{i+1}" for i in range(n_comp)])

        pct = st.slider("% de puntos a mostrar", 5, 100, 25)
        idx = pca_df.sample(frac=pct/100, random_state=0).index

        fig = px.scatter(
            pca_df.loc[idx],
            x="PC1",
            y=("PC2" if n_comp >= 2 else None),
            title=f"PCA Scatter ({pct}% puntos individuales)",
            render_mode="webgl"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write("Varianza explicada:",
                 {f"PC{i+1}": round(v, 3) for i, v in enumerate(var_ratio)})
    
    else:
        # PCA sobre datos agregados por variedad
        st.markdown("#### PCA de datos agregados por variedad")
        
        try:
            # Verificar si tenemos datos agregados de segmentación
            agg_data = None
            if "agg_groups_plum" in st.session_state:
                agg_data = st.session_state["agg_groups_plum"].copy()
                species_name = "Ciruela"
            elif "agg_groups_nect" in st.session_state:
                agg_data = st.session_state["agg_groups_nect"].copy()  
                species_name = "Nectarina"
            
            if agg_data is not None and len(agg_data) > 0:
                # Usar las mismas features que en segmentacion_base.py
                pca_features = [
                    "promedio_cond_sum",
                    "promedio_brix", 
                    "promedio_acidez",
                    "promedio_firmeza_punto",
                    "promedio_mejillas",
                ]
                
                # Filtrar columnas que existen en los datos
                available_features = [col for col in pca_features if col in agg_data.columns]
                
                if len(available_features) >= 2:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.decomposition import PCA
                    
                    # Preparar datos para PCA
                    df_features = agg_data[available_features].fillna(0)
                    
                    # Normalizar
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(df_features)
                    
                    # Aplicar PCA
                    pca = PCA(n_components=2)
                    pcs = pca.fit_transform(X_scaled)
                    
                    # Crear DataFrame con componentes principales
                    agg_data_pca = agg_data.copy()
                    agg_data_pca["PC1"] = pcs[:, 0]
                    agg_data_pca["PC2"] = pcs[:, 1]
                    
                    # Crear gráfico PCA interactivo
                    cluster_col = "cluster_grp" if "cluster_grp" in agg_data_pca.columns else None
                    
                    if cluster_col:
                        # Gráfico coloreado por cluster
                        fig = px.scatter(
                            agg_data_pca,
                            x="PC1",
                            y="PC2", 
                            color=cluster_col,
                            hover_data={
                                "Variedad": True,
                                "promedio_brix": ":.2f" if "promedio_brix" in agg_data_pca.columns else False,
                                "promedio_acidez": ":.2f" if "promedio_acidez" in agg_data_pca.columns else False,
                                "cluster_grp": True if cluster_col else False
                            },
                            title=f"PCA - {species_name} (agregado por variedad)",
                            color_continuous_scale="viridis"
                        )
                    else:
                        # Gráfico sin colorear por cluster
                        fig = px.scatter(
                            agg_data_pca,
                            x="PC1", 
                            y="PC2",
                            hover_data={
                                "Variedad": True,
                                "promedio_brix": ":.2f" if "promedio_brix" in agg_data_pca.columns else False,
                                "promedio_acidez": ":.2f" if "promedio_acidez" in agg_data_pca.columns else False
                            },
                            title=f"PCA - {species_name} (agregado por variedad)"
                        )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar varianza explicada
                    var_explained = pca.explained_variance_ratio_
                    st.write(f"**Varianza explicada:** PC1: {var_explained[0]:.3f}, PC2: {var_explained[1]:.3f}")
                    st.write(f"**Total varianza explicada:** {sum(var_explained):.3f}")
                    
                    # Mostrar información de features utilizadas
                    st.write(f"**Features utilizadas:** {', '.join(available_features)}")
                    
                    # Tabla de datos agregados
                    st.markdown("#### Datos agregados utilizados en PCA")
                    display_cols = ["Variedad"] + available_features
                    if cluster_col:
                        display_cols.append(cluster_col)
                    st.dataframe(agg_data_pca[display_cols], use_container_width=True)
                    
                else:
                    st.warning(f"Se necesitan al menos 2 features numéricas. Disponibles: {available_features}")
            else:
                st.info("Para ver el PCA agregado por variedad, ejecuta primero la segmentación en las páginas de Ciruela o Nectarina.")
                
        except Exception as e:
            st.error(f"Error al generar PCA agregado: {str(e)}")
            
            # Fallback al PCA individual si falla el agregado
            st.markdown("Mostrando PCA de registros individuales como alternativa:")
            n_comp = st.slider("Componentes PCA", 2, 4, 2, key="fallback_pca")
            pc, var_ratio = run_pca(df[numeric_cols].fillna(0), n_comp)
            pca_df = pd.DataFrame(pc, columns=[f"PC{i+1}" for i in range(n_comp)])
            
            fig = px.scatter(
                pca_df,
                x="PC1",
                y="PC2" if n_comp >= 2 else None,
                title="PCA Scatter (fallback - registros individuales)"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.write("Varianza explicada:", 
                     {f"PC{i+1}": round(v, 3) for i, v in enumerate(var_ratio)})

# --- 5. Clustering interactivo por modo ---
with tab5:
    st.subheader("5. K-Means Clustering")
    
    # Selector de tipo de clustering
    cluster_type = st.radio(
        "Tipo de clustering:", 
        ["Registros individuales", "Datos agregados por variedad"], 
        index=0
    )
    
    if cluster_type == "Registros individuales":
        # Clustering original sobre datos individuales
        # sliders para k y porcentaje
        best_k = st.slider("k definitivo", 2, 10, 4, key="best_k")
        pct_cluster = st.slider("% puntos scatter", 5, 100, 25, key="pct_cluster")
        
        # Calcular PCA para clustering (necesario para el análisis)
        n_comp_cluster = 2  # Componentes fijos para clustering
        pc, var_ratio = run_pca(df[numeric_cols].fillna(0), n_comp_cluster)
        pca_df = pd.DataFrame(pc, columns=[f"PC{i+1}" for i in range(n_comp_cluster)])

        # Recalcular genérico
        labels_gen, inertia_gen, sil_gen = run_km(pc, best_k)
        labels_gen_s = pd.Series(labels_gen, index=pca_df.index)
        idx_gen = pca_df.sample(frac=pct_cluster/100, random_state=0).index

        # Recalcular Ciruela
        df_plum = df[df["Especie"] == "Ciruela"]
        pc_plum, _ = run_pca(df_plum[numeric_cols].fillna(0), n_comp_cluster)
        pca_df_plum = pd.DataFrame(pc_plum,
                                   columns=[f"PC{i+1}" for i in range(n_comp_cluster)],
                                   index=df_plum.index)
        labels_plum, inertia_plum, sil_plum = run_km(pc_plum, best_k)
        labels_plum_s = pd.Series(labels_plum, index=pca_df_plum.index)
        idx_plum = pca_df_plum.sample(frac=pct_cluster/100, random_state=0).index

        # Recalcular Nectarin
        df_nec = df[df["Especie"] == "Nectarin"]
        pc_nec, _ = run_pca(df_nec[numeric_cols].fillna(0), n_comp_cluster)
        pca_df_nec = pd.DataFrame(pc_nec,
                                  columns=[f"PC{i+1}" for i in range(n_comp_cluster)],
                                  index=df_nec.index)
        labels_nec, inertia_nec, sil_nec = run_km(pc_nec, best_k)
        labels_nec_s = pd.Series(labels_nec, index=pca_df_nec.index)
        idx_nec = pca_df_nec.sample(frac=pct_cluster/100, random_state=0).index

        # selector de modo
        modo = st.radio("Mostrar clustering para:", ["Genérico", "Ciruela", "Nectarin"], key="modo_individual")
        if modo == "Genérico":
            df_plot, labels_plot_s, idx_plot, title, sil, inertia, df_orig = (
                pca_df, labels_gen_s, idx_gen, f"Genérico k={best_k}", sil_gen, inertia_gen, df
            )
        elif modo == "Ciruela":
            df_plot, labels_plot_s, idx_plot, title, sil, inertia, df_orig = (
                pca_df_plum, labels_plum_s, idx_plum, f"Ciruela k={best_k}", sil_plum, inertia_plum, df_plum
            )
        else:
            df_plot, labels_plot_s, idx_plot, title, sil, inertia, df_orig = (
                pca_df_nec, labels_nec_s, idx_nec, f"Nectarin k={best_k}", sil_nec, inertia_nec, df_nec
            )

        fig = px.scatter(
            df_plot.loc[idx_plot],
            x="PC1",
            y=("PC2" if n_comp >= 2 else None),
            color=labels_plot_s.loc[idx_plot].astype(str),
            title=title,
            render_mode="webgl"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write("Tamaños de cluster:", labels_plot_s.value_counts().to_dict())
        st.write("Inercia:", round(inertia, 3))
        if sil is not None:
            st.write("Silhouette Score:", round(sil, 3))
        else:
            st.write("Silhouette Score: no disponible (1 solo cluster)")
        summary = df_orig.assign(cluster=labels_plot_s).groupby("cluster")[numeric_cols].mean()
        st.markdown("Promedios por cluster:")
        st.dataframe(summary, use_container_width=True)
    
    else:
        # Clustering sobre datos agregados por variedad
        st.markdown("#### Clustering de datos agregados por variedad")
        
        try:
            # Verificar si tenemos datos agregados de segmentación
            agg_data = None
            if "agg_groups_plum" in st.session_state:
                agg_data = st.session_state["agg_groups_plum"].copy()
                species_name = "Ciruela"
            elif "agg_groups_nect" in st.session_state:
                agg_data = st.session_state["agg_groups_nect"].copy()
                species_name = "Nectarina"
                
            if agg_data is not None and len(agg_data) > 0:
                # Features para clustering
                cluster_features = [
                    "promedio_cond_sum",
                    "promedio_brix", 
                    "promedio_acidez",
                    "promedio_firmeza_punto",
                    "promedio_mejillas",
                ]
                
                available_features = [col for col in cluster_features if col in agg_data.columns]
                
                if len(available_features) >= 2:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.cluster import KMeans
                    from sklearn.metrics import silhouette_score
                    
                    # Preparar datos
                    df_features = agg_data[available_features].fillna(0)
                    
                    # Normalizar
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(df_features)
                    
                    # Parámetros de clustering
                    k_agg = st.slider("Número de clusters para variedades", 2, min(8, len(agg_data)), 4, key="k_agg")
                    
                    # Aplicar K-means
                    kmeans = KMeans(n_clusters=k_agg, random_state=42)
                    cluster_labels = kmeans.fit_predict(X_scaled)
                    
                    # Calcular métricas
                    if len(set(cluster_labels)) > 1:
                        sil_score = silhouette_score(X_scaled, cluster_labels)
                    else:
                        sil_score = None
                        
                    inertia = kmeans.inertia_
                    
                    # Agregar clusters a los datos
                    agg_data_clustered = agg_data.copy()
                    agg_data_clustered["cluster_auto"] = cluster_labels
                    
                    # PCA para visualización
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    pcs = pca.fit_transform(X_scaled)
                    agg_data_clustered["PC1"] = pcs[:, 0] 
                    agg_data_clustered["PC2"] = pcs[:, 1]
                    
                    # Gráfico de clustering
                    fig = px.scatter(
                        agg_data_clustered,
                        x="PC1",
                        y="PC2",
                        color="cluster_auto",
                        hover_data={
                            "Variedad": True,
                            "promedio_brix": ":.2f" if "promedio_brix" in agg_data_clustered.columns else False,
                            "promedio_acidez": ":.2f" if "promedio_acidez" in agg_data_clustered.columns else False,
                            "cluster_grp": True if "cluster_grp" in agg_data_clustered.columns else False,
                            "cluster_auto": True
                        },
                        title=f"K-Means Clustering - {species_name} (k={k_agg})",
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar métricas
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Inercia", f"{inertia:.3f}")
                    with col2:
                        if sil_score is not None:
                            st.metric("Silhouette Score", f"{sil_score:.3f}")
                        else:
                            st.metric("Silhouette Score", "N/A")
                    
                    # Tamaños de clusters
                    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
                    st.write("**Tamaños de cluster:**", cluster_sizes.to_dict())
                    
                    # Comparación con clusters originales si existen
                    if "cluster_grp" in agg_data_clustered.columns:
                        st.markdown("#### Comparación: Clusters por reglas vs K-Means automático")
                        comparison = agg_data_clustered[["Variedad", "cluster_grp", "cluster_auto"]].copy()
                        st.dataframe(comparison, use_container_width=True)
                        
                        # Matriz de confusión
                        conf_matrix = pd.crosstab(agg_data_clustered["cluster_grp"], 
                                                agg_data_clustered["cluster_auto"], 
                                                margins=True)
                        st.markdown("#### Matriz de confusión (Reglas vs Automático)")
                        st.dataframe(conf_matrix, use_container_width=True)
                    
                    # Resumen estadístico por cluster
                    st.markdown("#### Promedios por cluster automático")
                    cluster_summary = agg_data_clustered.groupby("cluster_auto")[available_features].mean()
                    st.dataframe(cluster_summary, use_container_width=True)
                    
                    # Detalles de variedades por cluster
                    st.markdown("#### Variedades por cluster")
                    for cluster_id in sorted(cluster_labels):
                        varieties_in_cluster = agg_data_clustered[agg_data_clustered["cluster_auto"] == cluster_id]["Variedad"].tolist()
                        st.write(f"**Cluster {cluster_id}:** {', '.join(varieties_in_cluster)}")
                    
                else:
                    st.warning(f"Se necesitan al menos 2 features numéricas. Disponibles: {available_features}")
            else:
                st.info("Para ver clustering agregado por variedad, ejecuta primero la segmentación en las páginas de Ciruela o Nectarina.")
                
        except Exception as e:
            st.error(f"Error al generar clustering agregado: {str(e)}")

# --- 6. Exportar ---
with tab6:
    st.subheader("6. Guardar y Descargar")
    df["pca_1"] = pca_df["PC1"]
    if n_comp >= 2:
        df["pca_2"] = pca_df["PC2"]
    df["cluster_generic"] = labels_gen_s
    df.loc[df["Especie"] == "Ciruela", "cluster_plum"] = labels_plum_s
    df.loc[df["Especie"] == "Nectarin", "cluster_nec"] = labels_nec_s
    st.session_state["df_analizado"] = df

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Analizado")
    buf.seek(0)
    st.download_button(
        "📥 Descargar análisis",
        data=buf,
        file_name="carozos_analisis_por_especie.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
# ——— Contenido del tab7: Subtipos ———
with tab7:
    st.title("🌱 Análisis por Subtipos de Especie")

    # ——— Ciruela: plum_subtype ———
    st.markdown("#### 2.a. Estadísticas Numéricas en Ciruela por Subtipo")
    df_ciru = df[df["Especie"] == "Ciruela"]

    if "plum_subtype" in df_ciru.columns:
        st.dataframe(
            df_ciru.groupby("plum_subtype")[numeric_cols]
                  .describe()
                  .T,
            use_container_width=True,
        )

        st.markdown("#### 2.b. Boxplots en Ciruela por Subtipo")
        for col in numeric_cols:
            fig = px.box(
                df_ciru, x="plum_subtype", y=col, points="outliers",
                title=f"{col} por Subtipo en Ciruela",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No se encontró la columna 'plum_subtype' para Ciruela")

    st.markdown("---")

    # ——— Nectarina: Color de pulpa ———
    st.markdown("#### 2.c. Estadísticas Numéricas en Nectarina por Color de Pulpa")
    df_nec = df[df["Especie"] == "Nectarin"]

    if "Color de pulpa" in df_nec.columns:
        st.dataframe(
            df_nec.groupby("Color de pulpa")[numeric_cols]
                  .describe()
                  .T,
            use_container_width=True,
        )

        st.markdown("#### 2.d. Boxplots en Nectarina por Color de Pulpa")
        for col in numeric_cols:
            fig = px.box(
                df_nec, x="Color de pulpa", y=col, points="outliers",
                title=f"{col} por Color de Pulpa en Nectarina",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No se encontró la columna 'Color de pulpa' para Nectarina")
