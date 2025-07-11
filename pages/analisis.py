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

st.set_page_config(
    page_title="🔍 Análisis Exploratorio y Clustering",
    layout="wide",
)

# Sidebar
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
        if st.button('Segmentación de especies 🍑'):
            st.switch_page('pages/Segmentacion_especies.py')
        if st.button('Modelo de Clasificación'):
            st.switch_page('pages/Cluster_especies.py')
        if st.button('Análisis exploratorio'):
            st.switch_page('pages/04_Analisis.py')
generarMenu()
st.title("🔍 Análisis Exploratorio y Clustering")

if "df_seg_especies" not in st.session_state:
    st.warning("Procesa primero en 'Segmentación de especies' antes de ver el análisis.")
    st.stop()

df = st.session_state["df_seg_especies"].copy()
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# Pestañas
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Descriptivas",
    "🔗 Correlaciones",
    "📦 Boxplots",
    "📐 PCA",
    "🔬 Clustering",
    "📥 Exportar"
])

# 1. Estadísticas descriptivas
with tab1:
    st.subheader("1. Estadísticas Descriptivas")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

# 2. Correlaciones
with tab2:
    st.subheader("2. Heatmap de Correlaciones")
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr, text_auto=True, aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Matriz de Correlaciones"
    )
    st.plotly_chart(fig, use_container_width=True)

# 3. Boxplots univariados
with tab3:
    st.subheader("3. Boxplots Univariados (optimizados)")

    cols = st.multiselect(
        "Variables (max 4)", numeric_cols, default=numeric_cols[:4], max_selections=4
    )
    for row in [cols[i:i+2] for i in range(0, len(cols), 2)]:
        c1, c2 = st.columns(2)
        for name, holder in zip(row, (c1, c2)):
            # opción A: solo outliers
            fig = px.box(df, y=name, points="outliers", title=f"Boxplot de {name}")
            # opción B: full boxplot + sample 20%
            # muestra = df[name].dropna().sample(frac=0.2, random_state=0)
            # fig = px.box(y=muestra, points="all", title=f"Boxplot muestreado de {name} (20%)")
            holder.plotly_chart(fig, use_container_width=True)

# Helpers cacheados
@st.cache_data
def run_pca(data, n): 
    p = PCA(n_components=n); pcs = p.fit_transform(data); return pcs, p.explained_variance_ratio_
@st.cache_data
def run_km(pc, k):
    km = KMeans(n_clusters=k, random_state=0)
    l = km.fit_predict(pc)
    return l, km.inertia_, silhouette_score(pc, l)

# 4. PCA
with tab4:
    st.subheader("4. PCA")
    n_comp = st.slider("Componentes", 2, 4, 2)
    pc, var = run_pca(df[numeric_cols].fillna(0), n_comp)
    pca_df = pd.DataFrame(pc, columns=[f"PC{i+1}" for i in range(n_comp)])
    pct = st.slider("% de puntos", 5, 100, 25)
    idx = pca_df.sample(frac=pct/100, random_state=0).index
    fig = px.scatter(
        pca_df.loc[idx], x="PC1", y="PC2" if n_comp>=2 else None,
        title=f"PCA ({pct}% puntos)", render_mode="webgl"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write("Varianza explicada:", {f"PC{i+1}": round(v,3) for i,v in enumerate(var)})

# 5. Clustering
with tab5:
    st.subheader("5. K-Means Clustering")
    max_k = st.slider("k máximo", 2, 10, 6)
    ks = list(range(2, max_k+1))
    inert, sil = [], []
    for k in ks:
        _, i, s = run_km(pc, k)
        inert.append(i); sil.append(s)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.line(x=ks, y=inert, markers=True, title="Elbow Plot"), use_container_width=True)
    c2.plotly_chart(px.line(x=ks, y=sil, markers=True, title="Silhouette vs k"), use_container_width=True)
    bk = st.selectbox("k final", ks, index=0)
    labels, _, _ = run_km(pc, bk)
    fig = px.scatter(
        pca_df.loc[idx], x="PC1", y="PC2" if n_comp>=2 else None,
        color=labels[idx].astype(str), title=f"k={bk} sobre PCA",
        render_mode="webgl"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write("Tamaños de cluster:", pd.Series(labels).value_counts().to_dict())

# 6. Exportar
with tab6:
    st.subheader("6. Guardar y Descargar")
    df["pca_1"], df["kmeans_cluster"] = pca_df["PC1"], labels
    if n_comp>=2: df["pca_2"] = pca_df["PC2"]
    st.session_state["df_analizado"] = df
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w: df.to_excel(w, index=False, sheet_name="Analizado")
    buf.seek(0)
    st.download_button("📥 Descargar datos", data=buf, file_name="analisis.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
