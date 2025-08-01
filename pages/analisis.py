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
        if st.button('Página de Inicio 🏚️'):      st.switch_page('app.py')
        if st.button('Segmentación de especies 🍑'): st.switch_page('pages/Segmentacion_especies.py')
        if st.button('Modelo de Clasificación'): st.switch_page('pages/Cluster_especies.py')
        if st.button('Análisis exploratorio'):   st.switch_page('pages/analisis.py')
generarMenu()

st.title("🔍 Análisis Exploratorio y Clustering de Carozos")

# —————— Cargar datos ——————
if "df_seg_especies" not in st.session_state:
    st.warning("Procesa primero en 'Segmentación de especies' antes de ver el análisis.")
    st.stop()
df = st.session_state["df_seg_especies"].copy()
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()



# —————— Definir rangos y etiquetas ——————
# grp_cod_sum entre 1–4 ➔ Top 1; 5–8 ➔ Top 2; 9–12 ➔ Top 3; 13–16 ➔ Top 4
bins  = [0, 4,  8,   12,  16]                  # límites (0 para incluir 1)
labels = ["Top 1", "Top 2", "Top 3", "Top 4"]   # etiquetas

df["rankid"] = pd.cut(
    df["cond_sum_grp"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

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

    # 2) Filtra el DataFrame según la selección
    df_filtrado = df[df["rankid"].isin(seleccion)]

    # 3) Columnas a mostrar
    columnas_detalle = [
        "Especie", "Variedad", "Quilla", "Hombro", "Mejilla 1", "Mejilla 2",
        "BRIX", "Acidez (%)", "Punta", "Peso (g)",
        "cond_sum_grp", "rankid"
    ]

    # 4) Divide el layout en 2 columnas
    col1, col2 = st.columns(2)

    # 5) En la primera columna, solo Ciruela
    with col1:
        st.markdown("**Ciruela**")
        df_ciru = df_filtrado[df_filtrado["Especie"] == "Ciruela"]
        st.dataframe(df_ciru[columnas_detalle], use_container_width=True)

    # 6) En la segunda columna, solo Nectarina
    with col2:
        st.markdown("**Nectarina**")
        df_nec = df_filtrado[df_filtrado["Especie"] == "Nectarin"]
        st.dataframe(df_nec[columnas_detalle], use_container_width=True)

    # 5) Renderiza la tabla filtrada con esas columnas
    st.dataframe(df_filtrado[columnas_detalle])
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
    return labels, km.inertia_, silhouette_score(pc, labels)

# --- 4. PCA ---
with tab4:
    st.subheader("4. Análisis de Componentes Principales (PCA)")
    n_comp = st.slider("Componentes PCA", 2, 4, 2)
    pc, var_ratio = run_pca(df[numeric_cols].fillna(0), n_comp)
    pca_df = pd.DataFrame(pc, columns=[f"PC{i+1}" for i in range(n_comp)])

    pct = st.slider("% de puntos a mostrar", 5, 100, 25)
    idx = pca_df.sample(frac=pct/100, random_state=0).index

    fig = px.scatter(
        pca_df.loc[idx],
        x="PC1",
        y=("PC2" if n_comp >= 2 else None),
        title=f"PCA Scatter ({pct}% puntos)",
        render_mode="webgl"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write("Varianza explicada:",
             {f"PC{i+1}": round(v, 3) for i, v in enumerate(var_ratio)})

# --- 5. Clustering interactivo por modo ---
with tab5:
    st.subheader("5. K-Means Clustering (genérico o por especie)")

    # sliders para k y porcentaje
    best_k = st.slider("k definitivo", 2, 10, 4, key="best_k")
    pct_cluster = st.slider("% puntos scatter", 5, 100, 25, key="pct_cluster")

    # Recalcular genérico
    labels_gen = run_km(pc, best_k)[0]
    labels_gen_s = pd.Series(labels_gen, index=pca_df.index)
    idx_gen = pca_df.sample(frac=pct_cluster/100, random_state=0).index

    # Recalcular Ciruela
    df_plum = df[df["Especie"] == "Ciruela"]
    pc_plum, _ = run_pca(df_plum[numeric_cols].fillna(0), n_comp)
    pca_df_plum = pd.DataFrame(pc_plum,
                               columns=[f"PC{i+1}" for i in range(n_comp)],
                               index=df_plum.index)
    labels_plum = run_km(pc_plum, best_k)[0]
    labels_plum_s = pd.Series(labels_plum, index=pca_df_plum.index)
    idx_plum = pca_df_plum.sample(frac=pct_cluster/100, random_state=0).index

    # Recalcular Nectarin
    df_nec = df[df["Especie"] == "Nectarin"]
    pc_nec, _ = run_pca(df_nec[numeric_cols].fillna(0), n_comp)
    pca_df_nec = pd.DataFrame(pc_nec,
                              columns=[f"PC{i+1}" for i in range(n_comp)],
                              index=df_nec.index)
    labels_nec = run_km(pc_nec, best_k)[0]
    labels_nec_s = pd.Series(labels_nec, index=pca_df_nec.index)
    idx_nec = pca_df_nec.sample(frac=pct_cluster/100, random_state=0).index

    # selector de modo
    modo = st.radio("Mostrar clustering para:", ["Genérico", "Ciruela", "Nectarin"])
    if modo == "Genérico":
        df_plot, labels_plot_s, idx_plot, title = pca_df, labels_gen_s, idx_gen, f"Genérico k={best_k}"
    elif modo == "Ciruela":
        df_plot, labels_plot_s, idx_plot, title = pca_df_plum, labels_plum_s, idx_plum, f"Ciruela k={best_k}"
    else:
        df_plot, labels_plot_s, idx_plot, title = pca_df_nec, labels_nec_s, idx_nec, f"Nectarin k={best_k}"

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
    st.dataframe(
        df_ciru.groupby("plum_subtype")[numeric_cols]
              .describe()
              .T,
        use_container_width=True
    )

    st.markdown("#### 2.b. Boxplots en Ciruela por Subtipo")
    for col in numeric_cols:
        fig = px.box(
            df_ciru, x="plum_subtype", y=col, points="outliers",
            title=f"{col} por Subtipo en Ciruela"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ——— Nectarina: Color de pulpa ———
    st.markdown("#### 2.c. Estadísticas Numéricas en Nectarina por Color de Pulpa")
    df_nec = df[df["Especie"] == "Nectarin"]
    st.dataframe(
        df_nec.groupby("Color de pulpa")[numeric_cols]
              .describe()
              .T,
        use_container_width=True
    )

    st.markdown("#### 2.d. Boxplots en Nectarina por Color de Pulpa")
    for col in numeric_cols:
        fig = px.box(
            df_nec, x="Color de pulpa", y=col, points="outliers",
            title=f"{col} por Color de Pulpa en Nectarina"
        )
        st.plotly_chart(fig, use_container_width=True)
