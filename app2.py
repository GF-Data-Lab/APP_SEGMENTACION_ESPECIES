import streamlit as st
import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Optional: KMedoids from sklearn-extra
try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except ImportError:
    HAS_KMEDOIDS = False

############################################################
# ------------------ Rule‑based utilities ---------------- #
############################################################
# NOTE: The thresholds below are **illustrative** and were
#       extracted manually from the decision‑tree diagram.
#       Adjust them to reflect the definitive business rules
#       before deploying the app in production.
############################################################

def _scale_category(value: float, bins):
    """Return category id (1‑4) for *value* given list of
    (lower, upper, category) tuples sorted descending."""
    for lo, hi, label in bins:
        if lo <= value <= hi:
            return label
    return None

# Example rule sets. Add or edit as required.
BRIX_RULES = {
    "candy_plum": [
        (18.0, math.inf, 1),
        (16.0, 17.9, 2),
        (14.0, 15.9, 3),
        (-math.inf, 13.9, 4),
    ],
    "cherry_plum": [
        (21.0, math.inf, 1),
        (18.0, 20.9, 2),
        (15.0, 17.9, 3),
        (-math.inf, 14.9, 4),
    ],
}

FIRMNESS_POINT_DEBIL_RULES = [
    (7.0, math.inf, 1),
    (5.0, 6.9, 2),
    (4.0, 4.9, 3),
    (-math.inf, 3.9, 4),
]

FIRMNESS_MEJILLA_RULES = [
    (9.0, math.inf, 1),
    (7.0, 8.9, 2),
    (5.0, 6.9, 3),
    (-math.inf, 4.9, 4),
]

ACIDITY_RULES = [
    (-math.inf, 0.79, 1),
    (0.80, 0.8, 2),  # placeholder, replace with real thresholds
    (0.81, 0.99, 3),
    (1.0, math.inf, 4),
]

def categorize_row(row, fruit_type="candy_plum"):
    """Return a composite rule‑based cluster label for *row*.
       The label is a string concatenating the sub‑scores.
    """
    brix = row.get("Brix")
    fp  = row.get("Firmeza punto débil (lb)") or row.get("Firmeza punto debil (lb)")
    fm  = row.get("Firmeza mejillas (lb)")
    acid= row.get("Acidez (%)")
    
    scores = {
        "brix": _scale_category(brix, BRIX_RULES[fruit_type]) if pd.notna(brix) else None,
        "fp": _scale_category(fp, FIRMNESS_POINT_DEBIL_RULES) if pd.notna(fp) else None,
        "fm": _scale_category(fm, FIRMNESS_MEJILLA_RULES) if pd.notna(fm) else None,
        "acid": _scale_category(acid, ACIDITY_RULES) if pd.notna(acid) else None,
    }
    # Build a compact composite code like "b1_fp2_fm1_a3"
    label = "_".join(f"{k}{v}" for k, v in scores.items() if v is not None)
    return label or "unclassified"

def apply_rule_based(df, fruit_type="candy_plum"):
    df = df.copy()
    df["rule_cluster"] = df.apply(categorize_row, axis=1, fruit_type=fruit_type)
    return df

############################################################
# ------------------ Algorithmic clusters ---------------- #
############################################################

def run_clustering(df, features, n_clusters=4, method="kmeans", random_state=42):
    X = df[features].dropna()
    idx = X.index
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    elif method == "kmedoids" and HAS_KMEDOIDS:
        model = KMedoids(n_clusters=n_clusters, random_state=random_state)
    else:
        st.warning("Método no soportado o sklearn‑extra no instalado. Usando KMeans por defecto.")
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    
    labels = model.fit_predict(X_scaled)
    df_alg = df.loc[idx].copy()
    df_alg["alg_cluster"] = labels
    
    # PCA for 2‑D display
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    df_alg[["pca1","pca2"]] = coords
    
    return df_alg, model, scaler, pca

############################################################
# ----------------------- Streamlit ----------------------- #
############################################################

st.set_page_config(page_title="Clasificación y Clustering de Variedades", layout="wide")

st.title("🍒 Clasificación de Variedades y Clustering")

# Sidebar: data load
st.sidebar.header("1️⃣ Carga de datos")

uploaded_file = st.sidebar.file_uploader(
    "Sube tu archivo .xlsx de evaluación de cosecha",
    type=["xlsx", "csv"],
)

df_raw = None
if uploaded_file:
    if uploaded_file.name.lower().endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    else:
        try:
            df_raw = pd.read_excel(uploaded_file, sheet_name="Ev. Cosecha Extenso")
        except ValueError:
            st.error("No se encontró la hoja 'Ev. Cosecha Extenso'.")

if df_raw is not None:
    st.subheader("Datos originales")
    st.dataframe(df_raw.head())
    
    # Rule‑based classification
    st.sidebar.header("2️⃣ Clasificación por reglas (Basado en diagrama)")
    fruit_type = st.sidebar.selectbox(
        "Tipo de fruta/variedad",
        ["candy_plum", "cherry_plum"],
        index=0
    )
    if st.sidebar.button("Aplicar clasificación por reglas"):
        df_rules = apply_rule_based(df_raw, fruit_type=fruit_type)
        st.subheader("Resultado clasificación por reglas")
        st.dataframe(df_rules[["rule_cluster"] + df_rules.columns.tolist()])
        csv = df_rules.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar CSV clasificado",
            csv,
            "clasificado_reglas.csv",
            "text/csv"
        )
    
    # Algorithmic clustering
    st.sidebar.header("3️⃣ Clustering matemático")

    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    default_features = [c for c in numeric_cols if "Brix" in c or "Firmeza" in c][:4]
    
    features = st.sidebar.multiselect(
        "Selecciona variables para clustering",
        numeric_cols,
        default=default_features
    )
    n_clusters = st.sidebar.slider("Número de clusters", 2, 10, 4)
    method = st.sidebar.selectbox(
        "Algoritmo",
        ["kmeans"] + (["kmedoids"] if HAS_KMEDOIDS else []),
        index=0
    )
    
    if st.sidebar.button("Ejecutar clustering"):
        if len(features) < 2:
            st.error("Selecciona al menos dos variables númericas.")
        else:
            df_alg, model, scaler, pca = run_clustering(
                df_raw, features, n_clusters=n_clusters, method=method
            )
            st.subheader("Resultado clustering matemático")
            st.dataframe(df_alg[["alg_cluster"] + features].head())
            
            # Plot
            import plotly.express as px
            fig = px.scatter(
                df_alg,
                x="pca1", y="pca2",
                color="alg_cluster",
                title="Clusters (PCA 2D)",
                hover_data=features
            )
            st.plotly_chart(fig, use_container_width=True)
            
            csv2 = df_alg.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar CSV con clusters",
                csv2,
                "clusters_algoritmicos.csv",
                "text/csv"
            )
else:
    st.info("⬅️ Carga un archivo .xlsx para empezar.")
