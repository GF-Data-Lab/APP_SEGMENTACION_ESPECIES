# app_streamlit.py — COMPLETE WORKING SCRIPT (build 2025-06-16-stable)
# -----------------------------------------------------------------------------
# Streamlit app with three tabs:
#   1. Clasificación manual (árbol de decisión) + simulador de fila.
#   2. Clustering algorítmico (K-means, K-medoids, DBSCAN, GMM) con PCA scatter.
#   3. Estadística descriptiva (describe por especie) + boxplots Plotly.
# – Carga la hoja «Ev. Cosecha Extenso» del Excel subido.
# – Mapea nombres de columnas de forma flexible (regex).
# – Evita errores de fecha gracias a _safe_week().
# -----------------------------------------------------------------------------
from __future__ import annotations
import datetime as dt
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except ImportError:
    HAS_KMEDOIDS = False

###############################################################################
# 1. Column mapping helpers                                                   #
###############################################################################
COLUMN_PATTERNS: Dict[str, List[str]] = {
    "especie": ["especie"],
    "variedad": ["variedad"],
    # Brix puede venir en distintas columnas. Se prioriza la columna
    # "Sólidos solubles (%)" y luego otras variantes.
    "brix": [
        "s[oó]lidos?\s*solubles",  # "Sólidos solubles (%)" u otras variantes
        "s[oó]lido\s*soluble",     # columna adicional
        "brix", "°brix", "brx"
    ],
    "fpd": ["firmeza pd", "punto debil"],
    "fm": ["firmeza mj", "firmeza mej"],
    "firmeza": ["firmeza (g)", "firmeza fruit"],
    "acidez": ["acidez"],
    "peso": ["peso", "calibre"],
    "color_pulpa": ["color pulpa"],
    "color_fruto": ["color fruto", "epicarpio"],
    "fecha": ["fecha cosecha", "fecha"],
}

def find_column(df: pd.DataFrame, key: str) -> Optional[str]:
    for pat in COLUMN_PATTERNS.get(key, []):
        rx = re.compile(pat, re.IGNORECASE)
        for col in df.columns:
            if rx.search(str(col)):
                return col
    return None

###############################################################################
# 2. Rule classes + rule tables                                               #
###############################################################################
class Rule:
    def __init__(self, lo: float, hi: float, label: int):
        self.lo, self.hi, self.label = lo, hi, label
    def test(self, v):
        return v is not None and not pd.isna(v) and self.lo <= v <= self.hi

def _cat(v, rules: List[Rule]):
    if v is None or pd.isna(v):
        return None
    for r in rules:
        if r.test(v):
            return r.label
    return None

# --- Ciruelas ---
CIR_SIZE = {
    "candy_plum": lambda w: w is not None and w >= 60,
    "cherry_plum": lambda w: w is not None and w < 60,
}
CIR_BRIX = {
    "candy_plum": [Rule(18, 99, 1), Rule(16, 17.9, 2), Rule(14, 15.9, 3), Rule(-1, 13.9, 4)],
    "cherry_plum": [Rule(21, 99, 1), Rule(18, 20.9, 2), Rule(15, 17.9, 3), Rule(-1, 14.9, 4)],
}
CIR_FPD  = [Rule(7, 99, 1), Rule(5, 6.9, 2), Rule(4, 4.9, 3), Rule(-1, 3.9, 4)]
CIR_FM   = [Rule(9, 99, 1), Rule(7, 8.9, 2), Rule(5, 6.9, 3), Rule(-1, 4.9, 4)]
CIR_ACID = [Rule(-1, 0.79, 1), Rule(0.8, 0.8, 2), Rule(0.81, 0.99, 3), Rule(1, 99, 4)]

# --- Nectarinas ---
NECT_COLOR = {
    "amarilla": lambda c: str(c).lower().startswith("ama"),
    "blanca":   lambda c: str(c).lower().startswith("bla"),
}
NECT_WIN_A = {
    "muy_temprana":   lambda w: 26 <= w <= 28,
    "alta_temperada": lambda w: 29 <= w <= 32,
    "despues_33":     lambda w: w >= 33,
}
NECT_WIN_B = {
    "mayor_temprana": lambda w: 27 <= w <= 29,
    "media_estacion": lambda w: 30 <= w <= 33,
    "tardia":         lambda w: w >= 34,
}
NECT_BRIX = {
    "amarilla": {
        "muy_temprana":   [Rule(15, 99, 1), Rule(13, 14.9, 2), Rule(11, 12.9, 3), Rule(0, 10.9, 4)],
        "alta_temperada": [Rule(16, 99, 1), Rule(14, 15.9, 2), Rule(12, 13.9, 3), Rule(0, 11.9, 4)],
        "despues_33":     [Rule(18, 99, 1), Rule(16, 17.9, 2), Rule(14, 15.9, 3), Rule(0, 13.9, 4)],
    },
    "blanca": {
        "mayor_temprana": [Rule(13, 99, 1), Rule(11, 12.9, 2), Rule(9, 10.9, 3), Rule(0, 8.9, 4)],
        "media_estacion": [Rule(14, 99, 1), Rule(12, 13.9, 2), Rule(10, 11.9, 3), Rule(0, 9.9, 4)],
        "tardia":         [Rule(16, 99, 1), Rule(14, 15.9, 2), Rule(12, 13.9, 3), Rule(0, 11.9, 4)],
    },
}
NECT_FPD  = [Rule(8, 99, 1), Rule(6, 7.9, 2), Rule(4, 5.9, 3), Rule(0, 3.9, 4)]
NECT_FM   = [Rule(10, 99, 1), Rule(8, 9.9, 2), Rule(6, 7.9, 3), Rule(0, 5.9, 4)]
NECT_ACID = [Rule(0, 0.6, 4), Rule(0.61, 0.8, 3), Rule(0.81, 1.0, 2), Rule(1.01, 99, 1)]

# --- Cerezos ---
CER_COLOR = {
    "roja":    lambda c: str(c).lower().startswith("ro"),
    "bicolor": lambda c: str(c).lower().startswith("bi"),
}
CER_WIN = {
    "temprana": lambda w: 21 <= w <= 26,
    "media":    lambda w: 27 <= w <= 30,
    "tardia":   lambda w: w >= 31,
}
CER_BRIX = {
    "roja":    [Rule(16, 99, 1), Rule(14, 15.9, 2), Rule(12, 13.9, 3), Rule(0, 11.9, 4)],
    "bicolor": [Rule(17, 99, 1), Rule(15, 16.9, 2), Rule(13, 14.9, 3), Rule(0, 12.9, 4)],
}
CER_FIRM = [Rule(350, 999, 1), Rule(280, 349, 2), Rule(220, 279, 3), Rule(0, 219, 4)]
CER_ACID = [Rule(0, 0.5, 4), Rule(0.51, 0.7, 3), Rule(0.71, 0.9, 2), Rule(0.91, 99, 1)]

###############################################################################
# 3. Helper utilities                                                         #
###############################################################################

def _grade(scores: List[int | None]) -> int:
    valid = [s for s in scores if s is not None]
    if not valid:
        return 5
    m = sum(valid) / len(valid)
    if m <= 1.5:
        return 1
    if m <= 2.5:
        return 2
    if m <= 3.5:
        return 3
    return 4

def _safe_week(value) -> Optional[int]:
    """Return ISO week number from a date-like value."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (dt.date, dt.datetime)):
        return int(value.isocalendar()[1])
    try:
        d = pd.to_datetime(value)
        return int(d.isocalendar().week)
    except Exception:
        return None

def _map_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    # Identificar posibles columnas de Brix antes de renombrar
    rx_main = re.compile("s[oó]lidos?\s*solubles", re.IGNORECASE)
    rx_extra = re.compile("s[oó]lido\s*soluble", re.IGNORECASE)
    brix_main = next((c for c in df.columns if rx_main.search(str(c))), None)
    brix_extra = next((c for c in df.columns if rx_extra.search(str(c)) and c != brix_main), None)

    for key in COLUMN_PATTERNS:
        col = find_column(df, key)
        if col:
            mapping[key] = col

    if brix_main:
        mapping["brix"] = brix_main
    if brix_extra:
        mapping["brix_extra"] = brix_extra

    df = df.rename(columns=mapping)

    if "brix_extra" in df.columns:
        if "brix" in df.columns:
            df["brix"] = df["brix"].fillna(df["brix_extra"])
        else:
            df.rename(columns={"brix_extra": "brix"}, inplace=True)

    return df, mapping

###############################################################################
# 4. Rule-based classification                                                #
###############################################################################

def classify_plum(row: pd.Series) -> int:
    peso = row.get("peso")
    variedad = "candy_plum" if CIR_SIZE["candy_plum"](peso) else "cherry_plum"
    scores = [
        _cat(row.get("brix"), CIR_BRIX[variedad]),
        _cat(row.get("fpd"), CIR_FPD),
        _cat(row.get("fm"), CIR_FM),
        _cat(row.get("acidez"), CIR_ACID),
    ]
    return _grade(scores)

def classify_nectarine(row: pd.Series) -> int:
    color = next((k for k,f in NECT_COLOR.items() if f(row.get("color_pulpa"))), None)
    semana = _safe_week(row.get("fecha"))
    if color == "amarilla":
        win = next((k for k,f in NECT_WIN_A.items() if f(semana)), None)
    else:
        win = next((k for k,f in NECT_WIN_B.items() if f(semana)), None)
    br_rules = NECT_BRIX.get(color, {}).get(win, [])
    scores = [
        _cat(row.get("brix"), br_rules),
        _cat(row.get("fpd"), NECT_FPD),
        _cat(row.get("fm"), NECT_FM),
        _cat(row.get("acidez"), NECT_ACID),
    ]
    return _grade(scores)

def classify_cherry(row: pd.Series) -> int:
    color = next((k for k,f in CER_COLOR.items() if f(row.get("color_fruto"))), None)
    semana = _safe_week(row.get("fecha"))
    win = next((k for k,f in CER_WIN.items() if f(semana)), None)
    br_rules = CER_BRIX.get(color, [])
    scores = [
        _cat(row.get("brix"), br_rules),
        _cat(row.get("firmeza"), CER_FIRM),
        _cat(row.get("acidez"), CER_ACID),
    ]
    return _grade(scores)

def classify_row(row: pd.Series) -> int:
    especie = str(row.get("especie")).lower()
    if "ciruela" in especie:
        return classify_plum(row)
    if "nectarina" in especie:
        return classify_nectarine(row)
    if "cerezo" in especie or "cereza" in especie:
        return classify_cherry(row)
    return 5

###############################################################################
# 5. Algorithmic clustering                                                   #
###############################################################################

def run_clustering(
    df: pd.DataFrame,
    features: List[str],
    method: str = "kmeans",
    n_clusters: int = 4,
    eps: float = 0.5,
    min_samples: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, object, PCA, StandardScaler]:
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        labels = model.fit_predict(X_scaled)
    elif method == "kmedoids" and HAS_KMEDOIDS:
        model = KMedoids(n_clusters=n_clusters, random_state=random_state)
        labels = model.fit_predict(X_scaled)
    elif method == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
    elif method == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=random_state)
        labels = model.fit_predict(X_scaled)
    else:
        raise ValueError("Método de clustering no soportado")

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    df_out = df.loc[X.index].copy()
    df_out["cluster"] = labels
    df_out["pca1"], df_out["pca2"] = coords[:,0], coords[:,1]

    return df_out, model, pca, scaler

###############################################################################
# 6. Streamlit interface                                                      #
###############################################################################

st.set_page_config(page_title="Segmentación de especies", layout="wide")

st.title("🍑 App de Segmentación de Especies")

uploaded_file = st.sidebar.file_uploader(
    "Sube tu archivo .xlsx con la hoja 'Ev. Cosecha Extenso'",
    type=["xlsx"],
)

if uploaded_file:
    df_orig = pd.read_excel(uploaded_file, sheet_name="Ev. Cosecha Extenso")
    df_orig, mapping = _map_columns(df_orig)
    st.sidebar.success("Archivo cargado")

    tab1, tab2, tab3 = st.tabs([
        "Clasificación manual",
        "Clustering algorítmico",
        "Estadística descriptiva",
    ])

    with tab1:
        st.header("Clasificación manual por reglas")
        df_class = df_orig.copy()
        df_class["clasificacion"] = df_class.apply(classify_row, axis=1)
        st.dataframe(df_class.head())
        # Análisis exploratorio simple por clasificación
        if "clasificacion" in df_class.columns:
            num_cols = df_class.select_dtypes(include=[np.number]).columns
            if len(num_cols) >= 2:
                chart = (
                    alt.Chart(df_class)
                    .mark_circle(size=60)
                    .encode(
                        x=alt.X(num_cols[0]+':Q'),
                        y=alt.Y(num_cols[1]+':Q'),
                        color="clasificacion:N",
                        tooltip=list(df_class.columns)
                    )
                    .properties(title="Exploraci\u00f3n de datos (clasificaci\u00f3n)")
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)
        with st.expander("Ingresar fruto manualmente"):
            with st.form("manual_form"):
                especie = st.text_input("Especie")
                variedad = st.text_input("Variedad")
                brix = st.number_input("Brix", step=0.1)
                fpd = st.number_input("Firmeza PD (lb)", step=0.1)
                fm = st.number_input("Firmeza MJ (lb)", step=0.1)
                firmeza = st.number_input("Firmeza (g)", step=1.0)
                acidez = st.number_input("Acidez %", step=0.01)
                peso = st.number_input("Peso (g)", step=1.0)
                color_pulpa = st.text_input("Color pulpa")
                color_fruto = st.text_input("Color fruto")
                fecha = st.date_input("Fecha cosecha", value=dt.date.today())
                submit = st.form_submit_button("Clasificar")
            if submit:
                row = pd.Series({
                    "especie": especie,
                    "variedad": variedad,
                    "brix": brix,
                    "fpd": fpd,
                    "fm": fm,
                    "firmeza": firmeza,
                    "acidez": acidez,
                    "peso": peso,
                    "color_pulpa": color_pulpa,
                    "color_fruto": color_fruto,
                    "fecha": fecha,
                })
                grade = classify_row(row)
                st.success(f"Clasificación obtenida: {grade}")
        csv = df_class.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar clasificación",
            csv,
            "clasificacion.csv",
            mime="text/csv",
        )

    with tab2:
        st.header("Clustering algorítmico")
        numeric_cols = df_orig.select_dtypes(include=[np.number]).columns.tolist()
        features = st.multiselect(
            "Variables", options=numeric_cols, default=numeric_cols[:4]
        )
        method = st.selectbox(
            "Método",
            ["kmeans", "kmedoids" if HAS_KMEDOIDS else None, "dbscan", "gmm"],
            format_func=lambda x: x if x else "kmeans",
        )
        method = method or "kmeans"
        n_clusters = st.slider("Número de clusters", 2, 10, 4)
        eps = st.number_input("DBSCAN eps", 0.1, 5.0, 0.5, 0.1)
        min_samples = st.number_input("DBSCAN min_samples", 1, 20, 5)

        if st.button("Calcular clustering"):
            try:
                df_cluster, model, pca, scaler = run_clustering(
                    df_orig, features, method, n_clusters, eps, min_samples
                )
                st.dataframe(df_cluster.head())
                import plotly.express as px

                fig = px.scatter(
                    df_cluster,
                    x="pca1",
                    y="pca2",
                    color="cluster",
                    hover_data=features,
                    title="Clusters (PCA 2D)",
                )
                st.plotly_chart(fig, use_container_width=True)
                # Exploración de datos de clusters con Altair
                chart = (
                    alt.Chart(df_cluster)
                    .mark_circle(size=60)
                    .encode(
                        x="pca1",
                        y="pca2",
                        color="cluster:N",
                        tooltip=features
                    )
                    .properties(title="Clusters PCA Altair")
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)
                csvc = df_cluster.to_csv(index=False).encode("utf-8")
                st.download_button("Descargar clusters", csvc, "clusters.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error en clustering: {e}")

    with tab3:
        st.header("Estadística descriptiva")
        especie_col = "especie" if "especie" in df_orig.columns else df_orig.columns[0]
        stats = df_orig.groupby(especie_col).describe()
        st.dataframe(stats)
        num_cols = df_orig.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            chart = (
                alt.Chart(df_orig)
                .mark_boxplot()
                .encode(
                    x=alt.X(f"{especie_col}:N", title="Especie"),
                    y=alt.Y(f"{col}:Q", title=col),
                    color=f"{especie_col}:N",
                )
                .properties(title=f"Distribución de {col}")
            )
            st.altair_chart(chart, use_container_width=True)
else:
    st.info("Sube un archivo para comenzar")
