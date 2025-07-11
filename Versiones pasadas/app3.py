# app.py
# -----------------------------------------------------------------------------
# Streamlit APP: Clasificación de variedades frutales
# (c) 2025  --  Elaborado a partir del árbol de decisión provisto en imagen
# -----------------------------------------------------------------------------
import math
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except ImportError:
    HAS_KMEDOIDS = False
from sklearn.mixture import GaussianMixture

# -----------------------------------------------------------------------------
# ---------------------------- 1. UTILIDADES GENERALES ------------------------
# -----------------------------------------------------------------------------
@dataclass
class Rule:
    lo: float                 # límite inferior incluido
    hi: float                 # límite superior incluido
    label: str | int          # etiqueta / categoría

    def test(self, x: float | None):
        return x is not None and self.lo <= x <= self.hi

def _categorize(value: float | None, rules: list[Rule]):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    for rule in rules:
        if rule.test(value):
            return rule.label
    return None

# -----------------------------------------------------------------------------
# ---------------------------- 2. REGLAS MANUALES -----------------------------
# -----------------------------------------------------------------------------
#
#  Se extraen de la imagen “Modelo de clasificación de variedades”.
#  Para cada especie se definen:
#    • filtros visuales de descarte      (booleans en la fila)
#    • ramas del árbol (peso, color, semana de cosecha, etc.)
#    • tablas de umbrales   Brix / Firmeza / Acidez   → categoría (1‑4)
#
#  Notas:
#    · Las ‘semanas de cosecha’ se refieren a la semana ISO (1‑53) del año.
#    · Los textos largos de la imagen se simplifican a variables booleanas
#      (“manchas”, “rajado”, etc.) que el usuario debe suministrar.
# -----------------------------------------------------------------------------


# 2‑A)  CIRUELAS  ──────────────────────────────────────────────────────────────
CIRUELA_SIZE_RULE = {           # Calibre → grupo principal
    "candy_plum" : lambda peso: peso >= 60,   # ≥ 60 g
    "cherry_plum": lambda peso: peso < 60     #  < 60 g
}

CIRUELA_BRIX = {
    "candy_plum": [
        Rule(18.0,  99.9, 1),
        Rule(16.0, 17.9, 2),
        Rule(14.0, 15.9, 3),
        Rule(-1.0, 13.9, 4),
    ],
    "cherry_plum": [
        Rule(21.0,  99.9, 1),
        Rule(18.0, 20.9, 2),
        Rule(15.0, 17.9, 3),
        Rule(-1.0, 14.9, 4),
    ]
}
CIRUELA_FPD = [Rule(7.0, 99, 1), Rule(5.0, 6.9, 2),
               Rule(4.0, 4.9, 3), Rule(-1, 3.9, 4)]
CIRUELA_FM  = [Rule(9.0, 99, 1), Rule(7.0, 8.9, 2),
               Rule(5.0, 6.9, 3), Rule(-1, 4.9, 4)]
CIRUELA_ACIDEZ = [Rule(-1, 0.79, 1), Rule(0.80, 0.8, 2),
                  Rule(0.81, 0.99, 3), Rule(1.0, 99, 4)]

# 2‑B)  NECTARINAS  ────────────────────────────────────────────────────────────
NECTARINA_COLOR = {            # Color pulpa
    "amarilla": lambda color: str(color).strip().lower().startswith("ama"),
    "blanca"  : lambda color: str(color).strip().lower().startswith("bla")
}
#  Ventanas de cosecha (semana ISO)   → subgrupo
NECT_WIN_AMARILLAS = {
    "muy_temprana"   : lambda wk: 26 <= wk <= 28,
    "alta_temperada" : lambda wk: 29 <= wk <= 32,
    "después_33"     : lambda wk: wk >= 33,
}
NECT_WIN_BLANCAS = {
    "mayor_temprana" : lambda wk: 27 <= wk <= 29,
    "media_estación" : lambda wk: 30 <= wk <= 33,
    "tardía"         : lambda wk: wk >= 34,
}
#  Las tablas del diagrama (Brix/Firmeza/Acidez) por subgrupo
#  Se simplifican:   1 = excelente, 2 = buena, 3 = aceptable, 4 = baja
NECT_BRIX = {
    "amarilla": {
        "muy_temprana": [Rule(15, 99, 1), Rule(13, 14.9, 2),
                         Rule(11, 12.9, 3), Rule(0, 10.9, 4)],
        "alta_temperada": [Rule(16, 99, 1), Rule(14, 15.9, 2),
                           Rule(12, 13.9, 3), Rule(0, 11.9, 4)],
        "después_33": [Rule(18, 99, 1), Rule(16, 17.9, 2),
                       Rule(14, 15.9, 3), Rule(0, 13.9, 4)],
    },
    "blanca": {
        "mayor_temprana": [Rule(13, 99, 1), Rule(11, 12.9, 2),
                           Rule(9, 10.9, 3), Rule(0, 8.9, 4)],
        "media_estación": [Rule(14, 99, 1), Rule(12, 13.9, 2),
                           Rule(10, 11.9, 3), Rule(0, 9.9, 4)],
        "tardía": [Rule(16, 99, 1), Rule(14, 15.9, 2),
                   Rule(12, 13.9, 3), Rule(0, 11.9, 4)],
    }
}
#  Firmeza Punto Débil y Mejillas: mismas reglas para todas
NECT_FPD = [Rule(8, 99, 1), Rule(6, 7.9, 2),
            Rule(4, 5.9, 3), Rule(0, 3.9, 4)]
NECT_FM  = [Rule(10, 99, 1), Rule(8, 9.9, 2),
            Rule(6, 7.9, 3), Rule(0, 5.9, 4)]
NECT_ACIDEZ = [Rule(0, 0.6, 4), Rule(0.61, 0.8, 3),
               Rule(0.81, 1.0, 2), Rule(1.01, 99, 1)]
# 2‑C)  CEREZOS  ───────────────────────────────────────────────────────────────
CEREZA_COLOR = {
    "roja"   : lambda col: str(col).strip().lower().startswith("ro"),
    "bicolor": lambda col: str(col).strip().lower().startswith("bi")
}
CEREZA_WIN = {
    "temprana"       : lambda wk: 21 <= wk <= 26,
    "media_estación" : lambda wk: 27 <= wk <= 30,
    "tardía"         : lambda wk: wk >= 31,
}
CEREZA_BRIX = {
    "roja":      [Rule(16, 99, 1), Rule(14, 15.9, 2),
                  Rule(12, 13.9, 3), Rule(0, 11.9, 4)],
    "bicolor":   [Rule(17, 99, 1), Rule(15, 16.9, 2),
                  Rule(13, 14.9, 3), Rule(0, 12.9, 4)],
}
CEREZA_FIRMEZA = [Rule(350, 999, 1), Rule(280, 349, 2),
                  Rule(220, 279, 3), Rule(0, 219, 4)]  # g fuerza (Dynamometer)
CEREZA_ACIDEZ = [Rule(0, 0.5, 4), Rule(0.51, 0.7, 3),
                 Rule(0.71, 0.9, 2), Rule(0.91, 99, 1)]

# -----------------------------------------------------------------------------
# ------------------------- 3. CLASIFICACIÓN MANUAL ---------------------------
# -----------------------------------------------------------------------------
def clasificar_ciruela(row: pd.Series):
    # 0) Filtro de descarte visual (prototipo)
    if row.get("Descarte_visual", False):
        return "DESCARTE_VISUAL"
    # 1) Candy o Cherry
    peso = row.get("Peso", np.nan)
    grupo = next((g for g, fn in CIRUELA_SIZE_RULE.items() if fn(peso)), "desconocido")
    if grupo == "desconocido":
        return "PESO_NO_CLASIF"
    # 2) Categorizar propiedades
    b_idx = _categorize(row.get("Brix"), CIRUELA_BRIX[grupo])
    fpd   = _categorize(row.get("Firmeza_punto_debil"), CIRUELA_FPD)
    fm    = _categorize(row.get("Firmeza_mejillas"), CIRUELA_FM)
    acid  = _categorize(row.get("Acidez"), CIRUELA_ACIDEZ)
    return f"{grupo}|B{b_idx}-FP{fpd}-FM{fm}-A{acid}"

def clasificar_nectarina(row: pd.Series):
    if row.get("Descarte_visual", False):
        return "DESCARTE_VISUAL"
    # 1) Color pulpa
    color_key = next((k for k, fn in NECTARINA_COLOR.items() if fn(row.get("Color_pulpa",""))), None)
    if color_key is None:
        return "COLOR_NO_CLASIF"
    # 2) Semana cosecha
    fecha = row.get("Fecha_cosecha")
    if pd.isna(fecha):
        return "SIN_FECHA"
    wk = int(fecha.isocalendar()[1]) if isinstance(fecha, (dt.date, pd.Timestamp)) else int(fecha)
    win_key = next((k for k, fn in (NECT_WIN_AMARILLAS if color_key=="amarilla" else NECT_WIN_BLANCAS).items() if fn(wk)), None)
    if win_key is None:
        return "SEMANA_SIN_RANGO"
    # 3) Umbrales
    brix = _categorize(row.get("Brix"), NECT_BRIX[color_key][win_key])
    fpd  = _categorize(row.get("Firmeza_punto_debil"), NECT_FPD)
    fm   = _categorize(row.get("Firmeza_mejillas"), NECT_FM)
    acid = _categorize(row.get("Acidez"), NECT_ACIDEZ)
    return f"{color_key}-{win_key}|B{brix}-FP{fpd}-FM{fm}-A{acid}"

def clasificar_cereza(row: pd.Series):
    if row.get("Descarte_visual", False):
        return "DESCARTE_VISUAL"
    # 1) Color fruto
    color_key = next((k for k, fn in CEREZA_COLOR.items() if fn(row.get("Color_fruto",""))), None)
    if color_key is None:
        return "COLOR_NO_CLASIF"
    # 2) Semana cosecha
    fecha = row.get("Fecha_cosecha")
    wk = int(fecha.isocalendar()[1]) if isinstance(fecha, (dt.date, pd.Timestamp)) else int(fecha)
    win_key = next((k for k, fn in CEREZA_WIN.items() if fn(wk)), None)
    if win_key is None:
        return "SEMANA_SIN_RANGO"
    brix = _categorize(row.get("Brix"), CEREZA_BRIX[color_key])
    firm = _categorize(row.get("Firmeza"), CEREZA_FIRMEZA)
    acid = _categorize(row.get("Acidez"), CEREZA_ACIDEZ)
    return f"{color_key}-{win_key}|B{brix}-F{firm}-A{acid}"

def clasificar_fila(row: pd.Series):
    especie = str(row.get("Especie","")).strip().lower()
    if especie.startswith("ciru"):
        return clasificar_ciruela(row)
    if especie.startswith("necta"):
        return clasificar_nectarina(row)
    if especie.startswith("cere"):
        return clasificar_cereza(row)
    return "ESPECIE_DESCONOCIDA"

# -----------------------------------------------------------------------------
# ------------------------- 4. CLUSTERING ALGORÍTMICO --------------------------
# -----------------------------------------------------------------------------
def clusterizar(
    df: pd.DataFrame,
    cols: list[str],
    metodo: str = "kmeans",
    k: int = 3,
    eps: float = 0.5,
    minpts: int = 5,
    random_state: int = 0,
):
    X = df[cols].dropna()
    idx = X.index
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if metodo == "kmeans":
        modelo = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = modelo.fit_predict(Xs)
    elif metodo == "kmedoids" and HAS_KMEDOIDS:
        modelo = KMedoids(n_clusters=k, random_state=random_state)
        labels = modelo.fit_predict(Xs)
    elif metodo == "dbscan":
        modelo = DBSCAN(eps=eps, min_samples=minpts)
        labels = modelo.fit_predict(Xs)
    elif metodo == "gmm":
        modelo = GaussianMixture(n_components=k, random_state=random_state)
        labels = modelo.fit_predict(Xs)
    else:
        raise ValueError("Método de clustering no soportado.")

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(Xs)

    out = df.copy()
    out.loc[idx, "cluster_label"] = labels
    out.loc[idx, "pca1"] = coords[:,0]
    out.loc[idx, "pca2"] = coords[:,1]

    return out, modelo, scaler, pca

# -----------------------------------------------------------------------------
# ------------------------- 5. INTERFAZ STREAMLIT -----------------------------
# -----------------------------------------------------------------------------
st.set_page_config("Clasificación de Variedades", layout="wide")
st.title("🍑🍒 Clasificación y análisis de variedades frutales")

with st.sidebar:
    st.header("Carga de datos")
    archivo = st.file_uploader("Sube tu archivo .xlsx / .csv", type=["xlsx","csv"])
    if archivo:
        if Path(archivo.name).suffix.lower().startswith(".csv"):
            df = pd.read_csv(archivo)
        else:
            df = pd.read_excel(archivo)
        st.success(f"Archivo cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    else:
        st.info("Cargar un archivo para empezar.")

TAB1, TAB2, TAB3 = st.tabs(["📋 Clasificación manual", "📊 Clustering", "📈 Estadística"])

# ------------------- 5‑A  CLASIFICACIÓN MANUAL --------------------------------
with TAB1:
    if archivo:
        st.subheader("Datos originales")
        st.dataframe(df.head(50), use_container_width=True)
        # 1) aplicamos clasificación
        df_clas = df.copy()
        df_clas["clasificación"] = df_clas.apply(clasificar_fila, axis=1)
        st.subheader("Resultado")
        st.dataframe(df_clas[["Especie","clasificación"] + [c for c in df_clas.columns if c not in ("Especie","clasificación")]],
                     hide_index=True, use_container_width=True)
        # botón de descarga
        csv_bytes = df_clas.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar CSV clasificado", csv_bytes, "clasificado.csv", "text/csv")

    st.markdown("---")
    st.subheader("🔬 Simular nueva variedad")
    col1, col2 = st.columns(2)
    especie_sim = col1.selectbox("Especie", ["Ciruela","Nectarina","Cereza"])
    peso_sim    = col2.number_input("Peso (g)", 10.0, 200.0, 60.0)
    brix_sim    = col1.number_input("Brix (%)", 5.0, 30.0, 16.0, step=0.1)
    fpd_sim     = col2.number_input("Firmeza punto débil", 0.0, 20.0, 6.0, step=0.1)
    fm_sim      = col1.number_input("Firmeza mejillas / o Firmeza (cereza)", 0.0, 20.0, 8.0, step=0.1)
    acid_sim    = col2.number_input("Acidez (%)", 0.0, 5.0, 0.8, step=0.01)
    color_sim   = col1.text_input("Color pulpa (nect.) o fruto (cereza)", "Amarilla")
    fecha_sim   = col2.date_input("Fecha cosecha", dt.date.today())
    desc_sim    = st.checkbox("Presenta defectos visuales que descartan", value=False)

    if st.button("Clasificar simulación"):
        fila = pd.Series({
            "Especie": especie_sim,
            "Peso": peso_sim,
            "Brix": brix_sim,
            "Firmeza_punto_debil": fpd_sim,
            "Firmeza_mejillas": fm_sim,
            "Firmeza": fm_sim,  # para cereza
            "Acidez": acid_sim,
            "Color_pulpa": color_sim,
            "Color_fruto": color_sim,
            "Fecha_cosecha": fecha_sim,
            "Descarte_visual": desc_sim
        })
        resultado = clasificar_fila(fila)
        st.success(f"Resultado: **{resultado}**")

# ------------------- 5‑B  CLUSTERING -----------------------------------------
with TAB2:
    if not archivo:
        st.info("Carga primero un archivo para habilitar el clustering.")
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.write("Variables numéricas detectadas:", ", ".join(num_cols))
        sel_cols = st.multiselect("Selecciona variables para clustering", num_cols, default=num_cols[:4])
        metodo   = st.selectbox("Algoritmo",
                                options=["kmeans", "kmedoids" if HAS_KMEDOIDS else "— kmedoids (no instalado) —",
                                         "dbscan", "gmm"])
        if metodo == "kmeans" or metodo == "kmedoids" or metodo == "gmm":
            k = st.slider("k (número de clusters)", 2, 10, 3)
        if metodo == "dbscan":
            eps = st.number_input("eps", 0.1, 10.0, 0.5, 0.1)
            minpts = st.number_input("minPts", 2, 20, 5, 1)

        if st.button("Ejecutar clustering"):
            if len(sel_cols) < 2:
                st.error("Selecciona al menos dos variables.")
            else:
                df_clust, modelo, scaler, pca = clusterizar(
                    df, sel_cols, metodo,
                    k=k if metodo!="dbscan" else None,
                    eps=eps if metodo=="dbscan" else None,
                    minpts=minpts if metodo=="dbscan" else None
                )
                st.success("Clustering completado.")
                st.dataframe(df_clust[["cluster_label"] + sel_cols], use_container_width=True)

                # Gráfico PCA2D
                import plotly.express as px
                fig = px.scatter(
                    df_clust, x="pca1", y="pca2",
                    color=df_clust["cluster_label"].astype(str),
                    hover_data=sel_cols,
                    title="Proyección PCA (2 D)"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Radar de perfiles promedios
                st.subheader("Perfil promedio por cluster (Radar)")
                # Normalizar 0‑1 por columna para que todos los ejes tengan igual peso
                df_norm = df_clust.groupby("cluster_label")[sel_cols].mean()
                df_min = df[sel_cols].min()
                df_max = df[sel_cols].max()
                df_scaled = (df_norm - df_min) / (df_max - df_min)

                import plotly.graph_objects as go
                fig_radar = go.Figure()
                for idx, row in df_scaled.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=row.values,
                        theta=sel_cols,
                        fill='toself',
                        name=f"Cluster {idx}"
                    ))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                                         showlegend=True)
                st.plotly_chart(fig_radar, use_container_width=True)

# ------------------- 5‑C  ESTADÍSTICA DESCRIPTIVA ----------------------------
with TAB3:
    if not archivo:
        st.info("Carga un archivo primero.")
    else:
        st.subheader("Tabla descriptiva por especie")
        desc = df.groupby("Especie").describe().round(2)
        st.dataframe(desc)
        st.markdown("**Outliers (IQR) por especie y métrica**")
        out_tbl = []
        for esp, gdf in df.groupby("Especie"):
            for col in num_cols:
                q1, q3 = gdf[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = gdf[(gdf[col] < q1 - 1.5*iqr) | (gdf[col] > q3 + 1.5*iqr)]
                if not outliers.empty:
                    out_tbl.append({"Especie": esp,
                                    "Variable": col,
                                    "N outliers": len(outliers),
                                    "Índices": list(outliers.index)})
        if out_tbl:
            st.dataframe(pd.DataFrame(out_tbl))
        else:
            st.write("No se detectaron outliers con el criterio IQR.")

        st.subheader("Boxplots")
        import plotly.express as px
        for col in num_cols:
            fig = px.box(df, x="Especie", y=col, points="outliers", title=f"Distribución de {col}")
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# FIN DEL ARCHIVO
# -----------------------------------------------------------------------------
