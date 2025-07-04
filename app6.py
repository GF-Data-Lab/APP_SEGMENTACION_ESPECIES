# streamlit_app.py
"""Streamlit App – Clasificación y Clustering en Carozos
======================================================
Ejecuta en tu terminal:
```bash
streamlit run streamlit_app.py
```

### Dependencias
```
pip install streamlit pandas numpy scikit-learn scikit-learn-extra
```
`scikit-learn‑extra` es *opcional* ‑ sólo se necesita si quieres K‑Medoids.

La aplicación cuenta ahora con **dos pestañas complementarias** que te permiten
no perder el enfoque “reglas vs machine learning”:
1. **Clasificación (Reglas)** – corre exactamente el pipeline de
   `procesar_carozos.py` (versión 7) y muestra la tabla resultante.
   También puedes **simular** una muestra manual.
2. **Clustering avanzado** – ejecuta K‑Means, K‑Medoids o Clustering
   aglomerativo y compara el resultado contra los clústeres definidos por las
   reglas (individual *o* agregados). Para cuantificar la concordancia se
   incluyen:
   * **Matriz de confusión** (reglas × ML).
   * **Adjusted Rand Index (ARI)**.
   * Estadísticas descriptivas por clúster y variedad.

---
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

# K‑Medoids es opcional
try:
    from sklearn_extra.cluster import KMedoids
except ImportError:  # pragma: no cover
    KMedoids = None  # type: ignore

# ---------------------------------------------------------------------------
# Dependencia interna --------------------------------------------------------
# ---------------------------------------------------------------------------
from procesar_carozos import process_carozos  # importa la versión 7

# ---------------------------------------------------------------------------
# Config. global de la app ---------------------------------------------------
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Carozos – Clasificación y Clustering",
    layout="wide",
    page_icon="🍑",
)

# Barra lateral – navegación -------------------------------------------------
page = st.sidebar.radio(
    "Selecciona sección",
    ["Clasificación (Reglas)", "Clustering avanzado"],
)

# ---------------------------------------------------------------------------
# Utilidades -----------------------------------------------------------------
# ---------------------------------------------------------------------------

def _read_excel(upload) -> pd.DataFrame:
    try:
        return pd.read_excel(upload, sheet_name="CAROZOS", dtype=str)
    except Exception as e:
        st.error(f"No se pudo leer el Excel: {e}")
        raise

# ---------------------------------------------------------------------------
# 1 · Página de clasificación por reglas -------------------------------------
# ---------------------------------------------------------------------------
if page == "Clasificación (Reglas)":
    st.header("Clasificación de muestras según reglas v7")

    tab_archivo, tab_manual = st.tabs(["Procesar Excel", "Ingresar muestra"])

    # 1‑A Procesar archivo completo ---------------------------------------
    with tab_archivo:
        st.subheader("Procesar archivo completo")
        uploaded = st.file_uploader("Sube tu archivo Excel de laboratorio", type=["xlsx"])
        if uploaded:
            tmp = io.BytesIO(uploaded.read())
            try:
                df = process_carozos(tmp)
                st.success("Archivo procesado correctamente ✅")
                st.dataframe(df.head(1000))
                st.download_button(
                    label="Descargar tabla completa (CSV)",
                    data=df.to_csv(index=False).encode(),
                    file_name="clasificacion_carozos.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.exception(e)
        else:
            st.info("Mostrando las primeras 50 filas del maestro predeterminado…")
            df_master = process_carozos()
            st.dataframe(df_master.head(50))

    # 1‑B Simulación de muestra -------------------------------------------
    with tab_manual:
        st.subheader("Simular una muestra individual")
        especie = st.selectbox("Especie", ["Ciruela", "Nectarin"])
        variedad = st.text_input("Variedad", value="TestVar")
        fruto = st.number_input("N° de fruto", value=1, step=1, min_value=1)
        peso = st.number_input("Peso (g)", value=65.0, min_value=1.0)
        fecha = st.date_input("Fecha de cosecha")
        brix = st.number_input("°Brix", value=17.5)
        acidez = st.number_input("Acidez (%)", value=0.75)
        prod = st.number_input("Productividad (Ton)", value=30.0)
        col_pulpa = "Amarilla"
        if especie == "Nectarin":
            col_pulpa = st.selectbox("Color de pulpa", ["Amarilla", "Blanca"])
        firmezas = {}
        with st.expander("Firmezas (lb)"):
            firmezas["Quilla"] = st.number_input("Quilla", value=6.5)
            firmezas["Hombro"] = st.number_input("Hombro", value=6.0)
            firmezas["Mejilla 1"] = st.number_input("Mejilla 1", value=7.0)
            firmezas["Mejilla 2"] = st.number_input("Mejilla 2", value=6.8)

        if st.button("Clasificar muestra"):
            row = {
                "Especie": especie,
                "Variedad": variedad,
                "Fruto (n°)": fruto,
                "Peso (g)": peso,
                "Fecha cosecha": fecha,
                "Solidos solubles (%)": brix,
                "Acidez (%)": acidez,
                "Productividad (Ton)": prod,
                "Color de pulpa": col_pulpa,
            }
            row.update(firmezas)
            df_single = pd.DataFrame([row])
            # Guardamos en un buffer Excel para reutilizar process_carozos
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                df_single.to_excel(writer, sheet_name="CAROZOS", index=False)
            out.seek(0)
            clasif = process_carozos(out)
            st.dataframe(clasif)

# ---------------------------------------------------------------------------
# 2 · Página de Clustering avanzado -----------------------------------------
# ---------------------------------------------------------------------------
if page == "Clustering avanzado":
    st.header("Clustering avanzado de muestras")

    uploaded = st.file_uploader(
        "Sube un Excel de laboratorio (o vací­o para usar el maestro)",
        type=["xlsx"],
    )
    if uploaded:
        tmp = io.BytesIO(uploaded.read())
        base_df = process_carozos(tmp)
    else:
        base_df = process_carozos()

    # -------- Nivel de agregación --------------------------------------
    agg_level = st.selectbox(
        "Nivel de agregación para clustering",
        ["Muestra individual", "Variedad + Fruto"],
    )

    if agg_level == "Muestra individual":
        df_cluster = base_df.copy()
        rule_col = "cluster_row"
    else:
        grp_cols = ["Variedad", "Fruto (n°)"]
        df_cluster = (
            base_df.groupby(grp_cols, dropna=False)
            .mean(numeric_only=True)
            .reset_index()
        )
        rule_col = "cluster_grp"

    # -------- Variables numéricas candidatas ---------------------------
    numeric_cols = [c for c in df_cluster.columns if c.startswith("grp_")]
    st.write("Variables numéricas disponibles para clustering:", numeric_cols)
    feats = st.multiselect("Selecciona variables", numeric_cols, default=numeric_cols)

    # -------- Configuración de algoritmo -------------------------------
    algo = st.selectbox(
        "Algoritmo", ["K‑Means", "K‑Medoids" if KMedoids else "K‑Medoids (no instalado)", "Agglomerativo"]
    )
    k = st.slider("N° de clústeres (k)", min_value=2, max_value=10, value=4)

    # -------- Ejecución -------------------------------------------------
    if st.button("Ejecutar clustering") and feats:
        X = df_cluster[feats].astype(float)
        # Relleno de NaNs con la media de cada columna ------------------
        X = X.fillna(X.mean())
        X_std = StandardScaler().fit_transform(X)

        if algo.startswith("K‑Medoids"):
            if KMedoids is None:
                st.error("Necesitas instalar scikit‑learn‑extra para usar K‑Medoids")
                st.stop()
            model = KMedoids(n_clusters=k, random_state=0).fit(X_std)
        elif algo == "Agglomerativo":
            model = AgglomerativeClustering(n_clusters=k).fit(X_std)
        else:  # K‑Means
            model = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X_std)

        df_cluster["cluster_ml"] = model.labels_ + 1  # 1‑based para coincidir con reglas

        st.success("Clustering completado ✅")
        st.dataframe(df_cluster.head(1000))

        # -------- Comparación con clústeres por reglas -----------------
        if rule_col in df_cluster:
            if df_cluster[rule_col].isna().all():
                st.warning("La columna de referencia basada en reglas contiene sólo NaNs; no es posible comparar.")
            else:
                st.subheader("Comparación ML vs Reglas")
                compare_df = df_cluster[[rule_col, "cluster_ml"]].dropna()
                conf_matrix = pd.crosstab(compare_df[rule_col], compare_df["cluster_ml"], rownames=["Reglas"], colnames=["ML"])
                ari = adjusted_rand_score(compare_df[rule_col], compare_df["cluster_ml"])

                st.markdown("**Matriz de confusión**")
                st.dataframe(conf_matrix)
                st.markdown(f"**Adjusted Rand Index (ARI):** `{ari:.3f}`")
        else:
            st.warning("No se encontró la columna de clústeres por reglas en este nivel de agregación.")

        # -------- Estadísticas descriptivas ----------------------------
        st.subheader("Estadísticas descriptivas por clúster y variedad")
        by_cluster = (
            df_cluster.groupby(["cluster_ml", "Variedad" if "Variedad" in df_cluster else rule_col])[feats]
            .agg(["mean", "std", "count"])
            .round(2)
        )
        st.dataframe(by_cluster)

        # -------- Descarga --------------------------------------------
        st.download_button(
            "Descargar tabla clusterizada (CSV)",
            data=df_cluster.to_csv(index=False).encode(),
            file_name="clusters_carozos.csv",
            mime="text/csv",
        )
