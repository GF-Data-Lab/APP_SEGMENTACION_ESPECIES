# -*- coding: utf-8 -*-
"""
Procesamiento de la hoja «Carozos» del archivo Excel "MAESTRO CAROZOS FINAL COMPLETO CG.xlsx".

Versión 7 – 25-jun-2025
--------------------------------------------------
Novedades claves
----------------
1. **Dos sub-tipos de cherry plum** («small» ≤ 45 g y «mid» 46-60 g) con reglas
   independientes y fácilmente editables.
2. **Nuevos umbrales Candy / Cherry** (Brix, Firmeza, Acidez, Productividad)
   exactamente como en el flujograma.
3. **Color de pulpa** («Amarilla» ∣ «Blanca») para nectarines
   + reglas específicas por periodo de cosecha (muy temprana, temprana, tardía).
4. **Firmeza punto débil**  
   = *valor mínimo más frecuente* entre Quilla, Hombro, Mejilla 1 y 2,
   calculado en el **promedio del grupo Variedad + Fruto**.
5. **Relleno de nulos**: si una muestra carece de X, se toma el valor de la
   **muestra 1** del mismo grupo.
6. **Clúster doble**  
   - `cluster_row`  → cada registro individual  
   - `cluster_grp`  → promedio del grupo Variedad + Fruto
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union, Sequence
from collections.abc import Iterable
import streamlit as st
import io
from utils import show_logo



st.set_page_config(
    page_title="Segmentaciones",
    page_icon="G.png",
    layout="wide"
)
# --------------------------------------------------
# Configuración general
# --------------------------------------------------
st.markdown(
    """
    <style>
      /* Sólo los botones dentro del sidebar */
      [data-testid="stSidebar"] div.stButton > button {
        background-color: #D32F2F !important;  /* rojo fuerte */
        color: white !important;
        border: none !important;
      }
      [data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #B71C1C !important;  /* rojo más oscuro al pasar */
      }
    </style>
    """,
    unsafe_allow_html=True
)

FILE = Path(r"MAESTRO CAROZOS FINAL COMPLETO CG.xlsx")
SHEET_NAME = "CAROZOS"
USECOLS = "A:AP"
START_ROW = 2               # saltar filas informativas

ESPECIE_COLUMN = "Especie"
DATE_COLUMN = "Fecha cosecha"
COLOR_COLUMN = "Color de pulpa"           # solo existe en Nectarín
VAR_COLUMN   = "Variedad"                 # llave para agrupaciones
FRUTO_COLUMN = "Fruto (n°)"

ESPECIES_VALIDAS = {"Ciruela", "Nectarin"}

# ---------------------------------------------------------------------------
# Columnas físicas
# ---------------------------------------------------------------------------
WEIGHT_COLS = ("Peso (g)", "Calibre", "Peso")
COL_FIRMEZA_PUNTO    = ("Quilla", "Hombro")
COL_FIRMEZA_MEJILLAS = ("Mejilla 1", "Mejilla 2")
COL_FIRMEZA_ALL      = list(COL_FIRMEZA_PUNTO + COL_FIRMEZA_MEJILLAS)

COL_ORIG_BRIX = "Solidos solubles (%)"
COL_BRIX      = "BRIX"
COL_ACIDEZ    = "Acidez (%)"
#COL_PROD      = "Productividad (Ton)"

NUMERIC_COLS = (
    COL_FIRMEZA_ALL
    + [COL_BRIX, COL_ACIDEZ]#, COL_PROD
    + [c for c in WEIGHT_COLS]
)

# ---------------------------------------------------------------------------
# Tabla de reglas ― Ciruela --------------------------------------------------
#   Cada lista: [(mín, máx, grupo), …]
#   Límite inf. incluido, sup. excluido  → [mín, máx)
# ---------------------------------------------------------------------------
PLUM_RULES: Dict[str, Dict[str, List[Tuple[float, float, int]]]] = {
    "candy": {
        COL_BRIX:      [(18.0,  np.inf, 1), (16.0, 18.0, 2), (14.0, 16.0, 3), (-np.inf, 14.0, 4)],
        "FIRMEZA_PUNTO": [(7.0,  np.inf, 1), (5.0,  7.0, 2), (4.0,  5.0, 3), (-np.inf, 4.0, 4)],
        "FIRMEZA_MEJ":   [(9.0,  np.inf, 1), (7.0,  9.0, 2), (6.0,  7.0, 3), (-np.inf, 6.0, 4)],
        COL_ACIDEZ:    [(-np.inf, 0.60, 1), (0.60, 0.81, 2), (0.81, 1.00, 3), (1.00, np.inf, 4)],
    },
    "cherry": {
        COL_BRIX:      [(21.0, np.inf, 1), (18.0, 21.0, 2), (15.0, 18.0, 3), (-np.inf, 15.0, 4)],
        "FIRMEZA_PUNTO": [(6.0, np.inf, 1), (4.5,  6.0, 2), (3.0,  4.5, 3), (-np.inf, 3.0, 4)],
        "FIRMEZA_MEJ":   [(8.0, np.inf, 1), (5.0,  8.0, 2), (4.0,  5.0, 3), (-np.inf, 4.0, 4)],
        COL_ACIDEZ:    [(-np.inf, 0.60, 1), (0.60, 0.81, 2), (0.81, 1.00, 3), (1.00, np.inf, 4)],
    },
}

# ---------------------------------------------------------------------------
# Tabla de reglas ― Nectarin ------------------------------------------------
#   Se desdobla por color de pulpa y periodo de cosecha
# ---------------------------------------------------------------------------
def _mk_nec_rules(
    brix1: float, brix2: float, brix3: float,
    mej_1: float, mej_2: float,
) -> Dict[str, List[Tuple[float, float, int]]]:
    """Helper: genera tabla estándar Nectarín."""
    return {
        COL_BRIX: [(brix1, np.inf, 1), (brix2, brix1, 2), (brix3, brix2, 3), (-np.inf, brix3, 4)],
        "FIRMEZA_PUNTO": [(9.0, np.inf, 1), (8.0, 9.0, 2), (7.0, 8.0, 3), (-np.inf, 7.0, 4)],
        "FIRMEZA_MEJ": [(mej_1, np.inf, 1), (mej_2, mej_1, 2), (9.0, mej_2, 3), (-np.inf, 9.0, 4)],
        COL_ACIDEZ: [(-np.inf, 0.60, 1), (0.60, 0.81, 2), (0.81, 1.00, 3), (1.00, np.inf, 4)],
    }

NECT_RULES: Dict[str, Dict[str, Dict[str, List[Tuple[float, float, int]]]]] = {
    "amarilla": {
        "muy_temprana": _mk_nec_rules(13.0, 10.0, 9.0, 14.0, 12.0),
        "temprana":      _mk_nec_rules(13.0, 10.0, 9.0, 14.0, 12.0),
        "tardia":        _mk_nec_rules(14.0, 12.0, 10.0, 14.0, 12.0),
    },
    "blanca": {
        "muy_temprana": _mk_nec_rules(13.0, 10.0, 9.0, 13.0, 11.0),
        "temprana":      _mk_nec_rules(13.0, 10.0, 9.0, 13.0, 11.0),
        "tardia":        _mk_nec_rules(14.0, 12.0, 10.0, 13.0, 11.0),
    },
}

PERIOD_MAP = {
    "muy_temprana": "muy_temprana",
    "temprana": "temprana",
    "media": "temprana",
    "tardia": "tardia",
    "sin_fecha": "temprana",
}

# --------------------------------------------------
# Helpers fecha
# --------------------------------------------------
def _harvest_period(ts: pd.Timestamp | float | str) -> str:
    ts = pd.to_datetime(ts, errors="coerce")
    if pd.isna(ts):
        return "sin_fecha"
    m, d = ts.month, ts.day
    if (m, d) < (11, 25):
        return "muy_temprana"
    if (11, 25) <= (m, d) <= (12, 15):
        return "temprana"
    if (12, 16) <= (m, d) <= (2, 15):
        return("media")
    return "tardia"

# --------------------------------------------------
# Conversions & fills
# --------------------------------------------------
def _to_numeric(df: pd.DataFrame, cols: Sequence[str]) -> None:
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace("−", "-", regex=False)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

def _first_sample_fill(group: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Rellena NaNs del grupo con el valor de la primera muestra."""
    first = group.iloc[0]
    flat_cols: list[str] = []
    for c in cols:
        if isinstance(c, Iterable) and not isinstance(c, (str, bytes)):
            flat_cols.extend(c)
        else:
            flat_cols.append(c)
    for col in flat_cols:
        if col in group.columns:
            group[col] = group[col].fillna(first[col])
    return group

def _weight_value(row: pd.Series) -> float | None:
    for c in WEIGHT_COLS:
        if c in row and pd.notna(row[c]):
            try:
                return float(str(row[c]).replace(",", "."))
            except ValueError:
                continue
    return None

# --------------------------------------------------
# Detección de tipo de ciruela
# --------------------------------------------------
def _plum_subtype(row: pd.Series) -> str:
    if row[ESPECIE_COLUMN] != "Ciruela":
        return "non_plum"
    peso = _weight_value(row)
    if peso is None:
        return "unknown"
    if peso > 60:
        return "candy"
    # cualquier Ciruela ≤ 60 g es cherry
    return "cherry"

# --------------------------------------------------
# Firmeza punto débil (mínimo más frecuente)
# --------------------------------------------------
def _fpd_from_group(grp: pd.DataFrame) -> float | None:
    mean_vals = grp[COL_FIRMEZA_ALL].mean()
    min_val = mean_vals.min()
    if pd.isna(min_val):
        return np.nan
    return float(min_val)

# --------------------------------------------------
# Clasificador genérico
# --------------------------------------------------
def _rule_key(col: str) -> str:
    if col in COL_FIRMEZA_PUNTO or col == "Firmeza punto débil":
        return "FIRMEZA_PUNTO"
    if col in COL_FIRMEZA_MEJILLAS:
        return "FIRMEZA_MEJ"
    return col

def _classify_value(val: float, rules: List[Tuple[float, float, int]]) -> float:
    if pd.isna(val) or not rules:
        return np.nan
    for lo, hi, grp in rules:
        if lo <= val < hi:
            return grp
    return np.nan

def _classify_row(row: pd.Series, col: str) -> float:
    key = _rule_key(col)
    if row[ESPECIE_COLUMN] == "Ciruela":
        rules = PLUM_RULES.get(row["plum_subtype"], {}).get(key, [])
        if rules:
            return _classify_value(row[col], rules)
    if row[ESPECIE_COLUMN] == "Nectarin":
        color = str(row[COLOR_COLUMN]).strip().lower() or "amarilla"
        color = "blanca" if color.startswith("blanc") else "amarilla"
        period = PERIOD_MAP[row["harvest_period"]]
        rules = NECT_RULES[color][period].get(key, [])
        return _classify_value(row[col], rules)
    return np.nan

# --------------------------------------------------
# Pipeline principal
# --------------------------------------------------
def process_carozos(file: Union[str, Path] = FILE) -> pd.DataFrame:
    df = pd.read_excel(
        file, sheet_name=SHEET_NAME, usecols=USECOLS,
        skiprows=START_ROW, dtype=str
    )

    # 1) Filtros y renombres
    df = df[df[ESPECIE_COLUMN].isin(ESPECIES_VALIDAS)].copy()
    df.rename(columns={COL_ORIG_BRIX: COL_BRIX}, inplace=True)
    if COLOR_COLUMN not in df.columns:
        df[COLOR_COLUMN] = "Amarilla"

    # 2) Tipos y periodos
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df["harvest_period"] = df[DATE_COLUMN].apply(_harvest_period)
    df["plum_subtype"]   = df.apply(_plum_subtype, axis=1)

    # 3) Conversión a numérico
    _to_numeric(df, NUMERIC_COLS)

    # 3.1) Columna con la mejilla más débil
    df["firmezas mejillas"] = df[["Mejilla 1", "Mejilla 2"]].min(axis=1)

    # 3.2) Clasificación de mejillas para cherry, candy y nectarines blancos
    # 3.2) Clasificación de mejillas para cherry, candy, nectarines blancos y amarillos
    conds = [
        # cherry plum
        (df["plum_subtype"] == "cherry") & (df["firmezas mejillas"] >= 6),
        (df["plum_subtype"] == "cherry") & (df["firmezas mejillas"] >= 5) & (df["firmezas mejillas"] < 6),
        (df["plum_subtype"] == "cherry") & (df["firmezas mejillas"] >= 4) & (df["firmezas mejillas"] < 5),
        (df["plum_subtype"] == "cherry") & (df["firmezas mejillas"] < 4),

        # candy plum
        (df["plum_subtype"] == "candy")  & (df["firmezas mejillas"] >= 9),
        (df["plum_subtype"] == "candy")  & (df["firmezas mejillas"] >= 7) & (df["firmezas mejillas"] < 9),
        (df["plum_subtype"] == "candy")  & (df["firmezas mejillas"] >= 6) & (df["firmezas mejillas"] < 7),
        (df["plum_subtype"] == "candy")  & (df["firmezas mejillas"] < 6),

        # nectarin de pulpa blanca
        (df[ESPECIE_COLUMN] == "Nectarin")
          & (df[COLOR_COLUMN].str.strip().str.lower().str.startswith("blanc"))
          & (df["firmezas mejillas"] >= 13),
        (df[ESPECIE_COLUMN] == "Nectarin")
          & (df[COLOR_COLUMN].str.strip().str.lower().str.startswith("blanc"))
          & (df["firmezas mejillas"] >= 11) & (df["firmezas mejillas"] < 13),
        (df[ESPECIE_COLUMN] == "Nectarin")
          & (df[COLOR_COLUMN].str.strip().str.lower().str.startswith("blanc"))
          & (df["firmezas mejillas"] >=  9) & (df["firmezas mejillas"] < 11),
        (df[ESPECIE_COLUMN] == "Nectarin")
          & (df[COLOR_COLUMN].str.strip().str.lower().str.startswith("blanc"))
          & (df["firmezas mejillas"] <  9),

        # nectarin de pulpa amarilla
        (df[ESPECIE_COLUMN] == "Nectarin")
          & (df[COLOR_COLUMN].str.strip().str.lower().str.startswith("amarilla"))
          & (df["firmezas mejillas"] >= 14),
        (df[ESPECIE_COLUMN] == "Nectarin")
          & (df[COLOR_COLUMN].str.strip().str.lower().str.startswith("amarilla"))
          & (df["firmezas mejillas"] >= 12) & (df["firmezas mejillas"] < 14),
        (df[ESPECIE_COLUMN] == "Nectarin")
          & (df[COLOR_COLUMN].str.strip().str.lower().str.startswith("amarilla"))
          & (df["firmezas mejillas"] >=  9) & (df["firmezas mejillas"] < 12),
        (df[ESPECIE_COLUMN] == "Nectarin")
          & (df[COLOR_COLUMN].str.strip().str.lower().str.startswith("amarilla"))
          & (df["firmezas mejillas"] <  9),
    ]
    choices = [1, 2, 3, 4] * 4  # un bloque [1,2,3,4] para cada uno de los 4 sub-tipo
    df["grp_firmezas_mejillas"] = np.select(conds, choices, default=np.nan)



    # 4) Cálculo de Firmeza punto débil (mínimo absoluto)
    df["Firmeza punto débil"] = df[COL_FIRMEZA_ALL].min(axis=1)

    # 5) Relleno de nulos por primera muestra
    grp_keys = [VAR_COLUMN, FRUTO_COLUMN]
    df = (
        df.groupby(grp_keys, dropna=False, group_keys=False)
          .apply(
              _first_sample_fill,
              NUMERIC_COLS + ["Firmeza punto débil", "firmezas mejillas", "grp_firmezas_mejillas"]
          )
    )

    # 6) Clasificación de grupos
    cols_to_classify = ["Firmeza punto débil", COL_BRIX, COL_ACIDEZ]
    for col in cols_to_classify:
        out = f"grp_{col.replace(' ', '_')}"
        df[out] = df.apply(lambda r, c=col: _classify_row(r, c), axis=1)

    # 7) Cluster individual
    grp_cols = [c for c in df.columns if c.startswith("grp_")]
    df["cond_sum"] = df[grp_cols].sum(axis=1, min_count=1)
    if df["cond_sum"].notna().nunique() >= 4:
        df["cluster_row"] = pd.qcut(df["cond_sum"], 4, labels=[1,2,3,4])
    else:
        df["cluster_row"] = pd.cut(df["cond_sum"], 4, labels=[1,2,3,4])

    # 8) Cluster grupal (promedio)
    grp_cond = (
        df.groupby(grp_keys, dropna=False)["cond_sum"]
          .mean()
          .rename("cond_sum_grp")
          .reset_index()
    )
    df = df.merge(grp_cond, on=grp_keys, how="left")
    if grp_cond["cond_sum_grp"].notna().nunique() >= 4:
        bins = pd.qcut(grp_cond["cond_sum_grp"], 4, labels=[1,2,3,4])
    else:
        bins = pd.cut(grp_cond["cond_sum_grp"], 4, labels=[1,2,3,4])
    grp_cond["cluster_grp"] = bins
    df = df.merge(
        grp_cond[grp_keys + ["cluster_grp"]], on=grp_keys, how="left"
    )

    return df

# ------------------------------ STREAMLIT UI ------------------------------

@st.cache_data
def process_file(uploaded_file) -> Union[pd.DataFrame, None]:
    try:
        return process_carozos(uploaded_file)
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        return None

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
            st.switch_page('pages/analisis.py')
generarMenu()

st.title("🛠️ Segmentación por Especies")
st.write(
    "Sube tu archivo Excel 'MAESTRO CAROZOS FINAL COMPLETO CG.xlsx' y obtén los clusters, clasificaciones y resultados procesados según el flujograma."
)

uploaded = st.file_uploader(
    "Selecciona tu archivo Excel",
    type=["xls", "xlsx"],
    help="Debes incluir la hoja 'CAROZOS' con las columnas A:AP."
)

if uploaded:
    df = process_file(uploaded)
    if df is not None:
        st.success("¡Procesamiento completado con éxito! 🎉")
        st.data_editor(
        df,
        use_container_width=True,  # ocupa todo el ancho posible
        height=600                  # ajusta la altura a tu gusto
    )

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Carozos')
    buf.seek(0)
    st.download_button(
        label="📥 Descargar resultados como Excel",
        data=buf.getvalue(),
        file_name="carozos_procesados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.session_state["df_seg_especies"] = df
else:
    st.info("Esperando que subas un archivo para procesar...")
