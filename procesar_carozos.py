# -*- coding: utf-8 -*-
"""
Procesamiento de la hoja «Carozos» del archivo Excel "MAESTRO CAROZOS FINAL COMPLETO CG.xlsx".

Versión 7 – 25‑jun‑2025
--------------------------------------------------
Novedades claves
----------------
1. **Dos sub‑tipos de cherry plum** («small» ≤ 45 g y «mid» 46‑60 g) con reglas
   independientes y fácilmente editables.
2. **Nuevos umbrales Candy / Cherry** (Brix, Firmeza, Acidez, Productividad)
   exactamente como en el flujograma.
3. **Color de pulpa** («Amarilla» ∣ «Blanca») para nectarines
   + reglas específicas por periodo de cosecha (muy temprana, temprana, tardía).
4. **Firmeza punto débil**  
   = *valor mínimo más frecuente* entre Quilla, Hombro, Mejilla 1 y 2,
   calculado en el **promedio del grupo Variedad + Fruto**.
5. **Relleno de nulos**: si una muestra carece de X, se toma el valor de la
   **muestra 1** del mismo grupo.
6. **Clúster doble**  
   - `cluster_row`  → cada registro individual  
   - `cluster_grp`  → promedio del grupo Variedad + Fruto
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union, Sequence, Mapping

# --------------------------------------------------
# Configuración general
# --------------------------------------------------
FILE = Path(
    r"C:\Users\gonzalo.rojas\OneDrive - GARCES FRUIT\Escritorio\LAB DATOS GF"
    r"\___17__segementación especies\files\MAESTRO CAROZOS FINAL COMPLETO CG.xlsx"
)
SHEET_NAME = "CAROZOS"
USECOLS = "A:AP"
START_ROW = 2               # saltar filas informativas

ESPECIE_COLUMN = "Especie"
DATE_COLUMN = "Fecha cosecha"
COLOR_COLUMN = "Color de pulpa"           # solo existe en Nectarin

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
    + [COL_BRIX, COL_ACIDEZ]#, COL_PROD]
    + [c for c in WEIGHT_COLS]
)

# ---------------------------------------------------------------------------
# Tabla de reglas ― Ciruela --------------------------------------------------
#   Cada lista: [(mín, máx, grupo), …]
#   Límite inf. incluido, sup. excluido  → [mín, máx)
# ---------------------------------------------------------------------------
PLUM_RULES: Dict[str, Dict[str, List[Tuple[float, float, int]]]] = {
    # CANDY PLUM  Calibre > 60 g
    "candy": {
        COL_BRIX:      [(18.0,  np.inf, 1), (16.0, 18.0, 2), (14.0, 16.0, 3), (-np.inf, 14.0, 4)],
        "FIRMEZA_PUNTO": [(7.0,  np.inf, 1), (5.0,  7.0, 2), (4.0,  5.0, 3), (-np.inf, 4.0, 4)],
        "FIRMEZA_MEJ": [(9.0,  np.inf, 1), (7.0,  9.0, 2), (6.0,  7.0, 3), (-np.inf, 6.0, 4)],
        COL_ACIDEZ:    [(-np.inf, 0.60, 1), (0.60, 0.81, 2), (0.81, 1.00, 3), (1.00, np.inf, 4)]#,
       # COL_PROD:      [(40, np.inf, 1), (35, 40, 2), (25, 35, 3), (-np.inf, 25, 4)],
    },
    # CHERRY PLUM mid (46‑60 g)
    "cherry_mid": {
        COL_BRIX:      [(21.0, np.inf, 1), (18.0, 21.0, 2), (15.0, 18.0, 3), (-np.inf, 15.0, 4)],
        "FIRMEZA_PUNTO": [(6.0, np.inf, 1), (4.5,  6.0, 2), (3.0,  4.5, 3), (-np.inf, 3.0, 4)],
        "FIRMEZA_MEJ": [(8.0, np.inf, 1), (5.0,  8.0, 2), (4.0,  5.0, 3), (-np.inf, 4.0, 4)],
        COL_ACIDEZ:    [(-np.inf, 0.60, 1), (0.60, 0.81, 2), (0.81, 1.00, 3), (1.00, np.inf, 4)]#,
       # COL_PROD:      [(30, np.inf, 1), (20, 30, 2), (15, 20, 3), (-np.inf, 15, 4)],
    },
    # CHERRY PLUM small (≤ 45 g)  – por ahora mismo set que ‘mid’; cámbialo si
    #   el comité técnico define otros valores
    "cherry_small": {},   # se hereda dinámicamente más abajo
}

# hereda reglas mid si no se redefine
PLUM_RULES["cherry_small"] = PLUM_RULES["cherry_mid"].copy()

# ---------------------------------------------------------------------------
# Tabla de reglas ― Nectarin ------------------------------------------------
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
    # Color ↦ Periodo ↦ Reglas
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

# Periodos aceptados en Nectarín; si cae entre 16‑dic y 15‑feb lo normalizamos a
# «media» (se comporta como «temprana»)
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
        return "media"
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
            .str.replace("\u2212", "-", regex=False)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

def _first_sample_fill(group: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Rellena NaNs del grupo con el valor de la primera muestra."""
    first = group.iloc[0]
    for c in cols:
        if c in group:
            group[c] = group[c].fillna(first[c])
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
    """candy / cherry_mid / cherry_small / unknown / non_plum"""
    if row[ESPECIE_COLUMN] != "Ciruela":
        return "non_plum"
    peso = _weight_value(row)
    if peso is None:
        return "unknown"
    if peso > 60:
        return "candy"
    if peso > 45:
        return "cherry_mid"
    return "cherry_small"

# --------------------------------------------------
# Firmeza punto débil (mínimo más frecuente)
# --------------------------------------------------
def _fpd_from_group(grp: pd.DataFrame) -> float | None:
    mean_vals = grp[COL_FIRMEZA_ALL].mean()
    # frecuencia del valor mínimo
    min_val = mean_vals.min()
    if pd.isna(min_val):
        return np.nan
    # romper empates tomando el primer campo izquierda‑derecha
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
    # CIRUELA ---------------------------------------------------------------
    if row[ESPECIE_COLUMN] == "Ciruela":
        rules = PLUM_RULES.get(row["plum_subtype"], {}).get(key, [])
        if rules:
            return _classify_value(row[col], rules)
    # NECTARIN --------------------------------------------------------------
    if row[ESPECIE_COLUMN] == "Nectarin":
        color = str(row[COLOR_COLUMN]).strip().lower() or "amarilla"
        color = "blanca" if color.startswith("blanc") else "amarilla"
        period = PERIOD_MAP[row["harvest_period"]]
        rules = NECT_RULES[color][period].get(key, [])
        return _classify_value(row[col], rules)
    # Fallback
    return np.nan

# --------------------------------------------------
# Pipeline principal
# --------------------------------------------------
def process_carozos(file: Union[str, Path] = FILE) -> pd.DataFrame:
    df = pd.read_excel(
        file, sheet_name=SHEET_NAME, usecols=USECOLS,
        skiprows=START_ROW, dtype=str
    )

    # ------------ filtros & renombres -------------------------------------
    df = df[df[ESPECIE_COLUMN].isin(ESPECIES_VALIDAS)].copy()
    df.rename(columns={COL_ORIG_BRIX: COL_BRIX}, inplace=True)
    if COLOR_COLUMN not in df.columns:
        df[COLOR_COLUMN] = "Amarilla"

    # ------------ tipos y periodos ----------------------------------------
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df["harvest_period"] = df[DATE_COLUMN].apply(_harvest_period)
    df["plum_subtype"] = df.apply(_plum_subtype, axis=1)

    # ------------ numéricos ------------------------------------------------
    _to_numeric(df, NUMERIC_COLS)

    # ------------ agrupación Variedad + Fruto -----------------------------
    grp_keys = [VAR_COLUMN, FRUTO_COLUMN]
    # Firmeza punto débil grupal
    grp_fpd = (
        df.groupby(grp_keys, dropna=False)
          .apply(_fpd_from_group)
          .rename("Firmeza punto débil")
          .reset_index()
    )
    df = df.merge(grp_fpd, on=grp_keys, how="left")

    # Relleno de nulos por primera muestra
    df = (
        df.groupby(grp_keys, dropna=False, group_keys=False)
          .apply(_first_sample_fill, NUMERIC_COLS + ["Firmeza punto débil"])
    )

    # ------------ clasificación -------------------------------------------
    cols_to_classify = ["Firmeza punto débil"] + [COL_BRIX, COL_ACIDEZ]#, COL_PROD]
    for col in cols_to_classify:
        out = f"grp_{col.replace(' ', '_')}"
        df[out] = df.apply(lambda r, c=col: _classify_row(r, c), axis=1)

    # ------------ cluster individual --------------------------------------
    grp_cols = [c for c in df.columns if c.startswith("grp_")]
    df["cond_sum"] = df[grp_cols].sum(axis=1, min_count=1)
    if df["cond_sum"].notna().nunique() >= 4:
        df["cluster_row"] = pd.qcut(df["cond_sum"], 4, labels=[1, 2, 3, 4])
    else:
        df["cluster_row"] = pd.cut(df["cond_sum"], 4, labels=[1, 2, 3, 4])

    # ------------ cluster grupal (promedio) -------------------------------
    grp_cond = (
        df.groupby(grp_keys, dropna=False)["cond_sum"]
          .mean()
          .rename("cond_sum_grp")
          .reset_index()
    )
    df = df.merge(grp_cond, on=grp_keys, how="left")

    if grp_cond["cond_sum_grp"].notna().nunique() >= 4:
        bins = pd.qcut(grp_cond["cond_sum_grp"], 4, labels=[1, 2, 3, 4])
    else:
        bins = pd.cut(grp_cond["cond_sum_grp"], 4, labels=[1, 2, 3, 4])
    grp_cond["cluster_grp"] = bins

    df = df.merge(grp_cond[grp_keys + ["cluster_grp"]], on=grp_keys, how="left")

    return df

# ------------------------------ CLI ----------------------------------------
if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    df = process_carozos()
    print(process_carozos().head())
