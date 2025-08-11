"""
Módulo Streamlit mejorado para la segmentación de carozos.

Este módulo amplía la funcionalidad original permitiendo a los usuarios:

* Visualizar y editar las reglas de segmentación directamente en la interfaz.
* Definir valores por defecto para atributos ausentes (peso, color de pulpa, periodo de cosecha).
* Agregar nuevas bandas de reglas a través de un editor interactivo.
* Visualizar los resultados no sólo en un Excel descargable, sino también filtrados por variedad.

Para utilizarlo copia el contenido de este archivo en la página correspondiente
(`pages/Segmentacion_especies.py`) de tu proyecto Streamlit.  Se ha
organizado todo el código en una única función para facilitar su
integración.
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


def segmentacion_app():
    """Construye la página de segmentación de especies con mejoras interactivas."""
    # -------------------------------------------------------------------------
    # Configuración general de la página
    # -------------------------------------------------------------------------
    st.set_page_config(
        page_title="Segmentaciones",
        page_icon="G.png",
        layout="wide"
    )
    # Estilos para los botones del sidebar
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

    # -------------------------------------------------------------------------
    # Parámetros y constantes
    # -------------------------------------------------------------------------
    FILE = Path(r"MAESTRO CAROZOS FINAL COMPLETO CG.xlsx")
    SHEET_NAME = "CAROZOS"
    USECOLS = "A:AP"
    START_ROW = 2

    ESPECIE_COLUMN = "Especie"
    DATE_COLUMN = "Fecha evaluación"
    COLOR_COLUMN = "Color de pulpa"
    VAR_COLUMN = "Variedad"
    FRUTO_COLUMN = "Fruto (n°)"

    ESPECIES_VALIDAS = {"Ciruela", "Nectarin"}

    WEIGHT_COLS = ("Peso (g)", "Calibre", "Peso")
    COL_FIRMEZA_PUNTO = ("Quilla", "Hombro")
    COL_FIRMEZA_MEJILLAS = ("Mejilla 1", "Mejilla 2")
    COL_FIRMEZA_ALL = list(COL_FIRMEZA_PUNTO + COL_FIRMEZA_MEJILLAS)

    COL_ORIG_BRIX = "Solidos solubles (%)"
    COL_BRIX = "BRIX"
    COL_ACIDEZ = "Acidez (%)"

    NUMERIC_COLS = (
        COL_FIRMEZA_ALL
        + [COL_BRIX, COL_ACIDEZ]
        + [c for c in WEIGHT_COLS]
    )

    # -----------------------------------------------------------------------
    # Reglas originales (copiadas de la versión base).  Estas reglas se
    # convertirán en dataframes editables para que el usuario pueda
    # modificarlas en tiempo real.
    # -----------------------------------------------------------------------
    DEFAULT_PLUM_RULES: Dict[str, Dict[str, List[Tuple[float, float, int]]]] = {
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

    DEFAULT_NECT_RULES: Dict[str, Dict[str, Dict[str, List[Tuple[float, float, int]]]]] = {
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
        "sin_fecha": "tardia",
    }

    # -----------------------------------------------------------------------
    # Utilities for converting rules to/from DataFrames
    # -----------------------------------------------------------------------
    def plum_rules_to_df(rules: Dict[str, Dict[str, List[Tuple[float, float, int]]]]) -> pd.DataFrame:
        """Flatten PLUM_RULES into a dataframe for editing."""
        rows = []
        for subtype, metrics in rules.items():
            for metric, bands in metrics.items():
                for lo, hi, group in bands:
                    rows.append({
                        "subtype": subtype,
                        "metric": metric,
                        "min": lo,
                        "max": hi,
                        "group": group,
                    })
        df = pd.DataFrame(rows)
        # Añadir columna apply para permitir activar/desactivar reglas
        df["apply"] = True
        return df

    def df_to_plum_rules(df: pd.DataFrame) -> Dict[str, Dict[str, List[Tuple[float, float, int]]]]:
        """Build PLUM_RULES dict from an edited dataframe."""
        out: Dict[str, Dict[str, List[Tuple[float, float, int]]]] = {}
        # Filtrar filas válidas: apply=True y campos obligatorios no nulos
        df_valid = df[(df.get("apply", True)) & df["subtype"].notna() & df["metric"].notna()]
        for subtype in df_valid["subtype"].unique():
            sub = df_valid[df_valid["subtype"] == subtype]
            out[subtype] = {}
            for metric in sub["metric"].unique():
                mdf = sub[sub["metric"] == metric]
                # sort by min descending so highest band appears first
                mdf = mdf.sort_values(by=["min"], ascending=False)
                bands: List[Tuple[float, float, int]] = []
                for _, r in mdf.iterrows():
                    # saltar filas con campos nulos
                    if pd.isna(r["min"]) or pd.isna(r["max"]) or pd.isna(r["group"]):
                        continue
                    bands.append((float(r["min"]), float(r["max"]), int(r["group"])) )
                if bands:
                    out[subtype][metric] = bands
        return out

    def nect_rules_to_df(rules: Dict[str, Dict[str, Dict[str, List[Tuple[float, float, int]]]]]) -> pd.DataFrame:
        """Flatten NECT_RULES into a dataframe for editing."""
        rows = []
        for color, periods in rules.items():
            for period, metrics in periods.items():
                for metric, bands in metrics.items():
                    for lo, hi, group in bands:
                        rows.append({
                            "color": color,
                            "period": period,
                            "metric": metric,
                            "min": lo,
                            "max": hi,
                            "group": group,
                        })
        df = pd.DataFrame(rows)
        df["apply"] = True
        return df

    def df_to_nect_rules(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, List[Tuple[float, float, int]]]]]:
        """Build NECT_RULES dict from an edited dataframe."""
        out: Dict[str, Dict[str, Dict[str, List[Tuple[float, float, int]]]]] = {}
        df_valid = df[(df.get("apply", True)) & df["color"].notna() & df["period"].notna() & df["metric"].notna()]
        for color in df_valid["color"].unique():
            cf = df_valid[df_valid["color"] == color]
            out[color] = {}
            for period in cf["period"].unique():
                pf = cf[cf["period"] == period]
                out[color][period] = {}
                for metric in pf["metric"].unique():
                    mf = pf[pf["metric"] == metric]
                    mf = mf.sort_values(by=["min"], ascending=False)
                    bands: List[Tuple[float, float, int]] = []
                    for _, r in mf.iterrows():
                        if pd.isna(r["min"]) or pd.isna(r["max"]) or pd.isna(r["group"]):
                            continue
                        bands.append((float(r["min"]), float(r["max"]), int(r["group"])) )
                    if bands:
                        out[color][period][metric] = bands
        return out

    # -----------------------------------------------------------------------
    # Procesamiento de datos (igual que en la versión base excepto por el uso
    # de reglas editables y valores por defecto configurables)
    # -----------------------------------------------------------------------
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

    # función para detectar sub-tipo de ciruela con valores por defecto
    def _plum_subtype(row: pd.Series) -> str:
        # Si no es ciruela retorna identificador nulo
        if row[ESPECIE_COLUMN] != "Ciruela":
            return "non_plum"
        peso = _weight_value(row)
        if peso is None:
            # Si el peso falta utilizamos el valor por defecto
            return st.session_state.get("default_plum_subtype", "cherry")
        # Lógica de clasificación configurable por el usuario
        cherry_cut = st.session_state.get("cherry_upper", 60.0)
        if peso > cherry_cut:
            return "candy"
        # Cualquier ciruela con peso ≤ cherry_cut se considera cherry
        return "cherry"

    def _fpd_from_group(grp: pd.DataFrame) -> float | None:
        mean_vals = grp[COL_FIRMEZA_ALL].mean()
        min_val = mean_vals.min()
        if pd.isna(min_val):
            return np.nan
        return float(min_val)

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

    def _classify_row(row: pd.Series, col: str, rules_plum: Dict, rules_nect: Dict) -> float:
        key = _rule_key(col)
        if row[ESPECIE_COLUMN] == "Ciruela":
            subtype = row.get("plum_subtype", "cherry")
            rule_dict = rules_plum.get(subtype, {}).get(key, [])
            return _classify_value(row[col], rule_dict)
        if row[ESPECIE_COLUMN] == "Nectarin":
            color = str(row.get(COLOR_COLUMN, "")).strip().lower() or st.session_state.get("default_color", "amarilla")
            color = "blanca" if color.startswith("blanc") else "amarilla"
            period = PERIOD_MAP.get(row.get("harvest_period", "sin_fecha"), "tardia")
            rule_dict = rules_nect.get(color, {}).get(period, {}).get(key, [])
            return _classify_value(row[col], rule_dict)
        return np.nan

    def _harvest_period_a(ts: pd.Timestamp | float | str) -> str:
        ts = pd.to_datetime(ts, errors="coerce")
        if pd.isna(ts):
            return st.session_state.get("default_period", "sin_fecha")
        m, d = ts.month, ts.day
        if (m, d) < (11, 22):
            return "muy_temprana"
        if (11, 22) <= (m, d) <= (12, 22):
            return "temprana"
        if (12, 23) <= (m, d) <= (2, 15):
            return "tardia"
        return "tardia"

    def _harvest_period_b(ts: pd.Timestamp | float | str) -> str:
        ts = pd.to_datetime(ts, errors="coerce")
        if pd.isna(ts):
            return st.session_state.get("default_period", "sin_fecha")
        m, d = ts.month, ts.day
        if (m, d) < (11, 25):
            return "muy_temprana"
        if (11, 25) <= (m, d) <= (12, 15):
            return "temprana"
        if (12, 16) <= (m, d) <= (2, 15):
            return "tardia"
        return "tardia"

    def process_carozos(
        file: Union[str, Path, pd.DataFrame],
        rules_plum: Dict,
        rules_nect: Dict,
        cond_method: str = "suma",
        grp_method: str = "mean",
    ) -> pd.DataFrame:
        # Permitir que el parámetro file sea un DataFrame ya cargado
        if isinstance(file, pd.DataFrame):
            df = file.copy()
        else:
            df = pd.read_excel(
                file, sheet_name=SHEET_NAME, usecols=USECOLS,
                skiprows=START_ROW, dtype=str
            )
        # 1) Filtros y renombres
        df = df[df[ESPECIE_COLUMN].isin(ESPECIES_VALIDAS)].copy()
        df.rename(columns={COL_ORIG_BRIX: COL_BRIX}, inplace=True)
        # Color de pulpa: reemplazar nulos por valor por defecto
        default_color = st.session_state.get("default_color", "Amarilla")
        if COLOR_COLUMN not in df.columns:
            df[COLOR_COLUMN] = default_color
        df[COLOR_COLUMN] = df[COLOR_COLUMN].fillna(default_color)
        # 2) Tipos y periodos
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
        # Asignación de período y sub-tipo según especie
        df["plum_subtype"] = df.apply(_plum_subtype, axis=1)
        df["harvest_period"] = "sin_fecha"  # inicializar
        # para nectarines, determinar periodo según color
        idx_nectar = df[ESPECIE_COLUMN] == "Nectarin"
        # Para cada fila, usar función A o B según color
        for i in df[idx_nectar].index:
            color = str(df.at[i, COLOR_COLUMN]).strip().lower()
            if color.startswith("blanc"):
                df.at[i, "harvest_period"] = _harvest_period_b(df.at[i, DATE_COLUMN])
            else:
                df.at[i, "harvest_period"] = _harvest_period_a(df.at[i, DATE_COLUMN])
        # 3) Conversión a numérico
        _to_numeric(df, NUMERIC_COLS)
        # 3.1) Columna con la mejilla más débil
        df["firmezas mejillas"] = df[["Mejilla 1", "Mejilla 2"]].min(axis=1)
        # 3.2) Clasificación de mejillas rápida (igual que original)
        conds = [
            (df["plum_subtype"] == "cherry") & (df["firmezas mejillas"] >= 6),
            (df["plum_subtype"] == "cherry") & (df["firmezas mejillas"] >= 5) & (df["firmezas mejillas"] < 6),
            (df["plum_subtype"] == "cherry") & (df["firmezas mejillas"] >= 4) & (df["firmezas mejillas"] < 5),
            (df["plum_subtype"] == "cherry") & (df["firmezas mejillas"] < 4),
            (df["plum_subtype"] == "candy")  & (df["firmezas mejillas"] >= 9),
            (df["plum_subtype"] == "candy")  & (df["firmezas mejillas"] >= 7) & (df["firmezas mejillas"] < 9),
            (df["plum_subtype"] == "candy")  & (df["firmezas mejillas"] >= 6) & (df["firmezas mejillas"] < 7),
            (df["plum_subtype"] == "candy")  & (df["firmezas mejillas"] < 6),
            # Nectarin blanca
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
            # Nectarin amarilla
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
        choices = [1, 2, 3, 4] * 4
        df["grp_firmezas_mejillas"] = np.select(conds, choices, default=np.nan)
        # 4) Firmeza punto débil (mínimo absoluto)
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
            df[out] = df.apply(lambda r, c=col: _classify_row(r, c, rules_plum, rules_nect), axis=1)
        # 7) Cluster individual
        grp_cols = [c for c in df.columns if c.startswith("grp_")]
        # 7) Cluster individual: cálculo de cond_sum según el método elegido
        if cond_method == "media":
            df["cond_sum"] = df[grp_cols].mean(axis=1, skipna=True)
        else:  # suma
            df["cond_sum"] = df[grp_cols].sum(axis=1, min_count=1)
        # Discretizar cond_sum en 4 clusters
        if df["cond_sum"].notna().nunique() >= 4:
            df["cluster_row"] = pd.qcut(df["cond_sum"], 4, labels=[1,2,3,4])
        else:
            df["cluster_row"] = pd.cut(df["cond_sum"], 4, labels=[1,2,3,4])
        # 8) Cluster grupal (promedio)
        # 8) Cluster grupal (promedio o moda de cond_sum según grp_method)
        if grp_method == "mode":
            def _mode_series(s):
                m = s.mode()
                return m.iloc[0] if not m.empty else np.nan
            grp_cond = (
                df.groupby(grp_keys, dropna=False)["cond_sum"]
                  .agg(_mode_series)
                  .rename("cond_sum_grp")
                  .reset_index()
            )
        else:  # mean
            grp_cond = (
                df.groupby(grp_keys, dropna=False)["cond_sum"]
                  .mean()
                  .rename("cond_sum_grp")
                  .reset_index()
            )
        df = df.merge(grp_cond, on=grp_keys, how="left")
        # Discretización de cond_sum_grp en 4 clusters
        if grp_cond["cond_sum_grp"].notna().nunique() >= 4:
            bins = pd.qcut(grp_cond["cond_sum_grp"], 4, labels=[1,2,3,4])
        else:
            bins = pd.cut(grp_cond["cond_sum_grp"], 4, labels=[1,2,3,4])
        grp_cond["cluster_grp"] = bins
        df = df.merge(
            grp_cond[grp_keys + ["cluster_grp"]], on=grp_keys, how="left"
        )
        return df

    # -----------------------------------------------------------------------
    # Sidebar con menú
    # -----------------------------------------------------------------------
    def generar_menu():
        with st.sidebar:
            show_logo()
            if st.button('Página de Inicio 🏚️'):
                st.switch_page('app.py')
            if st.button('Segmentación de especies 🍑'):
                # no hacemos nada: estamos en esta página
                pass
            if st.button('Modelo de Clasificación'):
                st.switch_page('pages/Cluster_especies.py')
            if st.button('Análisis exploratorio'):
                st.switch_page('pages/analisis.py')

    generar_menu()

    st.title("🛠️ Segmentación por Especies (Mejorada)")
    st.write(
        """
        Sube tu archivo Excel `MAESTRO CAROZOS FINAL COMPLETO CG.xlsx` y obtén los clusters,
        clasificaciones y resultados procesados según el flujograma.  Además,
        puedes visualizar y editar las reglas de segmentación y definir valores
        por defecto para campos faltantes.
        """
    )

    # -----------------------------------------------------------------------
    # Valores por defecto configurables
    # -----------------------------------------------------------------------
    st.subheader("Valores por defecto")
    # Inicializar session_state con defaults si no existen
    if "default_plum_subtype" not in st.session_state:
        st.session_state["default_plum_subtype"] = "cherry"
    if "cherry_upper" not in st.session_state:
        st.session_state["cherry_upper"] = 60.0
    if "default_color" not in st.session_state:
        st.session_state["default_color"] = "Amarilla"
    if "default_period" not in st.session_state:
        st.session_state["default_period"] = "tardia"

    # Selector para tipo de ciruela por defecto cuando el peso es desconocido
    default_plum = st.selectbox(
        "Tipo de ciruela por defecto si el peso no está disponible",
        options=["cherry", "candy"],
        index=["cherry", "candy"].index(st.session_state["default_plum_subtype"]),
        key="default_plum_subtype"
    )
    # Sliders para umbrales de peso
    col1, col2 = st.columns(2)
    with col1:
        st.session_state["cherry_upper"] = st.number_input(
            "Peso máximo para cherry (g)",
            min_value=10.0,
            max_value=200.0,
            value=float(st.session_state["cherry_upper"]),
            step=1.0
        )
    with col2:
        st.session_state["default_color"] = st.selectbox(
            "Color de pulpa por defecto para Nectarín (si falta)",
            options=["Amarilla", "Blanca"],
            index=["Amarilla", "Blanca"].index(st.session_state["default_color"])
        )
    st.session_state["default_period"] = st.selectbox(
        "Periodo de cosecha por defecto para Nectarín (si falta fecha)",
        options=["muy_temprana", "temprana", "tardia", "sin_fecha"],
        index=["muy_temprana", "temprana", "tardia", "sin_fecha"].index(st.session_state["default_period"])
    )

    st.info(
        "Puedes cambiar estos valores y ver cómo afectan a las clasificaciones al recargar el archivo."
    )

    # -----------------------------------------------------------------------
    # Reglas editables
    # -----------------------------------------------------------------------
    st.subheader("Editar reglas de segmentación")
    # Inicializar reglas en session_state la primera vez
    if "plum_rules_df" not in st.session_state:
        st.session_state["plum_rules_df"] = plum_rules_to_df(DEFAULT_PLUM_RULES)
    if "nect_rules_df" not in st.session_state:
        st.session_state["nect_rules_df"] = nect_rules_to_df(DEFAULT_NECT_RULES)

    # Mostrar cada regla por separado para mayor claridad
    # Se eliminó la vista de lectura detallada para evitar ruido visual

    # Construir los diccionarios de reglas a partir de las tablas guardadas en session_state
    current_plum_rules = df_to_plum_rules(st.session_state["plum_rules_df"])
    current_nect_rules = df_to_nect_rules(st.session_state["nect_rules_df"])

    # -----------------------------------------------------------------------
    # Editor tabular para añadir/eliminar reglas y decidir si se aplican
    # -----------------------------------------------------------------------
    st.subheader("Editar y añadir reglas (vista tabla)")
    st.write("Aquí puedes modificar las reglas existentes o agregar nuevas filas. El campo `apply` permite activar o desactivar una regla sin borrarla.")
    colp, coln = st.columns(2)
    with colp:
        st.markdown("**Reglas de Ciruela**")
        edited_plum_df = st.data_editor(
            st.session_state["plum_rules_df"],
            use_container_width=True,
            height=300,
            num_rows="dynamic",
            key="plum_rules_table_editor"
        )
        if st.button("Guardar cambios en reglas de Ciruela"):
            # Actualizar DataFrame y diccionario
            st.session_state["plum_rules_df"] = edited_plum_df
            current_plum_rules = df_to_plum_rules(edited_plum_df)
            st.success("Reglas de Ciruela actualizadas.")
    with coln:
        st.markdown("**Reglas de Nectarín**")
        edited_nect_df = st.data_editor(
            st.session_state["nect_rules_df"],
            use_container_width=True,
            height=300,
            num_rows="dynamic",
            key="nect_rules_table_editor"
        )
        if st.button("Guardar cambios en reglas de Nectarín"):
            st.session_state["nect_rules_df"] = edited_nect_df
            current_nect_rules = df_to_nect_rules(edited_nect_df)
            st.success("Reglas de Nectarín actualizadas.")

    # -----------------------------------------------------------------------
    # Árbol de decisión para visualizar y editar reglas de forma más intuitiva
    # -----------------------------------------------------------------------
    st.subheader("Árbol de decisiones y editor de reglas")
    st.write(
        "Selecciona los parámetros en el siguiente orden (especie → tipo/color/periodo → métrica) para ver y modificar las bandas de cada regla con un código de colores similar al flujograma. Los cambios se reflejan automáticamente en la tabla de reglas."  
    )
    # Mapa de colores para grupos 1–4 inspirado en el diagrama
    group_colors = {
        1: '#a8e6cf',  # verde claro
        2: '#ffd3b6',  # naranja claro
        3: '#ffaaa5',  # coral
        4: '#ff8b94',  # rojo rosado
    }
    especie_seleccion = st.selectbox("Selecciona la especie", ["Ciruela", "Nectarin"])
    if especie_seleccion == "Ciruela":
        subtipo_sel = st.selectbox("Sub‑tipo de ciruela", list(current_plum_rules.keys()))
        metrica_sel = st.selectbox("Métrica", list(current_plum_rules[subtipo_sel].keys()))
        bandas = current_plum_rules[subtipo_sel][metrica_sel]
        # Mostrar bandas con colores
        bandas_df = pd.DataFrame(bandas, columns=["Min", "Max", "Grupo"])
        def _apply_colors_plum(row):
            return [f"background-color: {group_colors.get(int(row['Grupo']), '')}" for _ in row]
        st.write(bandas_df.style.apply(_apply_colors_plum, axis=1))
        # Editor de bandas
        st.markdown("**Editar bandas**")
        nuevas_bandas = []
        for i, (lo, hi, grp) in enumerate(bandas):
            cols = st.columns([2,2,1])
            # Asegurarse de que los valores infinitos no causen errores: reemplazar inf por un número grande
            import math
            lo_val = float(lo) if math.isfinite(lo) else -1e6
            hi_val = float(hi) if math.isfinite(hi) else 1e6
            lo_new = cols[0].number_input(
                f"Mín banda {i+1}",
                value=lo_val,
                key=f"plum_{subtipo_sel}_{metrica_sel}_min_{i}"
            )
            hi_new = cols[1].number_input(
                f"Máx banda {i+1}",
                value=hi_val,
                key=f"plum_{subtipo_sel}_{metrica_sel}_max_{i}"
            )
            grp_new = cols[2].selectbox(
                f"Grupo banda {i+1}",
                options=[1,2,3,4],
                index=int(grp)-1 if not math.isnan(grp) else 0,
                key=f"plum_{subtipo_sel}_{metrica_sel}_grp_{i}"
            )
            nuevas_bandas.append((lo_new, hi_new, grp_new))
        # Opción para añadir una nueva banda
        if st.button("Agregar banda", key=f"add_plum_{subtipo_sel}_{metrica_sel}"):
            # Añadir una banda con valores por defecto (continuando el último rango)
            last_hi = nuevas_bandas[-1][1] if nuevas_bandas else 0
            nuevas_bandas.append((last_hi, last_hi + 1, 4))
        # Guardar cambios
        if st.button("Guardar cambios de regla", key=f"save_plum_{subtipo_sel}_{metrica_sel}"):
            # Actualizar diccionario y DataFrame
            current_plum_rules[subtipo_sel][metrica_sel] = nuevas_bandas
            # Convertir a DataFrame y actualizar plum_rules_df en session_state
            st.session_state["plum_rules_df"] = plum_rules_to_df(current_plum_rules)
            st.success("Regla actualizada para ciruela.")
    else:  # Nectarín
        color_sel = st.selectbox("Color de pulpa", list(current_nect_rules.keys()))
        periodo_sel = st.selectbox("Periodo de cosecha", list(current_nect_rules[color_sel].keys()))
        metrica_sel_n = st.selectbox("Métrica", list(current_nect_rules[color_sel][periodo_sel].keys()))
        bandas_n = current_nect_rules[color_sel][periodo_sel][metrica_sel_n]
        # Mostrar bandas con colores
        bandas_df_n = pd.DataFrame(bandas_n, columns=["Min", "Max", "Grupo"])
        def _apply_colors_nect(row):
            return [f"background-color: {group_colors.get(int(row['Grupo']), '')}" for _ in row]
        st.write(bandas_df_n.style.apply(_apply_colors_nect, axis=1))
        # Editor de bandas
        st.markdown("**Editar bandas**")
        nuevas_bandas_n = []
        for i, (lo, hi, grp) in enumerate(bandas_n):
            cols = st.columns([2,2,1])
            import math
            lo_val = float(lo) if math.isfinite(lo) else -1e6
            hi_val = float(hi) if math.isfinite(hi) else 1e6
            lo_new = cols[0].number_input(
                f"Mín banda {i+1}",
                value=lo_val,
                key=f"nect_{color_sel}_{periodo_sel}_{metrica_sel_n}_min_{i}"
            )
            hi_new = cols[1].number_input(
                f"Máx banda {i+1}",
                value=hi_val,
                key=f"nect_{color_sel}_{periodo_sel}_{metrica_sel_n}_max_{i}"
            )
            grp_new = cols[2].selectbox(
                f"Grupo banda {i+1}",
                options=[1,2,3,4],
                index=int(grp)-1 if not math.isnan(grp) else 0,
                key=f"nect_{color_sel}_{periodo_sel}_{metrica_sel_n}_grp_{i}"
            )
            nuevas_bandas_n.append((lo_new, hi_new, grp_new))
        if st.button("Agregar banda", key=f"add_nect_{color_sel}_{periodo_sel}_{metrica_sel_n}"):
            last_hi = nuevas_bandas_n[-1][1] if nuevas_bandas_n else 0
            nuevas_bandas_n.append((last_hi, last_hi + 1, 4))
        if st.button("Guardar cambios de regla", key=f"save_nect_{color_sel}_{periodo_sel}_{metrica_sel_n}"):
            current_nect_rules[color_sel][periodo_sel][metrica_sel_n] = nuevas_bandas_n
            st.session_state["nect_rules_df"] = nect_rules_to_df(current_nect_rules)
            st.success("Regla actualizada para nectarin.")

    # -----------------------------------------------------------------------
    # Subida y procesamiento del archivo
    # -----------------------------------------------------------------------
    uploaded = st.file_uploader(
        "Selecciona tu archivo Excel",
        type=["xls", "xlsx"],
        help="Debes incluir la hoja 'CAROZOS' con las columnas A:AP."
    )
    if uploaded:
        # -------------------------------------------------------------------
        # Lectura del archivo y detección de outliers antes del procesamiento
        # -------------------------------------------------------------------
        try:
            if isinstance(uploaded, pd.DataFrame):
                df_upload = uploaded.copy()
            else:
                df_upload = pd.read_excel(uploaded, sheet_name=SHEET_NAME, usecols=USECOLS, skiprows=START_ROW, dtype=str)
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            df_upload = None
        if df_upload is not None:
            # Convertir columnas numéricas a numérico y calcular harvest_period para una detección más granular de outliers
            tmp_df = df_upload.copy()
            # Convertir fechas
            if DATE_COLUMN in tmp_df.columns:
                tmp_df[DATE_COLUMN] = pd.to_datetime(tmp_df[DATE_COLUMN], errors="coerce")
                # Asignar periodo para cada registro (utilizando funciones de cosecha)
                periods = []
                for idx, row in tmp_df.iterrows():
                    especie = row.get(ESPECIE_COLUMN)
                    color = str(row.get(COLOR_COLUMN, '')).strip().lower()
                    if especie == 'Nectarin':
                        if color.startswith('blanc'):
                            periods.append(_harvest_period_b(row[DATE_COLUMN]))
                        else:
                            periods.append(_harvest_period_a(row[DATE_COLUMN]))
                    else:
                        periods.append('sin_fecha')
                tmp_df['harvest_period'] = periods
            else:
                tmp_df['harvest_period'] = 'sin_fecha'
            _to_numeric(tmp_df, NUMERIC_COLS)
            # Detección de outliers por especie, variedad, muestra y periodo (|z| > 2)
            outlier_flags = pd.Series(False, index=tmp_df.index)
            outlier_cols = {col: pd.Series(False, index=tmp_df.index) for col in NUMERIC_COLS if col in tmp_df.columns}
            group_cols = [ESPECIE_COLUMN, VAR_COLUMN, FRUTO_COLUMN, 'harvest_period']
            for col in [c for c in NUMERIC_COLS if c in tmp_df.columns]:
                for _, group_df in tmp_df.groupby(group_cols, dropna=False):
                    serie = group_df[col].astype(float)
                    if serie.empty:
                        continue
                    m = serie.mean()
                    s = serie.std()
                    if s == 0 or pd.isna(s):
                        continue
                    z = (serie - m) / s
                    mask = (z.abs() > 2)
                    outlier_flags.loc[group_df.index] = outlier_flags.loc[group_df.index] | mask
                    outlier_cols[col].loc[group_df.index] = outlier_cols[col].loc[group_df.index] | mask
            tmp_df['Outlier'] = outlier_flags
            # Registrar para cada columna si es outlier (columna de marcas)
            for col, flags in outlier_cols.items():
                tmp_df[f'Outlier_{col}'] = flags
            st.markdown("### Previsualización y edición del archivo cargado")
            st.write("Se detectan outliers por especie, variedad, muestra y periodo usando ±2 desviaciones estándar. Las celdas marcadas en rojo indican outliers.")
            # Mostrar sólo los outliers con estilo si el número de celdas a pintar es razonable
            df_outliers = tmp_df[tmp_df['Outlier'] == True]
            if not df_outliers.empty:
                # Limitar el número de filas a mostrar para evitar exceder el límite de estilos
                max_show = 300
                df_display = df_outliers.copy()
                if len(df_display) > max_show:
                    st.info(f"Hay {len(df_display)} filas con outliers; mostrando las primeras {max_show}.")
                    df_display = df_display.head(max_show)
                def highlight_outliers_cell(val, colname, row):
                    flag_col = f'Outlier_{colname}'
                    if flag_col in row and row[flag_col]:
                        return 'background-color: #ffcccc'
                    return ''
                def style_func(data):
                    return data.apply(lambda row: [ 'background-color: #ffcccc' if (f'Outlier_'+data.columns[i]) in row.index and row[f'Outlier_'+data.columns[i]] else '' for i in range(len(data.columns)) ], axis=1)
                st.write(df_display.style.apply(lambda row: [
                    'background-color: #ffcccc' if (f'Outlier_' + col) in row and row[f'Outlier_' + col] else ''
                    for col in df_display.columns
                ], axis=1))
            else:
                st.info("No se detectaron outliers en los datos cargados.")
            # Editor interactivo: sólo las columnas originales (sin columnas Outlier_*)
            cols_to_edit = [c for c in tmp_df.columns if not c.startswith('Outlier_')]
            edited_df = st.data_editor(
                tmp_df[cols_to_edit],
                use_container_width=True,
                height=350,
                num_rows="dynamic",
                key="uploaded_editor"
            )
            st.info("Si modificas valores en la tabla anterior, se usarán para el procesamiento.")
            # Botón para descargar el archivo editado
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                edited_df.to_excel(writer, index=False, sheet_name='Datos_editados')
            excel_buffer.seek(0)
            st.download_button(
                label="💾 Descargar Excel editado",
                data=excel_buffer.getvalue(),
                file_name="carozos_editado.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            # Configuración de cálculo para cond_sum y cluster grupal
            st.subheader("Configuración de cálculo de clusters")
            col_conf1, col_conf2 = st.columns(2)
            with col_conf1:
                cond_method = st.selectbox(
                    "Método de cálculo de cond_sum",
                    options=["suma", "media"],
                    key="cond_sum_method"
                )
            with col_conf2:
                grp_method = st.selectbox(
                    "Método de agregación para cluster grupal",
                    options=["mean", "mode"],
                    key="cluster_grp_method"
                )
            if st.button("Procesar datos editados y clasificar"):
                # Procesar utilizando el DataFrame editado
                try:
                    df_processed = process_carozos(edited_df, current_plum_rules, current_nect_rules, cond_method, grp_method)
                except Exception as e:
                    st.error(f"Error al procesar el archivo: {e}")
                    df_processed = None
                if df_processed is not None:
                    st.success("¡Procesamiento completado con éxito! 🎉")
                    # Visualización de la tabla completa con posibilidad de filtrar por variedad
                    st.markdown("### Tabla completa de resultados")
                    st.dataframe(df_processed, use_container_width=True)
                    # Selección de variedad para visualizar en detalle
                    variedades = sorted(df_processed[VAR_COLUMN].dropna().unique())
                    seleccion_var = st.selectbox(
                        "Selecciona una variedad para visualizar sus resultados",
                        options=["Todas"] + variedades,
                        key="select_variedad"
                    )
                    if seleccion_var and seleccion_var != "Todas":
                        filtro = df_processed[df_processed[VAR_COLUMN] == seleccion_var]
                    else:
                        filtro = df_processed
                    st.markdown(f"#### Resultados para la variedad: {seleccion_var}")
                    st.dataframe(filtro, use_container_width=True, height=400)
                    # Agregados por variedad usando el método seleccionado para cluster grupal
                    st.markdown("### Agregados por variedad")
                    # Calcular agregados dependiendo del método
                    if grp_method == "mean":
                        agg_cond = df_processed.groupby(VAR_COLUMN, dropna=False)["cond_sum"].mean()
                    else:  # mode
                        def _agg_mode(s):
                            m = s.mode()
                            return m.iloc[0] if not m.empty else np.nan
                        agg_cond = df_processed.groupby(VAR_COLUMN, dropna=False)["cond_sum"].agg(_agg_mode)
                    agg = (
                        df_processed
                        .groupby(VAR_COLUMN, dropna=False)
                        .agg(
                            promedio_cond_sum=("cond_sum", "mean"),
                            muestras=(VAR_COLUMN, "size"),
                        )
                        .reset_index()
                    )
                    agg["cond_sum_grp"] = agg_cond.values
                    # Binning de clusters grupales para visualización
                    # Para mantener coherencia con el método, calculamos los clusters
                    if agg["cond_sum_grp"].notna().nunique() >= 4:
                        bins = pd.qcut(agg["cond_sum_grp"], 4, labels=[1,2,3,4])
                    else:
                        bins = pd.cut(agg["cond_sum_grp"], 4, labels=[1,2,3,4])
                    agg["cluster_grp"] = bins
                    st.dataframe(agg[[VAR_COLUMN, "muestras", "promedio_cond_sum", "cond_sum_grp", "cluster_grp"]], use_container_width=True, height=300)
                    # Botón para descargar resultados completos y agregados
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                        df_processed.to_excel(writer, index=False, sheet_name='Carozos')
                        agg.to_excel(writer, index=False, sheet_name='Agregados_variedad')
                    buf.seek(0)
                    st.download_button(
                        label="📥 Descargar resultados como Excel",
                        data=buf.getvalue(),
                        file_name="carozos_procesados.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        else:
            st.info("No se pudo leer el archivo cargado.")
    else:
        st.info("Esperando que subas un archivo para procesar...")


if __name__ == "__main__":
    segmentacion_app()