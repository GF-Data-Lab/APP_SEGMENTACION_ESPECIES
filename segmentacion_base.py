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


def segmentacion_app(especie: str):
    """Construye la página de segmentación para una especie específica."""
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

    ESPECIES_VALIDAS = {especie}

    WEIGHT_COLS = ("Peso (g)", "Calibre", "Peso")
    COL_FIRMEZA_PUNTO = ("Quilla", "Hombro","Punta")
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
        # Si la columna corresponde al valor de firmeza punto o promedio de mejillas
        if col == "Firmeza punto valor":
            return "FIRMEZA_PUNTO"
        if col == "avg_mejillas":
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

    # -----------------------------------------------------------------------
    # Conversión segura de fechas
    #
    # Las fechas de evaluación pueden venir en formato dd-mm-aaaa o mm-dd-aaaa.
    # Esta función intenta interpretarlas usando dayfirst=True primero y, si
    # produce una fecha inválida, vuelve a intentar con dayfirst=False.
    # Devuelve pd.NaT si no es posible parsear.
    # -----------------------------------------------------------------------
    def _safe_parse_date(val: Union[str, float, int, pd.Timestamp]) -> pd.Timestamp | None:
        if isinstance(val, pd.Timestamp):
            return val
        if not isinstance(val, str):
            try:
                return pd.to_datetime(val, errors="coerce")
            except Exception:
                return pd.NaT
        try:
            dt = pd.to_datetime(val, dayfirst=True, errors="coerce")
        except Exception:
            dt = pd.NaT
        # Si no se obtuvo una fecha válida, intentamos dayfirst=False
        if pd.isna(dt):
            try:
                dt = pd.to_datetime(val, dayfirst=False, errors="coerce")
            except Exception:
                dt = pd.NaT
        if not pd.isna(dt) and dt.month not in [8,9,10,11,12,1,2]:
            return pd.NaT
        return dt

    def process_carozos(
        file: Union[str, Path, pd.DataFrame],
        rules_plum: Dict,
        rules_nect: Dict,
        cond_method: str = "suma",
        grp_method: str = "mean",
        fpd_vars: Sequence[str] | None = None,
        mejillas_method: str = "media",
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
        # Parsear fechas de manera robusta (dd-mm-aaaa o mm-dd-aaaa)
        df[DATE_COLUMN] = df[DATE_COLUMN].apply(_safe_parse_date)
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
        # 3.1) Cálculo de la medida de mejillas: promedio de Mejilla 1 y 2 por fruto
        grp_keys_fp = [VAR_COLUMN, FRUTO_COLUMN]
        # Media de cada mejilla por fruto y luego promedio de ambas
        mej_means = df.groupby(grp_keys_fp)[list(COL_FIRMEZA_MEJILLAS)].transform('mean')
        df["avg_mejillas"] = mej_means.mean(axis=1)
        # Aplicar método de agregación seleccionado (media o moda) para promediar entre réplicas
        if mejillas_method == "moda":
            # Para cada fruto calculamos la moda de avg_mejillas
            def _mode_series(s):
                m = s.mode()
                return m.iloc[0] if not m.empty else s.iloc[0]
            df["avg_mejillas"] = df.groupby(grp_keys_fp)["avg_mejillas"].transform(_mode_series)
        else:
            # Media por fruto (ya calculada) se recalcula por claridad
            df["avg_mejillas"] = df.groupby(grp_keys_fp)["avg_mejillas"].transform('mean')
        # 3.2) Cálculo de la firmeza punto débil según variables seleccionadas
        # Si el usuario no pasa una lista de variables utilizamos todas las disponibles
        if fpd_vars:
            # Calcular el promedio por fruto de cada variable seleccionada
            fpd_means = df.groupby(grp_keys_fp)[list(fpd_vars)].transform('mean')
            # Valor de firmeza punto débil: mínimo de los promedios
            df["Firmeza punto valor"] = fpd_means.min(axis=1)
            # Registrar la columna que dio el mínimo
            df["Firmeza punto columna"] = fpd_means.idxmin(axis=1)
        else:
            # Usar todas las variables físicas si no se especifican
            fpd_means = df.groupby(grp_keys_fp)[list(COL_FIRMEZA_ALL)].transform('mean')
            df["Firmeza punto valor"] = fpd_means.min(axis=1)
            df["Firmeza punto columna"] = fpd_means.idxmin(axis=1)
        # 3.3) Relleno de nulos por primera muestra
        # 5) Relleno de nulos por primera muestra
        # Agrupaciones para relleno, clasificación y clusters incluyen especie y periodo de cosecha
        grp_keys = [ESPECIE_COLUMN, VAR_COLUMN, FRUTO_COLUMN, 'harvest_period']
        df = (
            df.groupby(grp_keys, dropna=False, group_keys=False)
              .apply(
                  _first_sample_fill,
                  NUMERIC_COLS
                  + ["avg_mejillas", "Firmeza punto valor", "Firmeza punto columna"]
              )
        )
        # 4) Clasificación de grupos usando las reglas configuradas
        # Preparamos una lista de columnas a clasificar con sus alias apropiados
        cols_to_classify = [
            ("Firmeza punto valor", "Firmeza_punto_valor"),
            ("avg_mejillas", "avg_mejillas"),
            (COL_BRIX, COL_BRIX),
            (COL_ACIDEZ, COL_ACIDEZ),
        ]
        for col_orig, alias in cols_to_classify:
            out = f"grp_{alias}"
            df[out] = df.apply(lambda r, c=col_orig: _classify_row(r, c, rules_plum, rules_nect), axis=1)
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
        # 9) Validación de fechas: identificar registros con periodo 'muy_temprana' cuyo mes esté fuera del rango esperado
        try:
            df["periodo_inconsistente"] = False
            mask_mt = df["harvest_period"] == "muy_temprana"
            # Fechas fuera de agosto-noviembre son atípicas para muy temprana
            df.loc[mask_mt, "periodo_inconsistente"] = ~df.loc[mask_mt, DATE_COLUMN].dt.month.isin([8,9,10,11])
        except Exception:
            df["periodo_inconsistente"] = False
        return df

    # -----------------------------------------------------------------------
    # Sidebar con menú
    # -----------------------------------------------------------------------
    def generar_menu():
        with st.sidebar:
            show_logo()
            if st.button('Página de Inicio 🏚️'):
                st.switch_page('app.py')
            if st.button('Segmentación Ciruela 🍑'):
                st.switch_page('pages/segmentacion_ciruela.py')
            if st.button('Segmentación Nectarina 🍑'):
                st.switch_page('pages/segmentacion_nectarina.py')
            if st.button('Modelo de Clasificación'):
                st.switch_page('pages/Cluster_especies.py')
            if st.button('Análisis exploratorio'):
                st.switch_page('pages/analisis.py')

    generar_menu()

    titulo_especie = "Nectarina" if especie.lower().startswith("nect") else "Ciruela"
    st.title(f"🛠️ Segmentación {titulo_especie}")
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

    if especie == "Ciruela":
        default_plum = st.selectbox(
            "Tipo de ciruela por defecto si el peso no está disponible",
            options=["cherry", "candy"],
            index=["cherry", "candy"].index(st.session_state["default_plum_subtype"]),
            key="default_plum_subtype",
        )
        st.session_state["cherry_upper"] = st.number_input(
            "Peso máximo para cherry (g)",
            min_value=10.0,
            max_value=200.0,
            value=float(st.session_state["cherry_upper"]),
            step=1.0,
        )
    else:
        st.session_state["default_color"] = st.selectbox(
            "Color de pulpa por defecto para Nectarina (si falta)",
            options=["Amarilla", "Blanca"],
            index=["Amarilla", "Blanca"].index(st.session_state["default_color"]),
        )
        st.session_state["default_period"] = st.selectbox(
            "Periodo de cosecha por defecto para Nectarina (si falta fecha)",
            options=["muy_temprana", "temprana", "tardia", "sin_fecha"],
            index=["muy_temprana", "temprana", "tardia", "sin_fecha"].index(st.session_state["default_period"]),
        )

    st.info(
        "Puedes cambiar estos valores y ver cómo afectan a las clasificaciones al recargar el archivo."
    )

    # -----------------------------------------------------------------------
    # Selección de variables para la firmeza punto débil y método de mejillas
    # -----------------------------------------------------------------------
    st.subheader("Configuración de métricas de firmeza")
    # Variables disponibles para el cálculo de la firmeza punto débil
    available_fpd_vars = list(COL_FIRMEZA_ALL)
    # Inicializar selección por defecto si no existe
    if "fpd_vars" not in st.session_state:
        st.session_state["fpd_vars"] = available_fpd_vars
    fpd_selection = st.multiselect(
        "Variables a considerar para la firmeza punto débil",
        options=available_fpd_vars,
        default=st.session_state["fpd_vars"],
        key="fpd_vars"
    )
    # Método de agregación para las mejillas (media o moda)
    if "mejillas_method" not in st.session_state:
        st.session_state["mejillas_method"] = "media"
    st.session_state["mejillas_method"] = st.selectbox(
        "Método de agregación de la medida de mejillas (avg_mejillas)",
        options=["media", "moda"],
        index=["media", "moda"].index(st.session_state["mejillas_method"]),
        key="mejillas_method_select"
    )

    # -----------------------------------------------------------------------
    # Reglas editables
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # Reglas de segmentación
    #
    # Para mantener la compatibilidad con la edición de reglas a través del
    # árbol de decisiones, seguimos cargando las reglas desde la sesión.  Sin
    # embargo, ya no mostramos tablas de edición separadas.  El usuario
    # solamente puede crear nuevas métricas o modificar bandas a través del
    # árbol de decisiones.
    # -----------------------------------------------------------------------
    if "plum_rules_df" not in st.session_state:
        st.session_state["plum_rules_df"] = plum_rules_to_df(DEFAULT_PLUM_RULES)
    if "nect_rules_df" not in st.session_state:
        st.session_state["nect_rules_df"] = nect_rules_to_df(DEFAULT_NECT_RULES)

    current_plum_rules = df_to_plum_rules(st.session_state["plum_rules_df"])
    current_nect_rules = df_to_nect_rules(st.session_state["nect_rules_df"])

    # -----------------------------------------------------------------------
    # Editor tabular para añadir/eliminar reglas y decidir si se aplican
    # -----------------------------------------------------------------------
    # Árbol de decisión para visualizar y editar reglas de forma más intuitiva
    # -----------------------------------------------------------------------
    st.subheader("Árbol de decisiones y editor de reglas")
    st.write(
        "Selecciona los parámetros (tipo/color/periodo → métrica) para ver y modificar las bandas de cada regla con un código de colores similar al flujograma. Los cambios se reflejan automáticamente en la tabla de reglas."
    )
    # Mapa de colores para grupos 1–4 inspirado en el diagrama
    group_colors = {
        1: '#a8e6cf',  # verde claro
        2: '#ffd3b6',  # naranja claro
        3: '#ffaaa5',  # coral
        4: '#ff8b94',  # rojo rosado
    }
    especie_seleccion = especie
    if especie_seleccion == "Ciruela":
        subtipo_sel = st.selectbox("Sub‑tipo de ciruela", list(current_plum_rules.keys()))
        metrica_sel = st.selectbox("Métrica", list(current_plum_rules[subtipo_sel].keys()))
        # Posibilidad de añadir una nueva métrica para este subtipo
        with st.expander("Agregar nueva métrica para este sub‑tipo", expanded=False):
            nueva_metric = st.text_input("Nombre de la nueva métrica", key=f"new_metric_plum_{subtipo_sel}")
            if st.button("Crear métrica", key=f"create_metric_plum_{subtipo_sel}"):
                if nueva_metric:
                    if nueva_metric not in current_plum_rules[subtipo_sel]:
                        # Definir bandas por defecto: 4 grupos con límites equiespaciados 0,1,2 (el usuario debe ajustarlos)
                        default_bands = [(-np.inf, 0.0, 4), (0.0, 1.0, 3), (1.0, 2.0, 2), (2.0, np.inf, 1)]
                        current_plum_rules[subtipo_sel][nueva_metric] = default_bands
                        # Actualizar DataFrame
                        st.session_state["plum_rules_df"] = plum_rules_to_df(current_plum_rules)
                        st.success(f"Métrica '{nueva_metric}' añadida.")
                    else:
                        st.warning("La métrica ya existe.")
                else:
                    st.warning("Debes introducir un nombre para la nueva métrica.")
        bandas = current_plum_rules[subtipo_sel][metrica_sel]
        # Mostrar bandas con colores
        bandas_df = pd.DataFrame(bandas, columns=["Min", "Max", "Grupo"])
        def _apply_colors_plum(row):
            return [f"background-color: {group_colors.get(int(row['Grupo']), '')}" for _ in row]
        try:
            st.write(bandas_df.style.apply(_apply_colors_plum, axis=1))
        except AttributeError as e:
            # pandas.DataFrame.style requires jinja2. If it's missing, fall back to
            # displaying the table without styling and warn the user.
            if "jinja2" in str(e).lower():
                st.warning("Instala 'jinja2' para ver la tabla con colores.")
                st.write(bandas_df)
            else:
                raise
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
        # Posibilidad de añadir una nueva métrica para este color/periodo
        with st.expander("Agregar nueva métrica para este color/periodo", expanded=False):
            nueva_metric_n = st.text_input(
                "Nombre de la nueva métrica", key=f"new_metric_nect_{color_sel}_{periodo_sel}"
            )
            if st.button("Crear métrica", key=f"create_metric_nect_{color_sel}_{periodo_sel}"):
                if nueva_metric_n:
                    if nueva_metric_n not in current_nect_rules[color_sel][periodo_sel]:
                        default_bands_n = [(-np.inf, 0.0, 4), (0.0, 1.0, 3), (1.0, 2.0, 2), (2.0, np.inf, 1)]
                        current_nect_rules[color_sel][periodo_sel][nueva_metric_n] = default_bands_n
                        st.session_state["nect_rules_df"] = nect_rules_to_df(current_nect_rules)
                        st.success(f"Métrica '{nueva_metric_n}' añadida.")
                    else:
                        st.warning("La métrica ya existe.")
                else:
                    st.warning("Debes introducir un nombre para la nueva métrica.")
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
                # Parsear fechas utilizando el mismo método robusto de la función principal
                tmp_df[DATE_COLUMN] = tmp_df[DATE_COLUMN].apply(_safe_parse_date)
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
            # Construir tabla de outliers con media de grupo para cada métrica
            outlier_rows = []
            # Calcular medias de grupo para cada columna numérica
            group_cols = [ESPECIE_COLUMN, VAR_COLUMN, FRUTO_COLUMN, 'harvest_period']
            group_means = {}
            for col in [c for c in NUMERIC_COLS if c in tmp_df.columns]:
                group_means[col] = tmp_df.groupby(group_cols)[col].transform('mean')
                tmp_df[f'Mean_{col}'] = group_means[col]
                # crear fila por outlier
                flagged = tmp_df[tmp_df.get(f'Outlier_{col}', False)]
                for idx, r in flagged.iterrows():
                    outlier_rows.append({
                        'index': idx,
                        'Especie': r[ESPECIE_COLUMN],
                        'Variedad': r[VAR_COLUMN],
                        'Fruto': r[FRUTO_COLUMN],
                        'Periodo': r['harvest_period'],
                        'Métrica': col,
                        'Valor': r[col],
                        'Media_grupo': r[f'Mean_{col}'],
                        'Diferencia': r[col] - r[f'Mean_{col}'],
                    })
            if outlier_rows:
                df_out_table = pd.DataFrame(outlier_rows)
                st.info(f"Se detectaron {len(df_out_table)} valores outlier.")
                # Filtros para especie, variedad y periodo
                esp_options = ['Todas'] + sorted(df_out_table['Especie'].dropna().unique())
                esp_sel = st.selectbox("Filtrar por especie", options=esp_options, key="filtro_out_especie")
                var_options = ['Todas'] + sorted(df_out_table['Variedad'].dropna().unique())
                var_sel = st.selectbox("Filtrar por variedad", options=var_options, key="filtro_out_variedad")
                per_options = ['Todas'] + sorted(df_out_table['Periodo'].dropna().unique())
                per_sel = st.selectbox("Filtrar por periodo", options=per_options, key="filtro_out_periodo")
                df_filt = df_out_table.copy()
                if esp_sel != 'Todas':
                    df_filt = df_filt[df_filt['Especie'] == esp_sel]
                if var_sel != 'Todas':
                    df_filt = df_filt[df_filt['Variedad'] == var_sel]
                if per_sel != 'Todas':
                    df_filt = df_filt[df_filt['Periodo'] == per_sel]
                # Editor para modificar sólo la columna Valor
                edited_outliers = st.data_editor(
                    df_filt,
                    use_container_width=True,
                    height=350,
                    num_rows="dynamic",
                    column_config={
                        'Valor': {'editable': True},
                        'Media_grupo': {'editable': False},
                        'Diferencia': {'editable': False},
                        'index': {'editable': False},
                        'Especie': {'editable': False},
                        'Variedad': {'editable': False},
                        'Fruto': {'editable': False},
                        'Periodo': {'editable': False},
                        'Métrica': {'editable': False},
                    },
                    key="outliers_editor"
                )
                # Botón para guardar cambios en outliers
                if st.button("Aplicar cambios a los outliers"):
                    # Actualizar tmp_df con los valores editados
                    for _, row in edited_outliers.iterrows():
                        idx = int(row['index'])
                        col = row['Métrica']
                        val = row['Valor']
                        tmp_df.at[idx, col] = val
                    st.success("Se actualizaron los valores de outliers en el dataset.")
                # Descargar tabla de outliers
                out_buffer = io.BytesIO()
                with pd.ExcelWriter(out_buffer, engine='xlsxwriter') as writer:
                    edited_outliers.to_excel(writer, index=False, sheet_name='Outliers')
                out_buffer.seek(0)
                st.download_button(
                    label="📥 Descargar tabla de outliers",
                    data=out_buffer.getvalue(),
                    file_name="outliers.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.info("No se detectaron outliers en los datos cargados.")
            # Editor interactivo: sólo las columnas originales (sin columnas Outlier_*, Mean_*)
            cols_to_edit = [c for c in tmp_df.columns if not c.startswith('Outlier_') and not c.startswith('Mean_')]
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
                    df_processed = process_carozos(
                        edited_df,
                        current_plum_rules,
                        current_nect_rules,
                        cond_method,
                        grp_method,
                        fpd_vars=st.session_state.get("fpd_vars"),
                        mejillas_method=st.session_state.get("mejillas_method"),
                    )
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
                    # Agregados por grupo (especie, variedad, fruto y periodo)
                    st.markdown("### Agregados por combinación de especie, variedad, fruto y periodo")
                    group_cols = [ESPECIE_COLUMN, VAR_COLUMN, FRUTO_COLUMN, 'harvest_period']
                    # Cálculo de promedios de cond_sum y otros indicadores por grupo
                    if grp_method == "mean":
                        cond_agg = df_processed.groupby(group_cols, dropna=False)["cond_sum"].mean()
                    else:
                        def _agg_mode(s):
                            m = s.mode()
                            return m.iloc[0] if not m.empty else np.nan
                        cond_agg = df_processed.groupby(group_cols, dropna=False)["cond_sum"].agg(_agg_mode)
                    agg_groups = (
                        df_processed
                        .groupby(group_cols, dropna=False)
                        .agg(
                            muestras=("cond_sum", "size"),
                            promedio_cond_sum=("cond_sum", "mean"),
                            promedio_brix=(COL_BRIX, "mean"),
                            promedio_acidez=(COL_ACIDEZ, "mean"),
                            promedio_firmeza_punto=("Firmeza punto valor", "mean"),
                            promedio_mejillas=("avg_mejillas", "mean"),
                            periodo_inconsistente=("periodo_inconsistente", "max"),
                        )
                        .reset_index()
                    )
                    agg_groups["cond_sum_grp"] = cond_agg.values
                    # Binning de clusters grupales para visualización
                    if agg_groups["cond_sum_grp"].notna().nunique() >= 4:
                        bins = pd.qcut(agg_groups["cond_sum_grp"], 4, labels=[1,2,3,4])
                    else:
                        bins = pd.cut(agg_groups["cond_sum_grp"], 4, labels=[1,2,3,4])
                    agg_groups["cluster_grp"] = bins
                    # Calcular agregados por variedad (sin distinguir fruto ni periodo)
                    if grp_method == "mean":
                        cond_agg_var = df_processed.groupby(VAR_COLUMN, dropna=False)["cond_sum"].mean()
                    else:
                        def _agg_mode_var(s):
                            m = s.mode()
                            return m.iloc[0] if not m.empty else np.nan
                        cond_agg_var = df_processed.groupby(VAR_COLUMN, dropna=False)["cond_sum"].agg(_agg_mode_var)
                    agg_variedad = (
                        df_processed
                        .groupby(VAR_COLUMN, dropna=False)
                        .agg(
                            muestras=("cond_sum", "size"),
                            promedio_cond_sum=("cond_sum", "mean"),
                            promedio_brix=(COL_BRIX, "mean"),
                            promedio_acidez=(COL_ACIDEZ, "mean"),
                            promedio_firmeza_punto=("Firmeza punto valor", "mean"),
                            promedio_mejillas=("avg_mejillas", "mean"),
                        )
                        .reset_index()
                    )
                    agg_variedad["cond_sum_grp"] = cond_agg_var.values
                    if agg_variedad["cond_sum_grp"].notna().nunique() >= 4:
                        bins_var = pd.qcut(agg_variedad["cond_sum_grp"], 4, labels=[1,2,3,4])
                    else:
                        bins_var = pd.cut(agg_variedad["cond_sum_grp"], 4, labels=[1,2,3,4])
                    agg_variedad["cluster_grp"] = bins_var

                    # Clasificación agregada por métricas basándose en los promedios del grupo
                    # Construir un mapa con información de sub‑tipo y color para cada grupo
                    group_info = {}
                    for key, grp in df_processed.groupby(group_cols):
                        first_row = grp.iloc[0]
                        group_info[key] = {
                            'plum_subtype': first_row.get('plum_subtype', 'cherry'),
                            'color': str(first_row.get(COLOR_COLUMN, '')).strip().lower(),
                        }
                    # Clasificar cada métrica promedio
                    brix_classes = []
                    mej_classes = []
                    fpd_classes = []
                    acid_classes = []
                    for _, row in agg_groups.iterrows():
                        key = (row[ESPECIE_COLUMN], row[VAR_COLUMN], row[FRUTO_COLUMN], row['harvest_period'])
                        info = group_info.get(key, {})
                        especie = row[ESPECIE_COLUMN]
                        # Seleccionar reglas según especie
                        if especie == 'Ciruela':
                            subtype = info.get('plum_subtype', 'cherry')
                            rules_dict = current_plum_rules.get(subtype, {})
                        else:
                            # Determinar color base (amarilla o blanca)
                            color_key = 'blanca' if info.get('color', 'amarilla').startswith('blanc') else 'amarilla'
                            period_key = row['harvest_period']
                            rules_dict = current_nect_rules.get(color_key, {}).get(period_key, {})
                        # Obtener reglas por métrica
                        rules_brix = rules_dict.get(COL_BRIX, [])
                        rules_mej = rules_dict.get('FIRMEZA_MEJ', [])
                        rules_fpd = rules_dict.get('FIRMEZA_PUNTO', [])
                        rules_acid = rules_dict.get(COL_ACIDEZ, [])
                        # Clasificación usando las reglas
                        brix_classes.append(_classify_value(row['promedio_brix'], rules_brix))
                        mej_classes.append(_classify_value(row['promedio_mejillas'], rules_mej))
                        fpd_classes.append(_classify_value(row['promedio_firmeza_punto'], rules_fpd))
                        # Para acidez, utilizar sólo el primer valor de la muestra si existe
                        # En el promedio de acidez ya se utilizó la media; si se desea tomar el primer valor,
                        # podemos tomar el primer valor de esa combinación en df_processed.
                        # Buscamos el primer valor real (no nulo) en df_processed para este grupo
                        df_group = df_processed[(df_processed[ESPECIE_COLUMN] == row[ESPECIE_COLUMN]) &
                                               (df_processed[VAR_COLUMN] == row[VAR_COLUMN]) &
                                               (df_processed[FRUTO_COLUMN] == row[FRUTO_COLUMN]) &
                                               (df_processed['harvest_period'] == row['harvest_period'])]
                        first_acid = df_group[COL_ACIDEZ].dropna().iloc[0] if not df_group[COL_ACIDEZ].dropna().empty else row['promedio_acidez']
                        acid_classes.append(_classify_value(first_acid, rules_acid))
                    # Añadir columnas de bandas por métrica
                    agg_groups['grp_brix'] = brix_classes
                    agg_groups['grp_mejillas'] = mej_classes
                    agg_groups['grp_firmeza_punto'] = fpd_classes
                    agg_groups['grp_acidez'] = acid_classes
                    # Recalcular cond_sum a nivel de grupo con estas bandas
                    metric_groups = ['grp_brix', 'grp_mejillas', 'grp_firmeza_punto', 'grp_acidez']
                    if cond_method == 'media':
                        agg_groups['cond_sum_metric'] = agg_groups[metric_groups].mean(axis=1, skipna=True)
                    else:
                        agg_groups['cond_sum_metric'] = agg_groups[metric_groups].sum(axis=1, min_count=1)
                    # Calcular cluster basado en cond_sum_metric
                    if agg_groups['cond_sum_metric'].notna().nunique() >= 4:
                        bins_metric = pd.qcut(agg_groups['cond_sum_metric'], 4, labels=[1,2,3,4])
                    else:
                        bins_metric = pd.cut(agg_groups['cond_sum_metric'], 4, labels=[1,2,3,4])
                    agg_groups['cluster_metric'] = bins_metric
                    # Sustituir cluster_grp por el resultado basado en métricas agregadas
                    agg_groups['cluster_grp'] = agg_groups['cluster_metric']
                    agg_groups['cond_sum_grp'] = agg_groups['cond_sum_metric']

                    # Estilo para colorear según cluster
                    def color_cluster(val):
                        try:
                            grp = int(val)
                            return f"background-color: {group_colors.get(grp, '')}"
                        except:
                            return ''
                    def highlight_inconsistent(row):
                        """Devuelve estilos CSS para filas con periodos inconsistentes."""
                        if row.get('periodo_inconsistente'):
                            return ['background-color: #ffd6d6;' for _ in row]
                        else:
                            return ['' for _ in row]
                    styled_agg = (
                        agg_groups.style
                        .applymap(color_cluster, subset=["cluster_grp"])
                        .apply(highlight_inconsistent, axis=1)
                    )
                    st.dataframe(styled_agg, use_container_width=True, height=400)

                    # Gráfico de las combinaciones en dos dimensiones (PCA)
                    st.markdown("#### Distribución PCA de los grupos")
                    try:
                        # Importaciones necesarias para el gráfico PCA
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.decomposition import PCA
                        import altair as alt
                        # Seleccionamos las características numéricas para el análisis
                        pca_features = [
                            "promedio_cond_sum",
                            "promedio_brix",
                            "promedio_acidez",
                            "promedio_firmeza_punto",
                            "promedio_mejillas",
                        ]
                        df_features = agg_groups[pca_features].fillna(0)
                        # Normalizamos
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(df_features)
                        pca = PCA(n_components=2)
                        pcs = pca.fit_transform(X_scaled)
                        agg_groups["PC1"] = pcs[:, 0]
                        agg_groups["PC2"] = pcs[:, 1]
                        # Construir gráfico interactivo con Altair
                        color_scale = alt.Scale(domain=[1,2,3,4], range=[group_colors[1], group_colors[2], group_colors[3], group_colors[4]])
                        chart = (
                            alt.Chart(agg_groups)
                            .mark_circle(size=80)
                            .encode(
                                x=alt.X("PC1", title="Componente principal 1"),
                                y=alt.Y("PC2", title="Componente principal 2"),
                                color=alt.Color("cluster_grp:N", scale=color_scale, legend=alt.Legend(title="Cluster")),
                                tooltip=[ESPECIE_COLUMN, VAR_COLUMN, FRUTO_COLUMN, 'harvest_period', 'cluster_grp', 'promedio_acidez']
                            )
                            .properties(width='container', height=400)
                            .interactive()
                        )
                        st.altair_chart(chart, use_container_width=True)
                    except Exception as e:
                        st.info(f"No fue posible generar el gráfico PCA: {e}")

                    # Mostrar agregados por variedad (sin separar fruto ni periodo)
                    st.markdown("### Agregados por variedad")
                    st.dataframe(agg_variedad, use_container_width=True, height=300)
                    # Botón para descargar resultados completos y agregados
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                        df_processed.to_excel(writer, index=False, sheet_name='Carozos')
                        agg_groups.to_excel(writer, index=False, sheet_name='Agregados_grupo')
                        agg_variedad.to_excel(writer, index=False, sheet_name='Agregados_variedad')
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
