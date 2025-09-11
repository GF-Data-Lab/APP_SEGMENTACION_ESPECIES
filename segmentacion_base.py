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
import unicodedata
import io
import altair as alt

from utils import show_logo


def load_excel_with_headers_detection(file_path: Union[str, Path], sheet_name: str, usecols: str = None) -> pd.DataFrame:
    """
    Carga un archivo Excel detectando automáticamente dónde están los encabezados.
    
    Args:
        file_path: Ruta al archivo Excel o objeto de archivo
        sheet_name: Nombre de la hoja
        usecols: Columnas a leer (ej: "A:AP")
    
    Returns:
        DataFrame con los encabezados correctamente detectados
    """
    try:
        # Primero, leer las primeras filas sin skiprows para encontrar encabezados
        try:
            preview_df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=usecols, nrows=10, dtype=str)
        except UnicodeDecodeError:
            # Si hay problemas de codificación, intentar con diferentes engines
            try:
                preview_df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=usecols, nrows=10, dtype=str, engine='openpyxl')
            except:
                preview_df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=usecols, nrows=10, dtype=str, engine='xlrd')
        
        # Buscar la fila que probablemente contenga los encabezados
        # Los encabezados usualmente tienen más texto y menos nulos
        header_row = 0
        max_non_null = 0
        
        for i in range(min(5, len(preview_df))):  # Revisar las primeras 5 filas
            non_null_count = preview_df.iloc[i].notna().sum()
            # También verificar que no sean solo números (que serían datos, no encabezados)
            text_count = sum(1 for val in preview_df.iloc[i] if isinstance(str(val), str) and len(str(val)) > 2)
            
            if non_null_count > max_non_null and text_count > non_null_count * 0.3:
                max_non_null = non_null_count
                header_row = i
        
        # Si no encontramos encabezados convincentes en las primeras filas, usar la primera
        if max_non_null < 3:
            header_row = 0
            
        # Ahora cargar el archivo completo usando la fila de encabezados detectada
        try:
            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                usecols=usecols,
                header=header_row,
                dtype=str
            )
        except UnicodeDecodeError:
            # Si hay problemas de codificación, intentar con diferentes engines
            try:
                df = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                    usecols=usecols,
                    header=header_row,
                    dtype=str,
                    engine='openpyxl'
                )
            except:
                # Último intento con xlrd
                df = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                    usecols=usecols,
                    header=header_row,
                    dtype=str,
                    engine='xlrd'
                )
        
        # Limpiar encabezados: eliminar espacios extra, convertir a string, manejar encoding
        def safe_str_clean(col, i):
            try:
                if col is None:
                    return f"Column_{i}"
                # Convertir a string de forma segura
                if isinstance(col, bytes):
                    try:
                        col_str = col.decode('utf-8')
                    except UnicodeDecodeError:
                        col_str = col.decode('latin-1', errors='ignore')
                else:
                    col_str = str(col)
                return col_str.strip()
            except:
                return f"Column_{i}"
        
        cleaned_columns = [safe_str_clean(col, i) for i, col in enumerate(df.columns)]
        
        # Manejar columnas duplicadas
        seen_columns = {}
        final_columns = []
        
        for col in cleaned_columns:
            if col in seen_columns:
                seen_columns[col] += 1
                final_columns.append(f"{col}_{seen_columns[col]}")
            else:
                seen_columns[col] = 0
                final_columns.append(col)
        
        df.columns = final_columns
        
        # Filtrar filas vacías después de los encabezados
        df = df.dropna(how='all')
        
        # Limpiar valores de celdas con problemas de encoding
        def safe_cell_clean(val):
            if val is None or pd.isna(val):
                return val
            try:
                if isinstance(val, bytes):
                    try:
                        return val.decode('utf-8')
                    except UnicodeDecodeError:
                        return val.decode('latin-1', errors='ignore')
                return str(val)
            except:
                return str(val) if val is not None else val
        
        # Aplicar limpieza a todas las celdas de texto
        for col in df.columns:
            try:
                df[col] = df[col].apply(safe_cell_clean)
            except:
                continue
        
        return df
        
    except Exception as e:
        st.error(f"Error al cargar el archivo Excel: {e}")
        # Fallback: intentar con la configuración original
        return pd.read_excel(file_path, sheet_name=sheet_name, usecols=usecols, skiprows=2, dtype=str)


# ----------------------------------------------------------------------
# Bandas por defecto y utilidades de conversión disponibles a nivel de
# módulo para que otras páginas puedan reutilizarlas.
# ----------------------------------------------------------------------
DEFAULT_PLUM_RULES: Dict[str, Dict[str, List[Tuple[float, float, int]]]] = {
    "candy": {
        "BRIX": [(18.0, np.inf, 1), (16.0, 18.0, 2), (14.0, 16.0, 3), (-np.inf, 14.0, 4)],
        "FIRMEZA_PUNTO": [(7.0, np.inf, 1), (5.0, 7.0, 2), (4.0, 5.0, 3), (-np.inf, 4.0, 4)],
        "FIRMEZA_MEJ": [(9.0, np.inf, 1), (7.0, 9.0, 2), (6.0, 7.0, 3), (-np.inf, 6.0, 4)],
        "Acidez (%)": [(-np.inf, 0.60, 1), (0.60, 0.81, 2), (0.81, 1.00, 3), (1.00, np.inf, 4)],
    },
    "sugar": {
        "BRIX": [(21.0, np.inf, 1), (18.0, 21.0, 2), (15.0, 18.0, 3), (-np.inf, 15.0, 4)],
        "FIRMEZA_PUNTO": [(6.0, np.inf, 1), (4.5, 6.0, 2), (3.0, 4.5, 3), (-np.inf, 3.0, 4)],
        "FIRMEZA_MEJ": [(8.0, np.inf, 1), (5.0, 8.0, 2), (4.0, 5.0, 3), (-np.inf, 4.0, 4)],
        "Acidez (%)": [(-np.inf, 0.60, 1), (0.60, 0.81, 2), (0.81, 1.00, 3), (1.00, np.inf, 4)],
    },
}


def _mk_nec_rules(
    brix1: float, brix2: float, brix3: float,
    mej_1: float, mej_2: float,
) -> Dict[str, List[Tuple[float, float, int]]]:
    return {
        "BRIX": [(brix1, np.inf, 1), (brix2, brix1, 2), (brix3, brix2, 3), (-np.inf, brix3, 4)],
        "FIRMEZA_PUNTO": [(9.0, np.inf, 1), (8.0, 9.0, 2), (7.0, 8.0, 3), (-np.inf, 7.0, 4)],
        "FIRMEZA_MEJ": [(mej_1, np.inf, 1), (mej_2, mej_1, 2), (9.0, mej_2, 3), (-np.inf, 9.0, 4)],
        "Acidez (%)": [(-np.inf, 0.60, 1), (0.60, 0.81, 2), (0.81, 1.00, 3), (1.00, np.inf, 4)],
    }


DEFAULT_NECT_RULES: Dict[str, Dict[str, Dict[str, List[Tuple[float, float, int]]]]] = {
    "amarilla": {
        "muy_temprana": _mk_nec_rules(13.0, 10.0, 9.0, 14.0, 12.0),
        "temprana": _mk_nec_rules(13.0, 10.0, 9.0, 14.0, 12.0),
        "tardia": _mk_nec_rules(14.0, 12.0, 10.0, 14.0, 12.0),
    },
    "blanca": {
        "muy_temprana": _mk_nec_rules(13.0, 10.0, 9.0, 13.0, 11.0),
        "temprana": _mk_nec_rules(13.0, 10.0, 9.0, 13.0, 11.0),
        "tardia": _mk_nec_rules(14.0, 12.0, 10.0, 13.0, 11.0),
    },
}


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

    # Normalizar el nombre de la especie para que coincida con las etiquetas del
    # archivo de datos.  El usuario puede ingresar "Nectarina" pero en el Excel
    # la especie está registrada como "Nectarin".
    especie_key = "Nectarin" if especie.lower().startswith("nect") else "Ciruela"
    titulo_especie = "Nectarina" if especie_key == "Nectarin" else "Ciruela"

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

    ESPECIES_VALIDAS = {especie_key}

    WEIGHT_COLS = ("Peso (g)", "Calibre", "Peso")
    COL_FIRMEZA_PUNTO = ("Quilla", "Hombro","Punta")
    COL_FIRMEZA_MEJILLAS = ("Mejilla 1", "Mejilla 2")
    COL_FIRMEZA_ALL = list(COL_FIRMEZA_PUNTO + COL_FIRMEZA_MEJILLAS)

    COL_ORIG_BRIX = "Solidos solubles (%)"
    COL_BRIX = "BRIX"
    COL_ACIDEZ = "Acidez (%)"
    
    # Mapeo de periodos de cosecha 
    PERIOD_MAP = {
        "muy temprana": "muy_temprana",
        "muy_temprana": "muy_temprana",
        "temprana": "temprana", 
        "tardia": "tardia",
        "tardía": "tardia",
        "sin fecha": "sin_fecha",
        "sin_fecha": "sin_fecha",
        "": "sin_fecha"
    }

    NUMERIC_COLS = (
        COL_FIRMEZA_ALL
        + [COL_BRIX, COL_ACIDEZ]
        + [c for c in WEIGHT_COLS]
    )

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
            return st.session_state.get("default_plum_subtype", "sugar")
        # Lógica de clasificación configurable por el usuario
        sugar_cut = st.session_state.get("sugar_upper", 60.0)
        if peso > sugar_cut:
            return "candy"
        # Cualquier ciruela con peso ≤ sugar_cut se considera sugar
        return "sugar"

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
            subtype = row.get("plum_subtype", "sugar")
            rule_dict = rules_plum.get(subtype, {}).get(key, [])
            return _classify_value(row[col], rule_dict)
        if row[ESPECIE_COLUMN] == "Nectarin":
            color = str(row.get(COLOR_COLUMN, "")).strip().lower() or st.session_state.get("default_color", "amarilla")
            color = "blanca" if color.startswith("blanc") else "amarilla"
            raw_period = str(row.get("harvest_period", "")).strip().lower()
            raw_period = unicodedata.normalize("NFD", raw_period).encode("ascii", "ignore").decode("utf-8")
            period = PERIOD_MAP.get(
                raw_period,
                st.session_state.get("default_period", "tardia"),
            )
            rule_dict = rules_nect.get(color, {}).get(period, {}).get(key, [])
            return _classify_value(row[col], rule_dict)
        return np.nan

    def _harvest_period_a(ts: pd.Timestamp | float | str) -> str:
        ts = pd.to_datetime(ts, errors="coerce")
        if pd.isna(ts):
            return "Period tardia"
        m, d = ts.month, ts.day
        if (m, d) < (11, 22):
            return "Period muy_temprana"
        if (11, 22) <= (m, d) <= (12, 22):
            return "Period temprana"
        if (12, 23) <= (m, d) <= (2, 15):
            return "Period tardia"
        return "Period tardia"

    def _harvest_period_b(ts: pd.Timestamp | float | str) -> str:
        ts = pd.to_datetime(ts, errors="coerce")
        if pd.isna(ts):
            return "Period tardia"
        m, d = ts.month, ts.day
        if (m, d) < (11, 25):
            return "Period muy_temprana"
        if (11, 25) <= (m, d) <= (12, 15):
            return "Period temprana"
        if (12, 16) <= (m, d) <= (2, 15):
            return "Period tardia"
        return "Period tardia"

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
        # Manejar números de Excel (días desde 1900-01-01)
        if isinstance(val, (int, float)) and not pd.isna(val):
            try:
                # Excel guarda fechas como números
                # Intentar convertir directamente
                result = pd.to_datetime(val, unit='D', origin='1899-12-30', errors="coerce")
                if not pd.isna(result):
                    return result
                # Si falla, intentar conversión normal
                return pd.to_datetime(val, errors="coerce")
            except Exception:
                return pd.NaT
        # Manejar strings
        if isinstance(val, str):
            val = val.strip()
            if not val:
                return pd.NaT
            try:
                # Primero intentar con dayfirst=True (formato dd-mm-yyyy)
                dt = pd.to_datetime(val, dayfirst=True, errors="coerce")
                if not pd.isna(dt):
                    return dt
                # Si falla, intentar con dayfirst=False (formato mm-dd-yyyy)
                dt = pd.to_datetime(val, dayfirst=False, errors="coerce")
                return dt
            except Exception:
                return pd.NaT
        return pd.NaT

    def process_carozos(
        file: Union[str, Path, pd.DataFrame],
        rules_plum: Dict,
        rules_nect: Dict,
        cond_method: str = "suma",
        grp_method: str = "mean",
        fpd_vars: Sequence[str] | None = None,
        mejillas_method: str = "media",
        selected_metrics: List[str] = None,
    ) -> pd.DataFrame:
        # Permitir que el parámetro file sea un DataFrame ya cargado
        if isinstance(file, pd.DataFrame):
            df = file.copy()
        else:
            df = load_excel_with_headers_detection(file, SHEET_NAME, USECOLS)
        # 1) Filtros y renombres
        df = df[df[ESPECIE_COLUMN].isin(ESPECIES_VALIDAS)].copy()
        df.rename(columns={COL_ORIG_BRIX: COL_BRIX}, inplace=True)
        # Color de pulpa: reemplazar nulos por valor por defecto
        default_color = "Amarilla"
        if COLOR_COLUMN not in df.columns:
            df[COLOR_COLUMN] = np.nan
        mask_nect = df[ESPECIE_COLUMN] == "Nectarin"
        df.loc[mask_nect, COLOR_COLUMN] = (
            df.loc[mask_nect, COLOR_COLUMN]
            .replace("", np.nan)
            .fillna(default_color)
        )
        # 2) Tipos y periodos
        # Parsear fechas de manera robusta (dd-mm-aaaa o mm-dd-aaaa)
        df[DATE_COLUMN] = df[DATE_COLUMN].apply(_safe_parse_date)
        # Asignación de período y sub-tipo según especie
        df["plum_subtype"] = df.apply(_plum_subtype, axis=1)
        df["harvest_period"] = "Period sin_fecha"  # inicializar
        
        # Para ciruelas, usar periodo tipo A (sin distinción de color)
        idx_ciruela = df[ESPECIE_COLUMN] == "Ciruela"
        df.loc[idx_ciruela, "harvest_period"] = df.loc[idx_ciruela, DATE_COLUMN].apply(_harvest_period_a)
        
        # Para nectarines, determinar periodo según color
        idx_nectar = df[ESPECIE_COLUMN] == "Nectarin"
        color_series = (
            df.get(COLOR_COLUMN, pd.Series("", index=df.index))
              .fillna("")
              .astype(str)
              .str.strip()
              .str.lower()
        )
        idx_blanca = idx_nectar & color_series.str.startswith("blanc")
        df.loc[idx_blanca, "harvest_period"] = df.loc[idx_blanca, DATE_COLUMN].apply(_harvest_period_b)
        df.loc[idx_nectar & ~idx_blanca, "harvest_period"] = df.loc[idx_nectar & ~idx_blanca, DATE_COLUMN].apply(_harvest_period_a)
        
        # Para ambas especies, usar valor por defecto si no se pudo determinar
        df.loc[
            (idx_nectar | idx_ciruela) & (df["harvest_period"] == "Period sin_fecha"),
            "harvest_period",
        ] = "Period " + st.session_state.get("default_period", "tardia")
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
            df["Firmeza punto columna"] = fpd_means.idxmin(axis=1, skipna=True)
        else:
            # Usar todas las variables físicas si no se especifican
            fpd_means = df.groupby(grp_keys_fp)[list(COL_FIRMEZA_ALL)].transform('mean')
            df["Firmeza punto valor"] = fpd_means.min(axis=1)
            df["Firmeza punto columna"] = fpd_means.idxmin(axis=1, skipna=True)
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
        all_cols_to_classify = [
            ("Firmeza punto valor", "Firmeza_punto_valor", "Firmeza punto débil"),
            ("avg_mejillas", "avg_mejillas", "Mejillas"),
            (COL_BRIX, COL_BRIX, "BRIX"),
            (COL_ACIDEZ, COL_ACIDEZ, "Acidez (%)"),
        ]
        
        # Filtrar columnas según métricas seleccionadas
        if selected_metrics is None:
            selected_metrics = ["BRIX", "Acidez (%)", "Firmeza punto débil", "Mejillas"]
        
        cols_to_classify = [(col_orig, alias) for col_orig, alias, metric_name in all_cols_to_classify 
                           if metric_name in selected_metrics]
        
        # Clasificación especial para acidez: usar primer valor por combinación
        grp_keys_acid = [ESPECIE_COLUMN, VAR_COLUMN, "harvest_period"]
        
        for col_orig, alias in cols_to_classify:
            out = f"grp_{alias}"
            
            if col_orig == COL_ACIDEZ:
                # Para acidez, usar el primer valor de cada combinación especie-variedad-temporada
                first_acid_values = df.groupby(grp_keys_acid)[COL_ACIDEZ].first().reset_index()
                first_acid_values[out] = first_acid_values.apply(
                    lambda r: _classify_row(r, COL_ACIDEZ, rules_plum, rules_nect), axis=1
                )
                # Hacer merge para propagar el valor a todas las filas de la combinación
                df = df.merge(first_acid_values[grp_keys_acid + [out]], 
                             on=grp_keys_acid, how='left')
            else:
                # Para otras métricas, clasificación normal
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
            try:
                df["cluster_row"] = pd.qcut(df["cond_sum"], 4, labels=[1,2,3,4])
            except ValueError:
                df["cluster_row"] = pd.cut(df["cond_sum"], 4, labels=[1,2,3,4])
        else:
            df["cluster_row"] = pd.Series(np.nan, index=df.index)
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
        if grp_cond["cond_sum_grp"].notna().sum() >= 4:
            try:
                bins = pd.qcut(grp_cond["cond_sum_grp"], 4, labels=[1,2,3,4])
            except ValueError:
                bins = pd.cut(grp_cond["cond_sum_grp"], 4, labels=[1,2,3,4])
        else:
            bins = pd.Series(np.nan, index=grp_cond.index)
        grp_cond["cluster_grp"] = bins
        df = df.merge(
            grp_cond[grp_keys + ["cluster_grp"]], on=grp_keys, how="left"
        )
        # Asignar rangos de condición agrupada
        df["rankid"] = pd.cut(
            df["cond_sum_grp"],
            bins=[0, 4, 8, 12, 16],
            labels=["Top 1", "Top 2", "Top 3", "Top 4"],
            include_lowest=True,
        )
        # 9) Validación de fechas: identificar registros con periodo 'muy_temprana' cuyo mes esté fuera del rango esperado
        try:
            df["periodo_inconsistente"] = False
            mask_mt = df["harvest_period"] == "Period muy_temprana"
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
            if st.button('Carga de archivos 📁'):
                st.switch_page('pages/carga_datos.py')
            if st.button('Segmentación Ciruela 🍑'):
                st.switch_page('pages/segmentacion_ciruela.py')
            if st.button('Segmentación Nectarina 🍑'):
                st.switch_page('pages/segmentacion_nectarina.py')
            if st.button('Modelo de Clasificación'):
                st.switch_page('pages/Cluster_especies.py')
            if st.button('Análisis exploratorio'):
                st.switch_page('pages/analisis.py')
            if st.button('Métricas y Bandas 📊'):
                st.switch_page('pages/metricas_bandas.py')
            if st.button('Detección Outliers 🎯'):
                st.switch_page('pages/outliers.py')
            if st.button('Verificar Cálculos 🔍'):
                st.switch_page('pages/verificar_calculos.py')
            if st.button('Evolución Variedad 📈'):
                st.switch_page('pages/evolucion_variedad.py')

    generar_menu()

    st.title(f"🛠️ Segmentación {titulo_especie}")
    st.write(
        """
        Asegúrate de haber cargado previamente el archivo Excel correspondiente en la página
        "Carga de archivos" para obtener los clusters,
        clasificaciones y resultados procesados según el flujograma.  Además,
        puedes visualizar y editar las reglas de segmentación y definir valores
        por defecto para campos faltantes.
        """
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
        key="fpd_vars"
    )
    # Método de agregación para las mejillas (media o moda)
    if "mejillas_method" not in st.session_state:
        st.session_state["mejillas_method"] = "media"
    mejillas_method = st.selectbox(
        "Método de agregación de la medida de mejillas (avg_mejillas)",
        options=["media", "moda"],
        key="mejillas_method"
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
    # Configuración de reglas: Las reglas se gestionan ahora desde la página 
    # "Métricas y Bandas". Esta sección solo carga las reglas del session_state.
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Obtención del archivo cargado previamente
    # -----------------------------------------------------------------------
    if especie_key in ("Nectarin", "Ciruela"):
        # Verificar si hay datos filtrados disponibles
        df_filtered = st.session_state.get("carozos_df_filtered")
        df_upload = df_filtered if df_filtered is not None else st.session_state.get("carozos_df")
        file_label = "carozos"
        
        # Mostrar información sobre el tipo de datos usado
        if df_filtered is not None:
            st.info(f"📊 Usando datos filtrados: {len(df_filtered)} registros (de {len(st.session_state.get('carozos_df', []))} originales)")
            if st.button("🔄 Usar datos originales (sin filtros)"):
                if "carozos_df_filtered" in st.session_state:
                    del st.session_state["carozos_df_filtered"]
                st.rerun()
        else:
            df_original = st.session_state.get("carozos_df")
            if df_original is not None:
                st.info(f"📊 Usando todos los datos originales: {len(df_original)} registros")
    else:
        df_upload = st.session_state.get("cerezas_df")
        file_label = "cerezas"

    if df_upload is None:
        st.info(f"No se encontró el archivo de {file_label}. Primero súbelo en la página 'Carga de archivos'.")
        if st.button("📁 Ir a Carga de archivos"):
            st.switch_page('pages/carga_datos.py')
        return

    # -------------------------------------------------------------------
    # Detección de outliers antes del procesamiento (solo si no hay datos procesados)
    # -------------------------------------------------------------------
    process_key = f"processed_data_{especie_key}"
    outliers_key = f"outliers_data_{especie_key}"
    
    # Si ya hay datos procesados, omitir la detección de outliers para mejorar rendimiento
    if process_key not in st.session_state:
        if outliers_key not in st.session_state:
            # Procesar outliers solo la primera vez
            tmp_df = df_upload.copy()
            # Filtrar por especie para evitar mezclar datasets
            if ESPECIE_COLUMN in tmp_df.columns:
                tmp_df = tmp_df[tmp_df[ESPECIE_COLUMN] == especie_key].copy()

            # Convertir fechas
            if DATE_COLUMN in tmp_df.columns:
                tmp_df[DATE_COLUMN] = tmp_df[DATE_COLUMN].apply(_safe_parse_date)
                idx_nectar = tmp_df[ESPECIE_COLUMN] == "Nectarin"
                color_series = (
                    tmp_df.get(COLOR_COLUMN, pd.Series("", index=tmp_df.index))
                      .fillna("")
                      .astype(str)
                      .str.strip()
                      .str.lower()
                )
                idx_blanca = idx_nectar & color_series.str.startswith("blanc")
                tmp_df["harvest_period"] = "Period sin_fecha"
                tmp_df.loc[idx_blanca, "harvest_period"] = tmp_df.loc[idx_blanca, DATE_COLUMN].apply(_harvest_period_b)
                tmp_df.loc[idx_nectar & ~idx_blanca, "harvest_period"] = tmp_df.loc[idx_nectar & ~idx_blanca, DATE_COLUMN].apply(_harvest_period_a)
            else:
                tmp_df["harvest_period"] = "Period sin_fecha"
            
            _to_numeric(tmp_df, NUMERIC_COLS)
            # Detección de outliers por especie, variedad y muestra (|z| > 2)
            # Nota: harvest_period aún no existe en este punto, se crea después
            # Solo usar columnas que existen en el DataFrame
            available_group_cols = [col for col in [ESPECIE_COLUMN, VAR_COLUMN, FRUTO_COLUMN] if col in tmp_df.columns]
            if not available_group_cols:
                # Si no hay columnas de agrupación disponibles, usar índice
                available_group_cols = [tmp_df.index.name] if tmp_df.index.name else ['index']
                if 'index' not in tmp_df.columns:
                    tmp_df['index'] = tmp_df.index
            group_cols = available_group_cols
            outlier_cols = {}
            for col in [c for c in NUMERIC_COLS if c in tmp_df.columns]:
                def _zscore(s: pd.Series) -> pd.Series:
                    m = s.mean()
                    sd = s.std()
                    if sd == 0 or pd.isna(sd):
                        return pd.Series(np.nan, index=s.index)
                    return (s - m) / sd
                z = tmp_df.groupby(group_cols, dropna=False)[col].transform(_zscore)
                mask = z.abs() > 2
                outlier_cols[col] = mask.fillna(False)
                tmp_df[f"Outlier_{col}"] = outlier_cols[col]
            tmp_df["Outlier"] = pd.DataFrame(outlier_cols).any(axis=1) if outlier_cols else False
            
            # Guardar datos de outliers en session_state para evitar recalcular
            st.session_state[outliers_key] = tmp_df
        else:
            # Recuperar datos de outliers del caché
            tmp_df = st.session_state[outliers_key]
    else:
        # Si hay datos procesados, usar el dataframe original simplificado
        tmp_df = df_upload.copy()
        if ESPECIE_COLUMN in tmp_df.columns:
            tmp_df = tmp_df[tmp_df[ESPECIE_COLUMN] == especie_key].copy()
        tmp_df["Outlier"] = False  # No mostrar outliers si ya hay datos procesados
    
    # Inicializar outlier_rows para evitar UnboundLocalError
    outlier_rows = []
    
    # Solo mostrar sección de outliers si no hay datos procesados
    if process_key not in st.session_state:
        st.markdown("### Previsualización y edición del archivo cargado")
        st.write("Se detectan outliers por especie, variedad, muestra y periodo usando ±2 desviaciones estándar. Las celdas marcadas en rojo indican outliers.")
    # Calcular medias de grupo para cada columna numérica
    # Usar harvest_period solo si existe en el dataframe
    # Solo usar columnas que existen en el DataFrame
    available_base_cols = [col for col in [ESPECIE_COLUMN, VAR_COLUMN, FRUTO_COLUMN] if col in tmp_df.columns]
    if not available_base_cols:
        # Si no hay columnas de agrupación disponibles, usar índice
        available_base_cols = [tmp_df.index.name] if tmp_df.index.name else ['index']
        if 'index' not in tmp_df.columns:
            tmp_df['index'] = tmp_df.index
    
    if 'harvest_period' in tmp_df.columns:
        group_cols = available_base_cols + ['harvest_period']
    else:
        group_cols = available_base_cols
    # Asegurar conversión numérica antes de calcular medias
    _to_numeric(tmp_df, NUMERIC_COLS)
    
    group_means = {}
    for col in [c for c in NUMERIC_COLS if c in tmp_df.columns]:
        group_means[col] = tmp_df.groupby(group_cols)[col].transform('mean')
        tmp_df[f'Mean_{col}'] = group_means[col]
        # crear fila por outlier
        outlier_col = f'Outlier_{col}'
        if outlier_col in tmp_df.columns:
            flagged = tmp_df[tmp_df[outlier_col] == True]
        else:
            flagged = pd.DataFrame()  # DataFrame vacío si no existe la columna
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
            key="outliers_editor",
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
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("No se detectaron outliers en los datos cargados.")

    # Excluir outliers antes de la edición y procesamiento
    df_no_outliers = tmp_df[~tmp_df["Outlier"]].copy()
    cols_to_edit = [c for c in df_no_outliers.columns if not c.startswith('Outlier_') and not c.startswith('Mean_')]
    edited_df = st.data_editor(
        df_no_outliers[cols_to_edit],
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
    
    # Selector de métricas para clustering
    st.markdown("#### Selección de métricas para clustering")
    available_metrics = ["BRIX", "Acidez (%)", "Firmeza punto débil", "Mejillas"]
    
    # Inicializar selección por defecto si no existe
    if "selected_cluster_metrics" not in st.session_state:
        st.session_state["selected_cluster_metrics"] = available_metrics
    
    selected_metrics = st.multiselect(
        "Selecciona las métricas que se usarán para el cálculo del cluster:",
        options=available_metrics,
        default=st.session_state.get("selected_cluster_metrics", available_metrics),
        key="selected_cluster_metrics",
        help="Solo las métricas seleccionadas se usarán para calcular los grupos de cluster"
    )
    
    if not selected_metrics:
        st.warning("⚠️ Debes seleccionar al menos una métrica para el clustering.")
        selected_metrics = available_metrics
    
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
    # Inicializar clave de procesamiento en session_state
    process_key = f"processed_data_{especie_key}"
    
    if st.button("Procesar datos editados y clasificar"):
        # Procesar utilizando el DataFrame editado
        with st.spinner("Procesando datos..."):
            try:
              df_processed = process_carozos(
                  edited_df,
                  current_plum_rules,
                  current_nect_rules,
                  cond_method,
                  grp_method,
                  fpd_vars=st.session_state.get("fpd_vars"),
                  mejillas_method=st.session_state.get("mejillas_method"),
                  selected_metrics=selected_metrics,
              )
              # Guardar en session_state para evitar reprocesamiento
              st.session_state[process_key] = {
                  'df_processed': df_processed,
                  'cond_method': cond_method,
                  'grp_method': grp_method,
                  'current_plum_rules': current_plum_rules,
                  'current_nect_rules': current_nect_rules,
                  'selected_metrics': selected_metrics
              }
              st.success("¡Procesamiento completado con éxito! 🎉")
            except Exception as e:
              st.error(f"Error al procesar el archivo: {e}")
              df_processed = None
    
    # Verificar si hay datos procesados disponibles en session_state
    if process_key in st.session_state:
        processed_data = st.session_state[process_key]
        df_processed = processed_data['df_processed']
        cond_method = processed_data['cond_method']
        grp_method = processed_data['grp_method']
        current_plum_rules = processed_data['current_plum_rules']
        current_nect_rules = processed_data['current_nect_rules']
        selected_metrics = processed_data.get('selected_metrics', available_metrics)
        
        # Botón para limpiar datos procesados
        if st.button("🔄 Limpiar resultados y reprocesar"):
            if process_key in st.session_state:
                del st.session_state[process_key]
            st.rerun()
            
    else:
        df_processed = None
        
    if df_processed is not None:
          # Mostrar indicador de que hay datos procesados disponibles
          if process_key in st.session_state:
              metrics_used = ", ".join(selected_metrics) if selected_metrics else "Todas"
              st.info(f"📊 Mostrando resultados procesados guardados.\n\n**Métricas usadas para clustering**: {metrics_used}\n\nUsa el botón '🔄 Limpiar resultados' si quieres reprocesar con nuevos parámetros.")
          
          # Visualización de la tabla completa con posibilidad de filtrar por variedad
          st.markdown("### Tabla completa de resultados")
          
          # Crear lista de columnas relevantes para el análisis
          essential_cols = [
              ESPECIE_COLUMN,
              VAR_COLUMN,
              FRUTO_COLUMN,
              "harvest_period",
              COL_BRIX,
              COL_ACIDEZ,
              "Firmeza punto valor",
              "avg_mejillas",
              "cluster_row",
              "cluster_grp",
              "cond_sum",
              "cond_sum_grp",
              "rankid"
          ]
          
          # Agregar columnas grp_ que existen
          grp_cols = [col for col in df_processed.columns if col.startswith("grp_")]
          essential_cols.extend(grp_cols)
          
          # Filtrar solo columnas que existen en el dataframe y tienen datos significativos
          available_cols = []
          for col in essential_cols:
              if col in df_processed.columns:
                  # Para grp_acidez siempre incluirla (usa primer valor)
                  if col == "grp_Acidez (%)" or col.endswith("acidez"):
                      available_cols.append(col)
                  # Para otras columnas, verificar que no sean todas nulas
                  elif not df_processed[col].isna().all():
                      available_cols.append(col)
          
          # Mostrar dataframe con columnas filtradas
          df_display = df_processed[available_cols].copy()
          st.dataframe(df_display, use_container_width=True)
          
          st.info(f"💡 Mostrando {len(available_cols)} columnas relevantes de {len(df_processed.columns)} totales.")
          if st.button("Guardar dataframe procesado"):
              st.session_state["df_seg_especies"] = df_processed
              st.success("DataFrame guardado para otras páginas.")
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
                  punto_firmeza_min=("Firmeza punto columna", "first"),
                  periodo_inconsistente=("periodo_inconsistente", "max"),
              )
              .reset_index()
          )
          agg_groups["cond_sum_grp"] = cond_agg.values
          if agg_groups["cond_sum_grp"].notna().sum() >= 4:
              try:
                  bins = pd.qcut(agg_groups["cond_sum_grp"], 4, labels=[1,2,3,4])
              except ValueError:
                  bins = pd.cut(agg_groups["cond_sum_grp"], 4, labels=[1,2,3,4])
          elif agg_groups["cond_sum_grp"].notna().sum() >= 2:
              # Si tenemos al menos 2 valores, crear 2 clusters
              try:
                  bins = pd.qcut(agg_groups["cond_sum_grp"], 2, labels=[1,2])
              except ValueError:
                  bins = pd.cut(agg_groups["cond_sum_grp"], 2, labels=[1,2])
          elif agg_groups["cond_sum_grp"].notna().sum() >= 1:
              # Si tenemos al menos 1 valor, asignar cluster 1 a los válidos
              bins = pd.Series(1, index=agg_groups.index)
              bins[agg_groups["cond_sum_grp"].isna()] = np.nan
          else:
              # Si no hay valores válidos, asignar cluster 1 a todos
              bins = pd.Series(1, index=agg_groups.index)
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
                  punto_firmeza_min=("Firmeza punto columna", "first"),
              )
              .reset_index()
          )
          agg_variedad["cond_sum_grp"] = cond_agg_var.values
          if agg_variedad["cond_sum_grp"].notna().sum() >= 4:
              try:
                  bins_var = pd.qcut(agg_variedad["cond_sum_grp"], 4, labels=[1,2,3,4])
              except ValueError:
                  bins_var = pd.cut(agg_variedad["cond_sum_grp"], 4, labels=[1,2,3,4])
          else:
              bins_var = pd.Series(np.nan, index=agg_variedad.index)
          agg_variedad["cluster_grp"] = bins_var

          # Clasificación agregada por métricas basándose en los promedios del grupo
          # Construir un mapa con información de sub‑tipo y color para cada grupo
          group_info = {}
          for key, grp in df_processed.groupby(group_cols):
              first_row = grp.iloc[0]
              group_info[key] = {
                  'plum_subtype': first_row.get('plum_subtype', 'sugar'),
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
                subtype = info.get('plum_subtype', 'sugar')
                rules_dict = current_plum_rules.get(subtype, {})
            elif especie == 'Nectarin':
                color_key = 'blanca' if info.get('color', 'amarilla').startswith('blanc') else 'amarilla'
                period_key = row['harvest_period']
                rules_dict = current_nect_rules.get(color_key, {}).get(period_key, {})
            else:
                rules_dict = {}
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
          # Añadir columnas de bandas por métrica
          agg_groups['grp_brix'] = brix_classes
          agg_groups['grp_mejillas'] = mej_classes
          agg_groups['grp_firmeza_punto'] = fpd_classes
          agg_groups['grp_acidez'] = acid_classes
          # Recalcular cond_sum a nivel de grupo con estas bandas
          metric_groups = ['grp_brix', 'grp_mejillas', 'grp_firmeza_punto', 'grp_acidez']
          
          # Debug: verificar disponibilidad de métricas
          available_metrics = [col for col in metric_groups if col in agg_groups.columns]
          missing_metrics = [col for col in metric_groups if col not in agg_groups.columns]
          st.write(f"**Debug métricas**: Disponibles: {available_metrics}, Faltantes: {missing_metrics}")
          
          if len(available_metrics) == 0:
              st.warning("No hay métricas disponibles para clustering. Asignando cluster 1 a todos.")
              agg_groups['cond_sum_grp'] = 1.0
          else:
              if cond_method == 'media':
                  agg_groups['cond_sum_metric'] = agg_groups[available_metrics].mean(axis=1, skipna=True)
              else:
                  agg_groups['cond_sum_metric'] = agg_groups[available_metrics].sum(axis=1, min_count=1)
              agg_groups['cond_sum_grp'] = agg_groups['cond_sum_metric']
          # Debugging información de clustering
          valid_values = agg_groups['cond_sum_grp'].notna().sum()
          unique_values = agg_groups['cond_sum_grp'].dropna().nunique()
          st.write(f"**Debug clustering**: Valores válidos: {valid_values}, Únicos: {unique_values}")
          
          # NUEVA METODOLOGÍA: Clustering por variedad-temporada con bandas específicas
          st.markdown("#### 🎯 Nueva metodología de clustering por variedad-temporada")
          
          # Paso 1: Crear agregación correcta por variedad-temporada
          # Primero calcular promedios por muestra individual (por fruto)
          variety_season_cols = [ESPECIE_COLUMN, VAR_COLUMN, 'harvest_period']
          fruit_cols = variety_season_cols + [FRUTO_COLUMN]
          
          # Agregación por fruto individual (promedio de sus muestras)
          fruit_averages = (
              df_processed
              .groupby(fruit_cols, dropna=False)
              .agg(
                  muestras_fruto=("cond_sum", "size"),
                  promedio_brix_fruto=(COL_BRIX, "mean"),
                  promedio_acidez_fruto=(COL_ACIDEZ, "mean"), 
                  firmeza_min_fruto=("Firmeza punto valor", "min"),
                  firmeza_mode_fruto=("Firmeza punto valor", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
                  mejillas_promedio_fruto=("avg_mejillas", "mean"),
              )
              .reset_index()
          )
          
          st.write(f"**Debug**: Calculados promedios para {len(fruit_averages)} frutos individuales")
          
          # Luego agregar por variedad-temporada (promedio de los promedios de frutos)
          variety_season_agg = (
              fruit_averages
              .groupby(variety_season_cols, dropna=False)
              .agg(
                  frutos_total=(FRUTO_COLUMN, "nunique"),
                  muestras_total=("muestras_fruto", "sum"),
                  promedio_brix_var=("promedio_brix_fruto", "mean"),  # Promedio de promedios
                  promedio_acidez_var=("promedio_acidez_fruto", "mean"), 
                  firmeza_min_var=("firmeza_min_fruto", "min"),       # Mínimo de mínimos
                  firmeza_mode_var=("firmeza_mode_fruto", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
                  mejillas_promedio_var=("mejillas_promedio_fruto", "mean"),
              )
              .reset_index()
          )
          
          st.write(f"**Debug**: Creadas {len(variety_season_agg)} combinaciones variedad-temporada")
          
          # Función para asignar bandas basada en quartiles
          def assign_bands(values, metric_name):
              """Asigna bandas de 1-4 basadas en quartiles (1=mejor, 4=peor)"""
              valid_values = values.dropna()
              if len(valid_values) == 0:
                  return pd.Series(0, index=values.index)  # Asignar 0 si no hay valores
              elif len(valid_values) == 1:
                  return pd.Series(1, index=values.index)  # Un solo valor = banda 1
              
              try:
                  # Para BRIX y métricas positivas: mayor valor = mejor banda (1)
                  # Para firmeza: menor valor = mejor banda (1)
                  if metric_name in ['brix']:
                      # Mayor es mejor: invertir quartiles
                      bands = pd.qcut(valid_values, 4, labels=[4,3,2,1], duplicates='drop')
                  else:  # firmeza - menor es mejor
                      bands = pd.qcut(valid_values, 4, labels=[1,2,3,4], duplicates='drop')
                  
                  # Crear serie completa con 0 para valores faltantes
                  result = pd.Series(0, index=values.index)
                  result.loc[valid_values.index] = bands
                  return result
              except ValueError:
                  # Si qcut falla, usar cut o asignar banda 1
                  try:
                      if metric_name in ['brix']:
                          bands = pd.cut(valid_values, 4, labels=[4,3,2,1])
                      else:
                          bands = pd.cut(valid_values, 4, labels=[1,2,3,4])
                      result = pd.Series(0, index=values.index)
                      result.loc[valid_values.index] = bands
                      return result
                  except:
                      # Fallback: todos los valores válidos = banda 1
                      result = pd.Series(0, index=values.index)
                      result.loc[valid_values.index] = 1
                      return result
          
          # Paso 2: Calcular bandas para cada métrica
          variety_season_agg['banda_brix'] = assign_bands(variety_season_agg['promedio_brix_var'], 'brix')
          
          # Para firmeza: elegir entre mínimo o moda
          firmeza_method = st.selectbox(
              "Método para calcular Firmeza en agregación variedad-temporada:",
              options=['mínimo', 'moda'],
              index=0,
              key="firmeza_method_var_season",
              help="Método a usar para agregar valores de firmeza por variedad-temporada"
          )
          
          if firmeza_method == 'mínimo':
              variety_season_agg['banda_firmeza'] = assign_bands(variety_season_agg['firmeza_min_var'], 'firmeza')
              variety_season_agg['firmeza_metodo_usado'] = 'mínimo'
          else:
              variety_season_agg['banda_firmeza'] = assign_bands(variety_season_agg['firmeza_mode_var'], 'firmeza')
              variety_season_agg['firmeza_metodo_usado'] = 'moda'
          
          # Asignar banda para acidez si existe
          variety_season_agg['banda_acidez'] = assign_bands(variety_season_agg['promedio_acidez_var'], 'acidez') if 'promedio_acidez_var' in variety_season_agg.columns else 0
          
          # Paso 3: Calcular suma de bandas y cluster global
          metrics_used = []
          variety_season_agg['suma_bandas'] = 0
          
          if variety_season_agg['banda_brix'].sum() > 0:
              variety_season_agg['suma_bandas'] += variety_season_agg['banda_brix']
              metrics_used.append('BRIX')
              
          if variety_season_agg['banda_firmeza'].sum() > 0:
              variety_season_agg['suma_bandas'] += variety_season_agg['banda_firmeza']
              metrics_used.append('Firmeza')
              
          if variety_season_agg['banda_acidez'].sum() > 0:
              variety_season_agg['suma_bandas'] += variety_season_agg['banda_acidez']
              metrics_used.append('Acidez')
          
          # Calcular cluster global basado en la suma
          max_possible_sum = len(metrics_used) * 4  # Máximo posible
          
          def assign_global_cluster(suma_bandas, max_sum):
              """Asigna cluster global basado en la suma de bandas según rangos específicos"""
              if suma_bandas == 0:
                  return 4  # Sin datos = peor cluster
              # Ajustar rangos según especificación del usuario
              # Para 3 métricas: máximo = 12, rangos: 3-5=cluster1, 6-8=cluster2, 9-11=cluster3, 12+=cluster4
              # Para 2 métricas: máximo = 8, rangos: 2-3=cluster1, 4-5=cluster2, 6-7=cluster3, 8+=cluster4
              if max_sum == 12:  # 3 métricas (BRIX, Firmeza, Acidez)
                  if suma_bandas <= 5:
                      return 1  # Excelente (3-5)
                  elif suma_bandas <= 8:
                      return 2  # Bueno (6-8)
                  elif suma_bandas <= 11:
                      return 3  # Regular (9-11)
                  else:
                      return 4  # Deficiente (12)
              elif max_sum == 8:  # 2 métricas (BRIX, Firmeza)
                  if suma_bandas <= 3:
                      return 1  # Excelente (2-3)
                  elif suma_bandas <= 5:
                      return 2  # Bueno (4-5)
                  elif suma_bandas <= 7:
                      return 3  # Regular (6-7)
                  else:
                      return 4  # Deficiente (8)
              else:  # Fallback genérico para otros casos
                  if suma_bandas <= max_sum * 0.4:
                      return 1  # Excelente
                  elif suma_bandas <= max_sum * 0.6:
                      return 2  # Bueno
                  elif suma_bandas <= max_sum * 0.85:
                      return 3  # Regular
                  else:
                      return 4  # Deficiente
          
          variety_season_agg['cluster_variedad_temporada'] = variety_season_agg['suma_bandas'].apply(
              lambda x: assign_global_cluster(x, max_possible_sum)
          )
          
          # Mostrar información sobre el clustering
          st.write(f"**Métricas utilizadas**: {', '.join(metrics_used)}")
          st.write(f"**Suma máxima posible**: {max_possible_sum}")
          st.write(f"**Distribución de clusters por variedad-temporada**:")
          cluster_dist = variety_season_agg['cluster_variedad_temporada'].value_counts().sort_index()
          for cluster, count in cluster_dist.items():
              st.write(f"- Cluster {cluster}: {count} combinaciones variedad-temporada")
          
          # Mostrar tabla de bandas por variedad-temporada
          if st.checkbox("Mostrar tabla de bandas por variedad-temporada", key="show_variety_bands"):
              display_cols_var = variety_season_cols + [
                  'frutos_total', 'muestras_total', 
                  'promedio_brix_var', 'banda_brix',
                  'firmeza_min_var', 'banda_firmeza', 'firmeza_metodo_usado',
                  'suma_bandas', 'cluster_variedad_temporada'
              ]
              if 'banda_acidez' in variety_season_agg.columns and variety_season_agg['banda_acidez'].sum() > 0:
                  display_cols_var.insert(-2, 'promedio_acidez_var')
                  display_cols_var.insert(-2, 'banda_acidez')
              
              st.markdown("**Explicación del cálculo:**")
              st.write("1. **Frutos individuales**: Se calcula el promedio BRIX por cada fruto (de sus muestras)")
              st.write("2. **Variedad-temporada**: Se promedia los BRIX de todos los frutos de esa combinación")
              st.write("3. **Bandas**: Se asignan cuartiles 1-4 donde 1=mejor, 4=peor")
              st.write("4. **Cluster**: Suma de bandas con rangos específicos")
              
              st.dataframe(variety_season_agg[display_cols_var], use_container_width=True)
              
              # Mostrar tabla de frutos individuales si se solicita
              if st.checkbox("Ver promedios por fruto individual", key="show_fruit_averages"):
                  st.markdown("**Promedios por fruto individual:**")
                  fruit_display_cols = fruit_cols + [
                      'muestras_fruto', 'promedio_brix_fruto', 'promedio_acidez_fruto', 
                      'firmeza_min_fruto', 'mejillas_promedio_fruto'
                  ]
                  st.dataframe(fruit_averages[fruit_display_cols], use_container_width=True)
          
          # Paso 4: Hacer merge de vuelta a agg_groups (nivel fruto)
          merge_cols_back = [ESPECIE_COLUMN, VAR_COLUMN, 'harvest_period']
          agg_groups = agg_groups.merge(
              variety_season_agg[merge_cols_back + ['cluster_variedad_temporada', 'suma_bandas', 'banda_brix', 'banda_firmeza']],
              on=merge_cols_back,
              how='left'
          )
          
          # Asignar el cluster basado en la agregación variedad-temporada
          agg_groups['cluster_grp'] = agg_groups['cluster_variedad_temporada'].fillna(4).astype(int)

          # Determinar el punto de firmeza más bajo por cluster y repetirlo
          agg_groups['punto_firmeza_cluster'] = agg_groups['punto_firmeza_min']
          for clus, grp in agg_groups.groupby('cluster_grp'):
              if pd.isna(clus):
                  continue
              idx_min = grp['promedio_firmeza_punto'].idxmin(skipna=True)
              weak_pt = agg_groups.loc[idx_min, 'punto_firmeza_min']
              agg_groups.loc[agg_groups['cluster_grp'] == clus, 'punto_firmeza_cluster'] = weak_pt

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
              return ['' for _ in row]

          # Controles de visualización al principio
          st.markdown("### 📊 Opciones de Visualización")
          show_variety_table = st.checkbox("Mostrar tabla agregados por variedad", value=False, key="show_variety")
          show_group_table = st.checkbox("Mostrar tabla agregados por grupo", value=False, key="show_group")
          
          # Selector de métricas para clasificación
          st.markdown("### ⚙️ Configuración de Clasificación")
          all_numeric_cols = [col for col in agg_groups.columns if pd.api.types.is_numeric_dtype(agg_groups[col]) and not col.startswith('PC') and col != 'cluster_grp']
          
          selected_classification_metrics = st.multiselect(
              "Selecciona métricas para la clasificación:",
              options=all_numeric_cols,
              default=all_numeric_cols[:4] if len(all_numeric_cols) >= 4 else all_numeric_cols,
              key="classification_metrics",
              help="Estas métricas se usarán para calcular los clusters de clasificación"
          )
          
          if selected_classification_metrics and selected_classification_metrics != available_metrics:
              st.info("💡 Las métricas seleccionadas se aplicarán en la próxima ejecución de segmentación")

          # Mostrar siempre la agrupación por reglas y, si existe, comparar con
          # la clusterización automática ejecutada (KMeans u otro algoritmo).
          # Solo se muestran las columnas consideradas relevantes para el
          # usuario final - Solo columnas esenciales para el análisis
          display_cols = [
              VAR_COLUMN,
              "cluster_grp",
              "promedio_brix",
              "promedio_acidez",
              "promedio_firmeza_punto",
          ]
          
          # Mostrar tabla de agregados por grupo - condicional
          if show_group_table:
              if "cluster_auto" in agg_groups.columns:
                  st.markdown("#### Comparación de clusters (reglas vs automático)")
                  display_cols.append("cluster_auto")
                  subset_cols = ["cluster_grp", "cluster_auto"]
              else:
                  st.markdown("#### Segmentación por reglas")
                  subset_cols = ["cluster_grp"]

              styled_agg = (
                  agg_groups[display_cols]
                  .style
                  .map(color_cluster, subset=[c for c in subset_cols if c in display_cols]))
              styled_agg = (
                  agg_groups.style
                  .map(color_cluster, subset=["cluster_grp"])

                  .apply(highlight_inconsistent, axis=1)
              )
              st.dataframe(styled_agg, use_container_width=True, height=400)

          # Gráfico de las combinaciones en dos dimensiones (PCA)
          st.markdown("#### Distribución PCA de los grupos")
          try:
              # Importaciones necesarias para el gráfico PCA
              from sklearn.preprocessing import StandardScaler
              from sklearn.decomposition import PCA
              # altair ya está importado al inicio del archivo
              
              # Verificar que tenemos datos y las columnas necesarias
              if len(agg_groups) == 0:
                  st.warning("No hay datos suficientes para generar el gráfico PCA.")
                  st.stop()
              
              # Verificar columnas disponibles
              available_cols = agg_groups.columns.tolist()
              st.write(f"**Debug**: Columnas disponibles en agg_groups: {available_cols[:10]}...")  # Solo las primeras 10
              
              # Seleccionar características numéricas disponibles
              potential_features = [
                  "promedio_cond_sum",
                  "promedio_brix", 
                  "promedio_acidez",
                  "promedio_firmeza_punto",
                  "promedio_mejillas",
              ]
              pca_features = [col for col in potential_features if col in agg_groups.columns]
              
              if len(pca_features) < 2:
                  st.warning(f"No se encontraron suficientes características numéricas para PCA. Disponibles: {pca_features}")
                  st.stop()
              
              st.write(f"**Debug**: Usando características para PCA: {pca_features}")
              
              # Verificar datos no vacíos
              df_features = agg_groups[pca_features].fillna(0)
              if df_features.empty or df_features.isna().all().all():
                  st.warning("Todas las características están vacías o son NaN.")
                  st.stop()
              
              # Verificar cluster_grp
              if 'cluster_grp' not in agg_groups.columns:
                  st.warning("Columna 'cluster_grp' no encontrada.")
                  agg_groups['cluster_grp'] = 1  # Valor por defecto
              
              # Definir colores para grupos 1-4 (usando colores de metricas_bandas.py)
              group_colors = {
                  1: '#a8e6cf',  # verde claro - Excelente
                  2: '#ffd3b6',  # naranja claro - Bueno
                  3: '#ffaaa5',  # coral - Regular
                  4: '#ff8b94',  # rojo rosado - Deficiente
              }
              
              # Mostrar leyenda de colores de clusters
              st.markdown("#### Interpretación de Clusters")
              col1, col2, col3, col4 = st.columns(4)
              with col1:
                  st.markdown(f'<div style="background-color: {group_colors[1]}; padding: 10px; border-radius: 5px; text-align: center; color: black; font-weight: bold;">Cluster 1: Excelente</div>', unsafe_allow_html=True)
              with col2:
                  st.markdown(f'<div style="background-color: {group_colors[2]}; padding: 10px; border-radius: 5px; text-align: center; color: black; font-weight: bold;">Cluster 2: Bueno</div>', unsafe_allow_html=True)
              with col3:
                  st.markdown(f'<div style="background-color: {group_colors[3]}; padding: 10px; border-radius: 5px; text-align: center; color: black; font-weight: bold;">Cluster 3: Regular</div>', unsafe_allow_html=True)
              with col4:
                  st.markdown(f'<div style="background-color: {group_colors[4]}; padding: 10px; border-radius: 5px; text-align: center; color: black; font-weight: bold;">Cluster 4: Deficiente</div>', unsafe_allow_html=True)
              
              # Normalizamos
              scaler = StandardScaler()
              X_scaled = scaler.fit_transform(df_features)
              pca = PCA(n_components=2)
              pcs = pca.fit_transform(X_scaled)
              agg_groups["PC1"] = pcs[:, 0]
              agg_groups["PC2"] = pcs[:, 1]
              
              # Verificar que tenemos clusters válidos
              unique_clusters = agg_groups['cluster_grp'].dropna().unique()
              st.write(f"**Debug**: Clusters únicos: {unique_clusters}")
              
              # Construir gráfico interactivo con Altair
              valid_clusters = [c for c in unique_clusters if not pd.isna(c)]
              if len(valid_clusters) == 0:
                  st.warning("No hay clusters válidos para mostrar. Mostrando PCA sin colorear por cluster.")
                  # Crear gráfico sin clusters
                  chart = (
                      alt.Chart(agg_groups.dropna(subset=['PC1', 'PC2']))
                      .mark_circle(size=80, color='steelblue')
                      .encode(
                          x=alt.X("PC1", title="Componente principal 1"),
                          y=alt.Y("PC2", title="Componente principal 2"),
                          tooltip=[col for col in [ESPECIE_COLUMN, VAR_COLUMN, FRUTO_COLUMN, 'harvest_period'] if col in agg_groups.columns]
                      )
                      .properties(width='container', height=400)
                  )
                  st.altair_chart(chart, use_container_width=True)
                  return agg_groups
              
              color_domain = list(valid_clusters)
              color_range = [group_colors.get(int(c), '#cccccc') for c in valid_clusters]
              
              color_scale = alt.Scale(domain=color_domain, range=color_range)
              chart = (
                  alt.Chart(agg_groups.dropna(subset=['PC1', 'PC2', 'cluster_grp']))
                  .mark_circle(size=80)
                  .encode(
                      x=alt.X("PC1", title="Componente principal 1"),
                      y=alt.Y("PC2", title="Componente principal 2"), 
                      color=alt.Color("cluster_grp:N", scale=color_scale, legend=alt.Legend(title="Cluster")),
                      tooltip=[col for col in [ESPECIE_COLUMN, VAR_COLUMN, FRUTO_COLUMN, 'harvest_period', 'cluster_grp'] if col in agg_groups.columns]
                  )
                  .properties(width='container', height=400)
                  .interactive()
              )
              st.altair_chart(chart, use_container_width=True)
              st.success("Gráfico PCA generado exitosamente!")
              
              # Agregar gráfico por cluster de datos agregados
              st.markdown("#### Análisis por Cluster")
              if len(valid_clusters) > 0:
                  
                  # Selector de métrica para visualizar por cluster
                  available_numeric_cols = [col for col in agg_groups.columns if pd.api.types.is_numeric_dtype(agg_groups[col]) and col not in ['PC1', 'PC2', 'cluster_grp']]
                  selected_metric = st.selectbox(
                      "Métrica a visualizar por cluster:",
                      options=available_numeric_cols,
                      index=0 if available_numeric_cols else None,
                      key="cluster_metric_viz"
                  )
                  
                  if selected_metric:
                      # Gráfico de barras por cluster
                      cluster_summary = agg_groups.groupby('cluster_grp')[selected_metric].agg(['mean', 'count']).reset_index()
                      cluster_summary.columns = ['Cluster', 'Promedio', 'Cantidad']
                      
                      fig_cluster = alt.Chart(cluster_summary).mark_bar().encode(
                          x=alt.X('Cluster:O', title='Cluster'),
                          y=alt.Y('Promedio:Q', title=f'Promedio de {selected_metric}'),
                          color=alt.Color('Cluster:O', scale=alt.Scale(range=[group_colors.get(int(c), '#cccccc') for c in valid_clusters])),
                          tooltip=['Cluster:O', 'Promedio:Q', 'Cantidad:Q']
                      ).properties(
                          width='container',
                          height=300,
                          title=f'{selected_metric} por Cluster'
                      )
                      
                      st.altair_chart(fig_cluster, use_container_width=True)
                      
                      # Tabla resumen por cluster
                      if st.checkbox("Mostrar resumen estadístico por cluster", key="show_cluster_stats"):
                          cluster_detailed = agg_groups.groupby('cluster_grp')[available_numeric_cols].agg(['mean', 'std', 'count']).round(2)
                          st.markdown("**Estadísticas por cluster:**")
                          st.dataframe(cluster_detailed, use_container_width=True)
                      
                      # Visualización K-means
                      st.markdown("#### Clustering K-means Comparativo")
                      if st.checkbox("Mostrar análisis K-means automático", key="show_kmeans"):
                          try:
                              from sklearn.cluster import KMeans
                              from sklearn.metrics import silhouette_score
                              
                              # Seleccionar número de clusters para K-means
                              n_clusters_kmeans = st.slider("Número de clusters K-means:", 2, 6, len(valid_clusters), key="n_clusters_kmeans")
                              
                              # Usar las métricas seleccionadas por el usuario para K-means
                              # Mapear nombres de métricas seleccionadas a columnas en agg_groups
                              metric_to_col_map = {
                                  "BRIX": "promedio_brix",
                                  "Acidez (%)": "promedio_acidez", 
                                  "Firmeza punto débil": "promedio_firmeza_punto",
                                  "Mejillas": "promedio_mejillas"
                              }
                              
                              selected_cols_kmeans = []
                              for metric in selected_metrics:
                                  if metric in metric_to_col_map and metric_to_col_map[metric] in agg_groups.columns:
                                      selected_cols_kmeans.append(metric_to_col_map[metric])
                              
                              if not selected_cols_kmeans:
                                  st.warning("No se encontraron columnas válidas para K-means con las métricas seleccionadas")
                              else:
                                  st.write(f"**Debug K-means**: Usando columnas: {selected_cols_kmeans}")
                                  X_kmeans = agg_groups[selected_cols_kmeans].fillna(0)
                              
                                  if len(X_kmeans) >= n_clusters_kmeans:
                                      # Aplicar K-means
                                      kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42)
                                      kmeans_labels = kmeans.fit_predict(X_kmeans)
                                      
                                      # Calcular silhouette score
                                      sil_score = silhouette_score(X_kmeans, kmeans_labels) if len(set(kmeans_labels)) > 1 else 0
                                      
                                      # Mostrar métricas
                                      col1, col2 = st.columns(2)
                                      with col1:
                                          st.metric("Silhouette Score K-means", f"{sil_score:.3f}")
                                      with col2:
                                          st.metric("Inercia K-means", f"{kmeans.inertia_:.3f}")
                                      
                                      # Crear DataFrame con resultados K-means
                                      agg_kmeans = agg_groups.copy()
                                      agg_kmeans['cluster_kmeans'] = kmeans_labels
                                      
                                      # Gráfico PCA con clusters K-means
                                      kmeans_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:n_clusters_kmeans]
                                      
                                      fig_kmeans = alt.Chart(agg_kmeans.dropna(subset=['PC1', 'PC2'])).mark_circle(size=80).encode(
                                          x=alt.X("PC1", title="Componente principal 1"),
                                          y=alt.Y("PC2", title="Componente principal 2"),
                                          color=alt.Color("cluster_kmeans:O", 
                                                        scale=alt.Scale(range=kmeans_colors),
                                                        legend=alt.Legend(title="Cluster K-means")),
                                          tooltip=[col for col in [VAR_COLUMN, 'cluster_grp', 'cluster_kmeans'] if col in agg_kmeans.columns]
                                      ).properties(
                                          width='container',
                                          height=400,
                                          title=f'PCA con K-means (k={n_clusters_kmeans})'
                                      ).interactive()
                                      
                                      st.altair_chart(fig_kmeans, use_container_width=True)
                                      
                                      # Comparación entre clusters por reglas y K-means
                                      if st.checkbox("Comparar clusters: Reglas vs K-means", key="compare_clusters"):
                                          comparison_df = agg_kmeans.groupby(['cluster_grp', 'cluster_kmeans']).size().reset_index(name='count')
                                          comparison_pivot = comparison_df.pivot(index='cluster_grp', columns='cluster_kmeans', values='count').fillna(0)
                                          
                                          st.markdown("**Matriz de confusión: Clusters por reglas vs K-means**")
                                          st.dataframe(comparison_pivot, use_container_width=True)
                                  else:
                                      st.warning("No hay suficientes datos para el número de clusters seleccionado.")
                          except Exception as e:
                              st.error(f"Error en análisis K-means: {e}")
              
          except Exception as e:
              st.error(f"Error generando el gráfico PCA: {e}")
              import traceback
              st.text("Traceback:")
              st.code(traceback.format_exc())

          # Datos por muestra individual con clustering
          st.markdown("### Datos por muestra individual con clustering")
          if st.checkbox("Mostrar datos individuales por muestra", key="show_individual_samples"):
              try:
                  # Crear datos individuales con clusters asignados
                  # Usar df_processed que contiene los datos originales por muestra
                  df_individual = df_processed.copy()
                  
                  # Agregar información de cluster para cada muestra
                  # Hacer merge con agg_groups para obtener cluster_grp
                  merge_cols = [col for col in [ESPECIE_COLUMN, VAR_COLUMN, FRUTO_COLUMN, 'harvest_period'] if col in df_individual.columns and col in agg_groups.columns]
                  
                  # Verificar que cluster_grp existe en agg_groups
                  if 'cluster_grp' not in agg_groups.columns:
                      st.error("Error: columna 'cluster_grp' no encontrada en agg_groups")
                      df_individual['cluster_grp'] = 1
                  elif merge_cols and len(merge_cols) >= 2:  # Necesitamos al menos 2 columnas para merge
                      try:
                          st.write(f"**Debug merge**: Usando columnas: {merge_cols}")
                          df_individual = df_individual.merge(
                              agg_groups[merge_cols + ['cluster_grp']],
                              on=merge_cols,
                              how='left'
                          )
                          # Rellenar valores NaN con cluster 4 (deficiente)
                          df_individual['cluster_grp'] = df_individual['cluster_grp'].fillna(4)
                      except Exception as e:
                          st.error(f"Error en merge: {e}")
                          df_individual['cluster_grp'] = 1
                  else:
                      # Si no hay suficientes columnas para merge, asignar cluster 1 a todos
                      st.warning("No hay suficientes columnas para hacer merge, asignando cluster 1 a todas las muestras")
                      df_individual['cluster_grp'] = 1
                  
                  # Mostrar estadísticas
                  total_samples = len(df_individual)
                  samples_with_cluster = df_individual['cluster_grp'].notna().sum()
                  st.write(f"**Total muestras individuales**: {total_samples}")
                  st.write(f"**Muestras con cluster asignado**: {samples_with_cluster}")
                  
                  # Mostrar distribución de clusters a nivel de muestra
                  cluster_dist = df_individual['cluster_grp'].value_counts().sort_index()
                  st.write("**Distribución de muestras por cluster:**")
                  for cluster, count in cluster_dist.items():
                      st.write(f"- Cluster {cluster}: {count} muestras")
                  
                  # Seleccionar columnas relevantes para mostrar
                  display_cols_individual = [col for col in [ESPECIE_COLUMN, VAR_COLUMN, FRUTO_COLUMN, 'harvest_period'] if col in df_individual.columns]
                  metric_cols_individual = []
                  
                  # Agregar columnas de métricas seleccionadas
                  for metric in selected_metrics:
                      if metric == "BRIX" and "Solidos solubles (%)" in df_individual.columns:
                          metric_cols_individual.append("Solidos solubles (%)")
                      elif metric == "Acidez (%)" and "Acidez (%)" in df_individual.columns:
                          metric_cols_individual.append("Acidez (%)")
                      elif metric == "Firmeza punto débil" and "firmeza_punto_min" in df_individual.columns:
                          metric_cols_individual.append("firmeza_punto_min")
                      elif metric == "Mejillas" and "promedio_mejillas_sample" in df_individual.columns:
                          metric_cols_individual.append("promedio_mejillas_sample")
                  
                  display_cols_individual.extend(metric_cols_individual)
                  display_cols_individual.append('cluster_grp')
                  
                  # Filtrar columnas que existen
                  display_cols_individual = [col for col in display_cols_individual if col in df_individual.columns]
                  
                  st.write(f"**Debug individual**: Columnas disponibles: {list(df_individual.columns)[:10]}...")
                  st.write(f"**Debug individual**: Columnas a mostrar: {display_cols_individual}")
                  
                  # Mostrar tabla con datos individuales
                  if display_cols_individual:
                      st.dataframe(
                          df_individual[display_cols_individual].head(50),
                          use_container_width=True
                      )
                  else:
                      st.warning("No se encontraron columnas válidas para mostrar datos individuales")
                      
              except Exception as e:
                  st.error(f"Error mostrando datos individuales: {e}")
                  import traceback
                  st.code(traceback.format_exc())

          # Mostrar agregados por variedad (sin separar fruto ni periodo) - condicional
          if show_variety_table:
              st.markdown("### Agregados por variedad")
              # Solo columnas esenciales para el análisis
              cols_variedad = [
                  VAR_COLUMN,
                  "cluster_grp",
                  "promedio_brix",
                  "promedio_acidez",
                  "promedio_firmeza_punto",
                  "muestras",
              ]
              st.dataframe(agg_variedad[cols_variedad], use_container_width=True, height=300)
          
          # Comparación entre cálculos a nivel de fruto vs variedad
          st.markdown("### Comparación: Cálculos por Fruto vs por Variedad")
          if st.checkbox("Mostrar comparación de cálculos", key="show_calculation_comparison"):
              try:
                  # Comparar los promedios entre agg_groups (por fruto) y agg_variedad (por variedad)
                  comparison_metrics = ['promedio_brix', 'promedio_acidez', 'promedio_firmeza_punto']
                  
                  # Crear tabla de comparación
                  comparison_data = []
                  for var in agg_variedad[VAR_COLUMN].unique():
                      if pd.notna(var):
                          # Datos por fruto (agg_groups) - promedio de la variedad
                          fruto_data = agg_groups[agg_groups[VAR_COLUMN] == var]
                          
                          # Datos por variedad (agg_variedad)
                          var_data = agg_variedad[agg_variedad[VAR_COLUMN] == var]
                          
                          if len(fruto_data) > 0 and len(var_data) > 0:
                              row = {'Variedad': var}
                              for metric in comparison_metrics:
                                  if metric in fruto_data.columns and metric in var_data.columns:
                                      fruto_avg = fruto_data[metric].mean()
                                      var_avg = var_data[metric].iloc[0]
                                      
                                      row[f'{metric}_fruto'] = round(fruto_avg, 2)
                                      row[f'{metric}_variedad'] = round(var_avg, 2)
                                      row[f'{metric}_diferencia'] = round(abs(fruto_avg - var_avg), 3)
                              
                              comparison_data.append(row)
                  
                  if comparison_data:
                      comparison_df = pd.DataFrame(comparison_data)
                      st.write("**Comparación de promedios: Agregación por fruto vs por variedad**")
                      st.write("- `_fruto`: Promedio de los grupos por fruto")
                      st.write("- `_variedad`: Cálculo directo por variedad")
                      st.write("- `_diferencia`: Diferencia absoluta entre ambos métodos")
                      st.dataframe(comparison_df, use_container_width=True)
                      
                      # Mostrar estadísticas de diferencias
                      st.write("**Estadísticas de diferencias:**")
                      for metric in comparison_metrics:
                          diff_col = f'{metric}_diferencia'
                          if diff_col in comparison_df.columns:
                              avg_diff = comparison_df[diff_col].mean()
                              max_diff = comparison_df[diff_col].max()
                              st.write(f"- {metric}: Diferencia promedio = {avg_diff:.3f}, Máxima = {max_diff:.3f}")
                  else:
                      st.warning("No se pudieron calcular las comparaciones")
                      
              except Exception as e:
                  st.error(f"Error en comparación de cálculos: {e}")
          
          # Verificación del tratamiento de acidez con primer registro
          st.markdown("### Verificación: Tratamiento de Acidez")
          if st.checkbox("Verificar tratamiento de acidez (primer registro)", key="show_acidez_verification"):
              try:
                  # Buscar en grp_acidez como se está manejando
                  if 'grp_acidez' in df_processed.columns:
                      acidez_sample = df_processed.groupby([ESPECIE_COLUMN, VAR_COLUMN, 'harvest_period']).agg({
                          'grp_acidez': ['first', 'count'],
                          'Acidez (%)': ['mean', 'count'] if 'Acidez (%)' in df_processed.columns else ['count']
                      }).round(3)
                      
                      st.write("**Verificación del manejo de grp_acidez:**")
                      st.write("- Se usa el primer valor (`first`) por combinación especie-variedad-periodo")
                      st.write("- `count` muestra cuántos registros hay por grupo")
                      st.dataframe(acidez_sample.head(10), use_container_width=True)
                  else:
                      st.warning("Columna 'grp_acidez' no encontrada")
                      
                  # Mostrar información sobre cómo se están tratando los valores nulos
                  st.write("**Información sobre tratamiento de valores nulos:**")
                  null_info = []
                  for col in ['grp_acidez', 'Acidez (%)']:
                      if col in df_processed.columns:
                          null_count = df_processed[col].isna().sum()
                          total_count = len(df_processed)
                          null_info.append({
                              'Columna': col,
                              'Valores nulos': null_count,
                              'Total registros': total_count,
                              'Porcentaje nulos': f"{(null_count/total_count)*100:.1f}%"
                          })
                  
                  if null_info:
                      st.dataframe(pd.DataFrame(null_info), use_container_width=True)
                      
              except Exception as e:
                  st.error(f"Error en verificación de acidez: {e}")
          
          # Botón para descargar resultados completos y agregados
          buf = io.BytesIO()
          with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
              df_processed.to_excel(writer, index=False, sheet_name='Carozos')
              agg_groups.to_excel(writer, index=False, sheet_name='Agregados_grupo')
              agg_variedad.to_excel(writer, index=False, sheet_name='Agregados_variedad')
          buf.seek(0)
          
          # Guardar datos agregados y procesados en session_state según la especie
          if especie_key == "Ciruela":
              st.session_state["agg_groups_plum"] = agg_groups
              st.session_state["df_processed_plum"] = df_processed
          else:  # Nectarin
              st.session_state["agg_groups_nect"] = agg_groups
              st.session_state["df_processed_nect"] = df_processed
          
          st.download_button(
              label="📥 Descargar resultados como Excel",
              data=buf.getvalue(),
              file_name="carozos_procesados.xlsx",
              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
          )
