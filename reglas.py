# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

# Suppress specific pandas warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# =============== Parámetros ===============
RUTA_EXCEL = Path(r"C:\Users\bherr\APP SEGMENTACIÓN\APP_SEGMENTACION_ESPECIES\MAESTRO CAROZO.xlsx")
NOMBRE_HOJA = "CAROZOS"
MODO_COND_SUM = "suma"     # "suma" | "media"
RUTA_SALIDA = Path(r"C:\Users\bherr\APP SEGMENTACIÓN\APP_SEGMENTACION_ESPECIES\validacion_calculos.xlsx")
UMBRAL_PESO_CANDY = 60.0   # gramos; >60 => Candy plum, si no Cherry plum

# Rangos válidos (ajústalos si aplica)
RANGOS_VALIDOS = {
    "brix": (0.0, 35.0),
    "mejillas": (0.0, 20.0),
    "firmeza_puntos": (0.0, 20.0),  # se aplica a prom_quilla/hombro/punta y fpunto_debil_valor
    "acidez": (0.0, 2.0),
    "peso": (0.0, 300.0),
}

# =============== Utilidades ===============
def _pick(df, candidates, required=False):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"No encontré ninguna de estas columnas: {candidates}")
    return None

def _to_num(s):
    return (
        s.astype(str)
         .str.replace(",", ".", regex=False)
         .str.replace("−", "-", regex=False)
         .pipe(pd.to_numeric, errors="coerce")
    )

def _parse_fecha(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")
    s1 = pd.to_datetime(s, dayfirst=True, errors="coerce")
    s2 = pd.to_datetime(s, dayfirst=False, errors="coerce")
    return s1.fillna(s2)

def first_non_null(s):
    idx = s.first_valid_index()
    return s.loc[idx] if idx is not None else np.nan

def _in_range(val, lo, hi):
    return pd.notna(val) and (lo <= val < hi)

# =============== Bandas (ajusta si aplica) ===============
PLUM_RULES = {
    "candy": {
        "BRIX": [(18.0, np.inf, 1), (16.0, 18.0, 2), (14.0, 16.0, 3), (-np.inf, 14.0, 4)],
        "FIRMEZA_PUNTO": [(7.0, np.inf, 1), (5.0, 7.0, 2), (4.0, 5.0, 3), (-np.inf, 4.0, 4)],
        "FIRMEZA_MEJ":   [(9.0, np.inf, 1), (7.0, 9.0, 2), (6.0, 7.0, 3), (-np.inf, 6.0, 4)],
        "ACIDEZ":        [(-np.inf, 0.60, 1), (0.60, 0.81, 2), (0.81, 1.00, 3), (1.00, np.inf, 4)],
    },
    "sugar": {
        "BRIX": [(21.0, np.inf, 1), (18.0, 21.0, 2), (15.0, 18.0, 3), (-np.inf, 15.0, 4)],
        "FIRMEZA_PUNTO": [(6.0, np.inf, 1), (4.5, 6.0, 2), (3.0, 4.5, 3), (-np.inf, 3.0, 4)],
        "FIRMEZA_MEJ":   [(6.0, np.inf, 1), (5.0, 6.0, 2), (4.0, 5.0, 3), (-np.inf, 4.0, 4)],
        "ACIDEZ":        [(-np.inf, 0.60, 1), (0.60, 0.81, 2), (0.81, 1.00, 3), (1.00, np.inf, 4)],
    },
}

def _mk_nect(b1, b2, b3, m1, m2):
    return {
        # BRIX usa intervalos [lo, hi)
        "BRIX": [(b1, float("inf"), 1), (b2, b1, 2), (b3, b2, 3), (-float("inf"), b3, 4)],
        # Firmeza punto: usa fpunto_debil_valor = min(prom_quilla, prom_hombro, prom_punta)
        "FIRMEZA_PUNTO": [(9.0, float("inf"), 1), (8.0, 9.0, 2), (7.0, 8.0, 3), (-float("inf"), 7.0, 4)],
        # Firmeza mejillas
        "FIRMEZA_MEJ":   [(m1, float("inf"), 1), (m2, m1, 2), (9.0, m2, 3), (-float("inf"), 9.0, 4)],
        # Acidez fija
        "ACIDEZ":        [(-float("inf"), 0.60, 1), (0.60, 0.81, 2), (0.81, 1.00, 3), (1.00, float("inf"), 4)],
    }

NECT_RULES = {
    "amarilla": {
        "muy_temprana": _mk_nect(13.0, 10.0, 9.0, 14.0, 12.0),
        "temprana":     _mk_nect(13.0, 11.0, 9.0, 14.0, 12.0),  # BRIX 13/11/9
        "tardia":       _mk_nect(14.0, 12.0, 10.0, 14.0, 12.0),
    },
    "blanca": {
        "muy_temprana": _mk_nect(13.0, 10.0, 9.0, 13.0, 11.0),
        "temprana":     _mk_nect(13.0, 11.0, 9.0, 13.0, 11.0),  # BRIX 13/11/9
        "tardia":       _mk_nect(14.0, 12.0, 10.0, 13.0, 11.0),
    },
}

def clasificar(val, bands):
    if pd.isna(val) or not bands:
        return np.nan
    for lo, hi, grp in bands:
        if lo <= val < hi:
            return grp
    return np.nan

def harvest_period(color, fecha):
    fecha = pd.to_datetime(fecha, errors="coerce")
    if pd.isna(fecha):
        return "tardia"
    m, d = fecha.month, fecha.day
    if color == "blanca":
        if (m, d) < (11,25): return "muy_temprana"
        if (11,25) <= (m, d) <= (12,15): return "temprana"
        if (12,16) <= (m, d) <= (2,15):  return "tardia"
    else:
        if (m, d) < (11,22): return "muy_temprana"
        if (11,22) <= (m, d) <= (12,22): return "temprana"
        if (12,23) <= (m, d) <= (2,15):  return "tardia"
    return "tardia"

# =============== Carga ===============
df = pd.read_excel(RUTA_EXCEL, sheet_name=NOMBRE_HOJA)

# =============== Mapeo de columnas (incluye 'temporada' y 'portainjerto') ===============
COLS = {
    "especie":      ["Especie"],
    "variedad":     ["Variedad"],
    "temporada":    ["Temporada", "SEASON", "Season", "TEMPORADA", "Temp"],
    "fecha":        ["Fecha evaluación", "Fecha de evaluación", "Fecha_eval", "Fecha"],
    "campo":        ["Campo", "Huerto", "Predio", "Cuartel", "Sector"],
    "portainjerto": ["Portainjerto", "Porta injerto", "Porta-injerto", "Portainjertos", "Rootstock"],
    "fruto":        ["Fruto (n°)", "Fruto", "N° fruto", "N° frutos", "Fruto Nº"],
    "brix":         ["Solidos solubles (%)", "Sólidos solubles (%)", "Solidos solubles", "Brix", "BRIX"],
    "acidez":       ["Acidez (%)", "Acidez"],
    "mej1":         ["Mejilla 1", "Mejilla1"],
    "mej2":         ["Mejilla 2", "Mejilla2"],
    "quilla":       ["Quilla"],
    "hombro":       ["Hombro"],
    "punta":        ["Punta"],
    "color":        ["Color de pulpa", "Color pulpa"],
    "peso":         ["Peso (g)", "Calibre", "Peso"],
}
c = {k: _pick(df, v, required=(k in ["especie","variedad","fecha","temporada"])) for k,v in COLS.items()}
df.rename(columns={c[k]: k for k in c if c[k] is not None}, inplace=True)

# Guardar bandera de 'campo' nulo original (antes de normalizar)
df["campo_nulo_original"] = df["campo"].isna() | (df["campo"].astype(str).str.strip() == "")

# Tipos y limpieza
for numcol in ["brix","acidez","mej1","mej2","quilla","hombro","punta","peso","fruto"]:
    if numcol in df:
        df[numcol] = _to_num(df[numcol])

df["fecha"] = _parse_fecha(df["fecha"])
df["temporada"] = df["temporada"].astype(str).str.strip()
if "campo" not in df: df["campo"] = "NA"
df.loc[df["campo"].astype(str).str.strip().eq(""), "campo"] = np.nan  # homogeniza vacíos
df["campo"] = df["campo"].fillna("NA")

if "portainjerto" not in df: df["portainjerto"] = "NA"
df.loc[df["portainjerto"].astype(str).str.strip().eq(""), "portainjerto"] = np.nan
df["portainjerto"] = df["portainjerto"].fillna("NA")

if "fruto" not in df: 
    df["fruto"] = 1
else:
    # normaliza a enteros si es posible (deja NaN si no)
    df["fruto"] = pd.to_numeric(df["fruto"], errors="coerce")

# >>>>>>>> CLAVES DE AGRUPACIÓN (incluye portainjerto) <<<<<<<<
grp_keys_fruto = ["especie","variedad","temporada","campo","portainjerto","fecha","fruto"]
grp_keys       = ["especie","variedad","temporada","campo","portainjerto","fecha"]
grp_wo_fruto   = ["especie","variedad","temporada","campo","portainjerto","fecha"]

# =============== IMPUTACIÓN: rellenar faltantes por promedio de grupo (excepto ACIDEZ) ===============
cols_to_impute = [col for col in ["brix","mej1","mej2","quilla","hombro","punta","peso"] if col in df.columns]
for col in cols_to_impute:
    grp_mean = df.groupby(grp_keys)[col].transform("mean", numeric_only=True)
    df[col] = df[col].fillna(grp_mean)

# =============== Métricas por fila ===============
if {"mej1","mej2"}.issubset(df.columns):
    df["avg_mejillas"] = df[["mej1","mej2"]].mean(axis=1)
else:
    df["avg_mejillas"] = np.nan

# Detección de duplicados de fruto y no correlativos por combinatoria (sin 'fruto' en la llave)
def _duplicados_lista(s):
    s = s.dropna().astype(int)
    dup = s[s.duplicated(keep=False)].unique()
    return ",".join(map(str, sorted(dup))) if len(dup) else ""

def _no_correlativos(s):
    s = s.dropna().astype(int)
    if s.empty: 
        return False
    arr = np.sort(s.unique())
    return not (arr[-1] - arr[0] + 1 == len(arr))  # True si NO es consecutivo

# flags a nivel fila
dup_counts = df.groupby(grp_wo_fruto)["fruto"].transform(lambda s: s.duplicated(keep=False))
df["fruto_duplicado_en_grupo"] = dup_counts.fillna(False).astype(bool)

# resumen de duplicados/no correlativos por grupo
rep = (
    df.groupby(grp_wo_fruto, dropna=False)["fruto"]
      .agg(frutos_detectados=lambda s: ",".join(map(str, sorted(s.dropna().astype(int).unique()))),
           frutos_no_correlativos=_no_correlativos,
           frutos_duplicados_lista=_duplicados_lista)
      .reset_index()
)
rep["tiene_fruto_duplicado"] = rep["frutos_duplicados_lista"].astype(str).str.len() > 0

# =============== Promedios por grupo y resumen base ===============
df = df.sort_values(grp_keys_fruto)
df["brix_prom_grupo"]  = df.groupby(grp_keys)["brix"].transform("mean")
df["acidez_1er_fruto"] = df.groupby(grp_keys)["acidez"].transform(first_non_null)
df["prom_mejillas_grupo"] = df.groupby(grp_keys)["avg_mejillas"].transform("mean")
if "peso" in df:
    df["peso_prom_grupo"] = df.groupby(grp_keys)["peso"].transform("mean")
else:
    df["peso_prom_grupo"] = np.nan

agg_dict = {
    "brix": ("brix","mean"),
    "acidez_1er_fruto": ("acidez_1er_fruto","first"),
    "avg_mejillas": ("avg_mejillas","mean"),
}
for pt in ["quilla","hombro","punta"]:
    if pt in df:
        agg_dict[pt] = (pt, "mean")
if "peso" in df:
    agg_dict["peso"] = ("peso","mean")

resumen = (
    df.groupby(grp_keys, dropna=False)
      .agg(n_reg=("brix","size"), **agg_dict)
      .reset_index()
      .rename(columns={
          "brix": "brix_prom",
          "acidez_1er_fruto": "acidez_1er",
          "avg_mejillas": "prom_mejillas",
          "quilla": "prom_quilla",
          "hombro": "prom_hombro",
          "punta": "prom_punta",
          "peso": "peso_prom"
      })
)

# Si faltan todas las columnas de puntos, crea columnas vacías
for col in ["prom_quilla","prom_hombro","prom_punta"]:
    if col not in resumen:
        resumen[col] = np.nan
if "peso_prom" not in resumen:
    resumen["peso_prom"] = np.nan

# Punto más débil: mínimo de los promedios por punto
puntos_cols = ["prom_quilla","prom_hombro","prom_punta"]
resumen["fpunto_debil_valor"] = resumen[puntos_cols].min(axis=1, skipna=True)

# Etiqueta del lugar más débil
def _lugar_debil(row):
    vals = {k: row[k] for k in puntos_cols if pd.notna(row[k])}
    if not vals:
        return np.nan
    minv = min(vals.values())
    lugares = [k.replace("prom_","").capitalize() for k,v in vals.items() if np.isclose(v, minv, equal_nan=False)]
    return "/".join(lugares)
resumen["fpunto_debil_lugar"] = resumen.apply(_lugar_debil, axis=1)

# Métrica compuesta
resumen["min_promedios"] = np.minimum(resumen["prom_mejillas"], resumen["fpunto_debil_valor"])

# ====== Marcas por 'campo' nulo original y duplicados/no correlativos ======
# Campo nulo original a nivel grupo
campo_nulo_grp = (
    df.groupby(grp_keys, dropna=False)["campo_nulo_original"]
      .any().rename("grupo_con_campo_nulo_original").reset_index()
)
resumen = resumen.merge(campo_nulo_grp, on=grp_keys, how="left")
# Duplicados / no correlativos (se agrupan sin 'fruto' en la llave)
resumen = resumen.merge(rep, left_on=grp_wo_fruto, right_on=grp_wo_fruto, how="left")

# ====== Peso y categoría de Ciruela (Cherry vs Candy) ======
def _definir_plum_tipo(row):
    if str(row["especie"]).strip().lower() == "ciruela":
        p = row.get("peso_prom", np.nan)
        if pd.notna(p):
            return ("Candy plum", "candy") if p > UMBRAL_PESO_CANDY else ("Cherry plum", "sugar")
        else:
            return ("Cherry plum (sin peso)", "sugar")
    return (np.nan, np.nan)

tipos = resumen.apply(_definir_plum_tipo, axis=1, result_type="expand")
tipos.columns = ["plum_categoria", "plum_subtype_key"]
resumen = pd.concat([resumen, tipos], axis=1)

# =============== ID y llave de grupo (incluye portainjerto) ===============
resumen["fecha_key"] = pd.to_datetime(resumen["fecha"], errors="coerce").dt.strftime("%Y-%m-%d")
resumen["grupo_key"] = (
    resumen["especie"].astype(str).str.strip() + "|" +
    resumen["variedad"].astype(str).str.strip() + "|" +
    resumen["temporada"].astype(str).str.strip() + "|" +
    resumen["campo"].astype(str).str.strip() + "|" +
    resumen["portainjerto"].astype(str).str.strip() + "|" +
    resumen["fecha_key"].fillna("NaT")
)
resumen["grupo_id"] = pd.factorize(resumen["grupo_key"], sort=True)[0] + 1
resumen = resumen.drop(columns=["fecha_key"])

# ====== Estado de cálculo, motivos y alertas ======
def _eval_calculo(row):
    reasons = []
    alerts = []

    especie = str(row["especie"]).strip().lower()
    tiene_reglas = especie in ("ciruela", "nectarin")
    if not tiene_reglas:
        reasons.append("sin reglas para especie")

    # brix
    lo, hi = RANGOS_VALIDOS["brix"]
    if pd.isna(row["brix_prom"]):
        reasons.append("brix_prom vacío")
    elif not _in_range(row["brix_prom"], lo, hi):
        reasons.append(f"brix_prom fuera de rango [{lo}, {hi})")

    # mejillas
    lo, hi = RANGOS_VALIDOS["mejillas"]
    if pd.isna(row["prom_mejillas"]):
        reasons.append("prom_mejillas vacío")
    elif not _in_range(row["prom_mejillas"], lo, hi):
        reasons.append(f"prom_mejillas fuera de rango [{lo}, {hi})")

    # firmeza puntos (min de promedios)
    lo, hi = RANGOS_VALIDOS["firmeza_puntos"]
    if pd.isna(row["fpunto_debil_valor"]):
        reasons.append("fpunto_debil_valor vacío")
    elif not _in_range(row["fpunto_debil_valor"], lo, hi):
        reasons.append(f"fpunto_debil_valor fuera de rango [{lo}, {hi})")

    # acidez (sin imputación)
    lo, hi = RANGOS_VALIDOS["acidez"]
    if pd.isna(row["acidez_1er"]):
        reasons.append("acidez_1er vacío")
    elif not _in_range(row["acidez_1er"], lo, hi):
        reasons.append(f"acidez_1er fuera de rango [{lo}, {hi})")

    # alertas
    if especie == "ciruela":
        lo, hi = RANGOS_VALIDOS["peso"]
        if pd.isna(row["peso_prom"]):
            alerts.append("peso_prom faltante ⇒ Cherry")
        elif not _in_range(row["peso_prom"], lo, hi):
            alerts.append(f"peso_prom fuera de rango [{lo}, {hi}) ⇒ Cherry")

    if row.get("grupo_con_campo_nulo_original", False):
        alerts.append("campo nulo original")

    if row.get("tiene_fruto_duplicado", False):
        alerts.append(f"fruto duplicado(s): {row.get('frutos_duplicados_lista','')}")

    if row.get("frutos_no_correlativos", False):
        alerts.append("números de fruto no correlativos")

    ok = (len(reasons) == 0) and tiene_reglas
    estado = "OK" if ok else "ERROR"
    return pd.Series({"estado_calculo": estado,
                      "motivo_error": "; ".join(reasons) if reasons else "",
                      "alertas": "; ".join([a for a in alerts if a])})

resumen = pd.concat([resumen, resumen.apply(_eval_calculo, axis=1)], axis=1)

# Propagar ID/Key, estado y marcas al detalle
df = df.merge(
    resumen[grp_keys + [
        "grupo_id","grupo_key","peso_prom","fpunto_debil_valor","fpunto_debil_lugar",
        "estado_calculo","motivo_error","alertas"
    ]],
    on=grp_keys, how="left", validate="m:1"
)
# Añadir marcas fila: campo nulo original y duplicado de fruto
df["campo_nulo_original"] = df["campo_nulo_original"].fillna(False)
df["fruto_duplicado_en_grupo"] = df["fruto_duplicado_en_grupo"].fillna(False)

# =============== Validaciones ===============
# 1) Acidez = primer fruto (en el detalle)
val1 = (
    df.groupby(grp_keys, dropna=False)
      .apply(lambda g: (g["acidez"].dropna().eq(g["acidez_1er_fruto"].iloc[0])).all(), include_groups=False)
      .rename("ok_acidez_es_el_primer_fruto")
      .reset_index()
)

# 2) BRIX promedio correcto (detalle vs resumen)
val2 = (
    df.groupby(grp_keys, dropna=False)
      .apply(lambda g: np.isclose(g["brix"].mean(skipna=True), g["brix_prom_grupo"].iloc[0], equal_nan=True), include_groups=False)
      .rename("ok_brix_promedio")
      .reset_index()
)

# 3) min(prom_mejillas, fpunto_debil_valor) consistente (validación a nivel resumen)
val3 = resumen[grp_keys].copy()
val3["ok_min_promedios"] = np.isclose(
    resumen["min_promedios"],
    np.minimum(resumen["prom_mejillas"], resumen["fpunto_debil_valor"]),
    equal_nan=True
)

# Armar tabla de validación
validacion = (
    resumen.merge(val1, on=grp_keys, how="left")
           .merge(val2, on=grp_keys, how="left")
           .merge(val3, on=grp_keys, how="left")
)

# Diferencia útil BRIX (detalle vs resumen)
brix_prom_calc = (
    df.groupby(grp_keys, dropna=False)["brix_prom_grupo"].first().rename("brix_prom_calc").reset_index()
)
validacion = validacion.merge(brix_prom_calc, on=grp_keys, how="left")
validacion["dif_brix_prom_vs_calc"] = validacion["brix_prom"] - validacion["brix_prom_calc"]

# =============== Bandas por métrica + clusters ===============
def clasificar_resumen(res):
    res = res.copy()
    grp_brix, grp_mej, grp_fp, grp_ac = [], [], [], []
    for _, r in res.iterrows():
        if r["especie"] == "Ciruela":
            rules = PLUM_RULES.get(r.get("plum_subtype_key", "sugar"), {})
        elif r["especie"] == "Nectarin":
            # Detectar color/periodo desde el detalle del grupo
            rows = df[
                (df["especie"]==r["especie"]) & (df["variedad"]==r["variedad"]) &
                (df["temporada"]==r["temporada"]) & (df["campo"]==r["campo"]) &
                (df["portainjerto"]==r["portainjerto"]) & (df["fecha"]==r["fecha"])
            ]
            if "color" in df.columns:
                series_color = rows["color"].dropna().astype(str)
                color = series_color.iloc[0].lower() if not series_color.empty else "amarilla"
            else:
                color = "amarilla"
            color = "blanca" if color.startswith("blanc") else "amarilla"
            period = harvest_period(color, r["fecha"])
            rules = NECT_RULES.get(color, {}).get(period, {})
        else:
            rules = {}

        grp_brix.append(clasificar(r["brix_prom"], rules.get("BRIX", [])))
        grp_mej.append(clasificar(r["prom_mejillas"], rules.get("FIRMEZA_MEJ", [])))
        grp_fp.append(clasificar(r["fpunto_debil_valor"], rules.get("FIRMEZA_PUNTO", [])))
        grp_ac.append(clasificar(r["acidez_1er"], rules.get("ACIDEZ", [])))

    res["grp_brix"] = grp_brix
    res["grp_mejillas"] = grp_mej
    res["grp_fpunto"] = grp_fp
    res["grp_acidez"] = grp_ac
    res["banda_fpunto"] = res["grp_fpunto"]

    if MODO_COND_SUM == "media":
        res["cond_sum"] = res[["grp_brix","grp_mejillas","grp_fpunto","grp_acidez"]].mean(axis=1, skipna=True)
    else:
        res["cond_sum"] = res[["grp_brix","grp_mejillas","grp_fpunto","grp_acidez"]].sum(axis=1, min_count=1)

    try:
        res["cluster"] = pd.qcut(res["cond_sum"], 4, labels=[1,2,3,4])
    except ValueError:
        res["cluster"] = pd.cut(res["cond_sum"], 4, labels=[1,2,3,4])
    return res

resumen_clasif = clasificar_resumen(resumen)

# Añadir ID/Key y variables relevantes
resumen_clasif = resumen_clasif.merge(
    resumen[grp_keys + [
        "grupo_id","grupo_key","peso_prom","plum_categoria","plum_subtype_key",
        "prom_quilla","prom_hombro","prom_punta","fpunto_debil_valor","fpunto_debil_lugar",
        "grupo_con_campo_nulo_original","tiene_fruto_duplicado","frutos_no_correlativos",
        "frutos_duplicados_lista","frutos_detectados",
        "estado_calculo","motivo_error","alertas"
    ]],
    on=grp_keys, how="left", validate="1:1"
)

# =============== Exportar (ID primero) ===============
RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)

def _move_id_first(df_):
    cols = list(df_.columns)
    # mover estas, si existen
    for k in ["grupo_id","grupo_key","estado_calculo","motivo_error","alertas",
              "grupo_con_campo_nulo_original","tiene_fruto_duplicado","frutos_no_correlativos"]:
        if k in cols:
            cols.insert(0, cols.pop(cols.index(k)))
    return df_[cols]

with pd.ExcelWriter(RUTA_SALIDA, engine="xlsxwriter") as w:
    _move_id_first(df).to_excel(w, index=False, sheet_name="detalle_filas")
    _move_id_first(resumen).to_excel(w, index=False, sheet_name="resumen")
    _move_id_first(validacion).to_excel(w, index=False, sheet_name="validacion")
    _move_id_first(resumen_clasif).to_excel(w, index=False, sheet_name="bandas_clusters")

print("OK - Archivo generado:", RUTA_SALIDA)
print("Total combinaciones:", len(resumen))
print("Con ERROR en cálculo:", (resumen["estado_calculo"]=="ERROR").sum())
print("Con campo nulo original:", resumen["grupo_con_campo_nulo_original"].sum())
print("Con fruto duplicado:", resumen["tiene_fruto_duplicado"].sum())
print("Con frutos no correlativos:", resumen["frutos_no_correlativos"].sum())
