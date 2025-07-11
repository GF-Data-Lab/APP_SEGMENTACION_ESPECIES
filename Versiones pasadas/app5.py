# streamlit_app.py
"""Streamlit app para clasificar muestras de carozos
===================================================

Ejecuta:
```
streamlit run streamlit_app.py
```

Requisitos:
- Python ≥ 3.9
- pip install streamlit pandas numpy openpyxl
- Asegúrate de que el módulo `procesar_carozos.py` (versión ≥ 7) esté
  en el mismo directorio o en el PYTHONPATH.

La app tiene dos secciones:
1. **Procesamiento completo** de un archivo Excel: sube tu hoja de
   laboratorio y obtén el *data‑frame* clasificado + clusters.
2. **Simulador**: ingresa manualmente los valores de una muestra y
   calcula al instante el *cluster* que recibiría.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Importamos la lógica de clasificación desde el módulo oficial  
# Si prefieres un nombre distinto, cámbialo aquí.
# ---------------------------------------------------------------------------
from procesar_carozos import (
    process_carozos,
    _plum_subtype,
    _harvest_period,
    _classify_row,
    _rule_key,
    PLUM_RULES,
    NECT_RULES,
    COL_BRIX,
    COL_ACIDEZ,
    #COL_PROD,
    COL_FIRMEZA_PUNTO,
    COL_FIRMEZA_MEJILLAS,
)

# Campos y constantes --------------------------------------------------------
ESPECIE_OPTS = ("Ciruela", "Nectarin")
COLOR_OPTS = ("Amarilla", "Blanca")

NUMERIC_FIELDS = (
    list(COL_FIRMEZA_PUNTO)
    + list(COL_FIRMEZA_MEJILLAS)
    + [
        "Peso (g)",  # suficiente para discriminar candy/cherry
        COL_BRIX,
        COL_ACIDEZ#,
        #COL_PROD,
    ]
)

# ---------------------------------------------------------------------------
# Utilidades locales
# ---------------------------------------------------------------------------

def _classify_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Devuelve las clasificaciones y cluster para una muestra suelta."""

    row = pd.Series(sample)

    # Derivados del pipeline oficial -------------------------------------
    row["plum_subtype"] = _plum_subtype(row)
    row["harvest_period"] = _harvest_period(row.get("Fecha cosecha"))

    # Firmeza punto débil = mínimo entre quilla/hombro/mejillas ----------
    firmness_vals = [row.get(f) for f in COL_FIRMEZA_PUNTO + COL_FIRMEZA_MEJILLAS]
    firmness_vals = [v for v in firmness_vals if pd.notna(v)]
    row["Firmeza punto débil"] = min(firmness_vals) if firmness_vals else np.nan

    # Clasificación campo a campo ----------------------------------------
    cols_to_class = [
        "Firmeza punto débil",
        COL_BRIX,
        COL_ACIDEZ#,
       # COL_PROD,
    ]
    for col in cols_to_class:
        row[f"grp_{col.replace(' ', '_')}"] = _classify_row(row, col)

    # Suma condicional y cluster -----------------------------------------
    grp_cols = [c for c in row.index if c.startswith("grp_")]
    cond_sum = pd.Series(row[grp_cols]).sum(min_count=1)
    row["cond_sum"] = cond_sum
    # con un único valor no hay cuartiles; usamos regla simple
    if pd.isna(cond_sum):
        row["cluster_est"] = np.nan
    elif cond_sum <= 6:
        row["cluster_est"] = 1
    elif cond_sum <= 8:
        row["cluster_est"] = 2
    elif cond_sum <= 10:
        row["cluster_est"] = 3
    else:
        row["cluster_est"] = 4

    return row.to_dict()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Clasificación de Carozos", layout="wide")

    st.title("🍑 Clasificación y Clustering de Carozos")

    # --------------------------------------------------
    # Sección 1 — Procesamiento completo del Excel
    # --------------------------------------------------
    st.header("1️⃣ Procesar archivo Excel completo")
    uploaded = st.file_uploader(
        "Carga tu archivo *.xlsx* con la hoja ‘CAROZOS’", type=["xlsx"], key="xlsx"
    )
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)
        df_proc = process_carozos(tmp_path)
        st.success(f"Archivo procesado: {len(df_proc)} registros")
        st.dataframe(df_proc, use_container_width=True)

        # Descarga opcional ------------------------------------------------
        csv_bytes = df_proc.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Descargar CSV procesado",
            data=csv_bytes,
            file_name="carozos_procesado.csv",
            mime="text/csv",
        )

    st.divider()

    # --------------------------------------------------
    # Sección 2 — Simulador rápido
    # --------------------------------------------------
    st.header("2️⃣ Simulador de muestra individual")

    with st.form("sim_form", clear_on_submit=False):
        cols1 = st.columns(4)
        especie = cols1[0].selectbox("Especie", ESPECIE_OPTS)
        variedad = cols1[1].text_input("Variedad", value="TestVar")
        fruto_n = cols1[2].number_input("Fruto (n°)", min_value=1, step=1, value=1)
        peso = cols1[3].number_input("Peso (g)", min_value=1.0, step=0.1, value=50.0)

        cols2 = st.columns(4)
        fecha = cols2[0].date_input("Fecha cosecha")
        color_pulpa = (
            cols2[1].selectbox("Color de pulpa", COLOR_OPTS)
            if especie == "Nectarin"
            else "Amarilla"
        )
        brix = cols2[2].number_input("Brix (°)" , min_value=0.0, step=0.1, value=17.0)
        acidez = cols2[3].number_input("Acidez (%)", min_value=0.0, step=0.01, value=0.85)

        cols3 = st.columns(4)
        prod = cols3[0].number_input("Productividad (Ton)", min_value=0.0, step=1.0, value=30.0)
        q_p = cols3[1].number_input("Quilla (lb)", min_value=0.0, step=0.1, value=5.5)
        q_h = cols3[2].number_input("Hombro (lb)", min_value=0.0, step=0.1, value=5.5)
        m1 = cols3[3].number_input("Mejilla 1 (lb)", min_value=0.0, step=0.1, value=7.0)

        cols4 = st.columns(4)
        m2 = cols4[0].number_input("Mejilla 2 (lb)", min_value=0.0, step=0.1, value=7.0)

        submitted = st.form_submit_button("Calcular clasificación")

    if submitted:
        sample = {
            "Especie": especie,
            "Variedad": variedad,
            "Fruto (n°)": fruto_n,
            "Peso (g)": peso,
            "Fecha cosecha": pd.Timestamp(fecha),
            "Color de pulpa": color_pulpa,
            COL_BRIX: brix,
            COL_ACIDEZ: acidez,
            #COL_PROD: prod,
            COL_FIRMEZA_PUNTO[0]: q_p,
            COL_FIRMEZA_PUNTO[1]: q_h,
            COL_FIRMEZA_MEJILLAS[0]: m1,
            COL_FIRMEZA_MEJILLAS[1]: m2,
        }

        result = _classify_sample(sample)

        st.subheader("Resultado de la muestra")
        show = {
            "Especie": result["Especie"],
            "plum_subtype": result.get("plum_subtype"),
            "harvest_period": result.get("harvest_period"),
            "Firmeza punto débil": result.get("Firmeza punto débil"),
            "Grupo Firmeza": result.get("grp_Firmeza_punto_débil"),
            "Grupo Brix": result.get("grp_BRIX"),
            "Grupo Acidez": result.get("grp_Acidez_(%)"),
            "Grupo Prod": result.get("grp_Productividad_(Ton)"),
            "Cluster estimado": result.get("cluster_est"),
        }
        st.table(pd.DataFrame([show]))

        st.info("Recuerda que los límites de los clusters son aproximados al ser una sola muestra; en procesamiento masivo empleamos cuartiles.")


if __name__ == "__main__":
    main()
