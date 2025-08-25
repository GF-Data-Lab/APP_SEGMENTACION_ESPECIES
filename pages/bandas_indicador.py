import streamlit as st
import pandas as pd
import numpy as np
from utils import show_logo
from segmentacion_base import (
    DEFAULT_PLUM_RULES,
    DEFAULT_NECT_RULES,
    plum_rules_to_df,
    nect_rules_to_df,
    df_to_plum_rules,
    df_to_nect_rules,
)


st.set_page_config(page_title="Bandas por indicador", page_icon="🎯", layout="wide")

st.markdown(
    """
    <style>
      [data-testid="stSidebar"] div.stButton > button {
        background-color: #D32F2F !important;
        color: white !important;
        border: none !important;
      }
      [data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #B71C1C !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def generar_menu():
    with st.sidebar:
        show_logo()
        if st.button('Página de Inicio 🏚️'):
            st.switch_page('app.py')
        if st.button('Carga de archivos 📁'):
            st.switch_page('pages/carga_datos.py')
        if st.button('Valores por defecto ⚙️'):
            st.switch_page('pages/default_values.py')
        if st.button('Bandas por indicador 🎯'):
            st.switch_page('pages/bandas_indicador.py')
        if st.button('Modelo y clustering 🧠'):
            st.switch_page('pages/modelo_cluster.py')
        if st.button('Análisis exploratorio 🔍'):
            st.switch_page('pages/analisis.py')


def main():
    generar_menu()
    st.title("Bandas por indicador")

    # Inicializar dataframes de reglas
    if "plum_rules_df" not in st.session_state:
        st.session_state["plum_rules_df"] = plum_rules_to_df(DEFAULT_PLUM_RULES)
    if "nect_rules_df" not in st.session_state:
        st.session_state["nect_rules_df"] = nect_rules_to_df(DEFAULT_NECT_RULES)

    especie = st.radio("Especie", ["Ciruela", "Nectarina"], horizontal=True)

    # Convertir los dataframes de la sesión a diccionarios para manipularlos
    current_plum_rules = df_to_plum_rules(st.session_state["plum_rules_df"])
    current_nect_rules = df_to_nect_rules(st.session_state["nect_rules_df"])

    # Mapa de colores para los grupos
    group_colors = {
        1: "#a8e6cf",  # verde claro
        2: "#ffd3b6",  # naranja claro
        3: "#ffaaa5",  # coral
        4: "#ff8b94",  # rojo rosado
    }

    if especie == "Ciruela":
        subtipo_sel = st.selectbox("Sub‑tipo de ciruela", list(current_plum_rules.keys()))
        metrica_sel = st.selectbox("Métrica", list(current_plum_rules[subtipo_sel].keys()))

        # Posibilidad de crear una nueva métrica
        with st.expander("Agregar nueva métrica para este sub‑tipo", expanded=False):
            nueva_metric = st.text_input("Nombre de la nueva métrica", key=f"new_metric_plum_{subtipo_sel}")
            if st.button("Crear métrica", key=f"create_metric_plum_{subtipo_sel}"):
                if nueva_metric:
                    if nueva_metric not in current_plum_rules[subtipo_sel]:
                        default_bands = [(-np.inf, 0.0, 4), (0.0, 1.0, 3), (1.0, 2.0, 2), (2.0, np.inf, 1)]
                        current_plum_rules[subtipo_sel][nueva_metric] = default_bands
                        st.session_state["plum_rules_df"] = plum_rules_to_df(current_plum_rules)
                        st.success(f"Métrica '{nueva_metric}' añadida.")
                    else:
                        st.warning("La métrica ya existe.")
                else:
                    st.warning("Debes introducir un nombre para la nueva métrica.")

        bandas = current_plum_rules[subtipo_sel][metrica_sel]
        bandas_df = pd.DataFrame(bandas, columns=["Min", "Max", "Grupo"])
        def _apply_colors_plum(row):
            return [f"background-color: {group_colors.get(int(row['Grupo']), '')}" for _ in row]
        try:
            st.write(bandas_df.style.apply(_apply_colors_plum, axis=1))
        except Exception:
            st.write(bandas_df)

        st.markdown("**Editar bandas**")
        nuevas_bandas = []
        import math
        for i, (lo, hi, grp) in enumerate(bandas):
            cols = st.columns([2, 2, 1])
            lo_val = float(lo) if math.isfinite(lo) else -1e6
            hi_val = float(hi) if math.isfinite(hi) else 1e6
            lo_new = cols[0].number_input(
                f"Mín banda {i+1}", value=lo_val, key=f"plum_{subtipo_sel}_{metrica_sel}_min_{i}"
            )
            hi_new = cols[1].number_input(
                f"Máx banda {i+1}", value=hi_val, key=f"plum_{subtipo_sel}_{metrica_sel}_max_{i}"
            )
            grp_new = cols[2].selectbox(
                f"Grupo banda {i+1}",
                options=[1, 2, 3, 4],
                index=int(grp) - 1 if not math.isnan(grp) else 0,
                key=f"plum_{subtipo_sel}_{metrica_sel}_grp_{i}"
            )
            nuevas_bandas.append((lo_new, hi_new, grp_new))

        if st.button("Agregar banda", key=f"add_plum_{subtipo_sel}_{metrica_sel}"):
            last_hi = nuevas_bandas[-1][1] if nuevas_bandas else 0
            nuevas_bandas.append((last_hi, last_hi + 1, 4))

        if st.button("Guardar cambios de regla", key=f"save_plum_{subtipo_sel}_{metrica_sel}"):
            current_plum_rules[subtipo_sel][metrica_sel] = nuevas_bandas
            st.session_state["plum_rules_df"] = plum_rules_to_df(current_plum_rules)
            st.success("Regla actualizada para ciruela.")

    else:
        color_sel = st.selectbox("Color de pulpa", list(current_nect_rules.keys()))
        periodo_sel = st.selectbox("Periodo de cosecha", list(current_nect_rules[color_sel].keys()))
        metrica_sel_n = st.selectbox("Métrica", list(current_nect_rules[color_sel][periodo_sel].keys()))

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
        bandas_df_n = pd.DataFrame(bandas_n, columns=["Min", "Max", "Grupo"])
        def _apply_colors_nect(row):
            return [f"background-color: {group_colors.get(int(row['Grupo']), '')}" for _ in row]
        try:
            st.write(bandas_df_n.style.apply(_apply_colors_nect, axis=1))
        except Exception:
            st.write(bandas_df_n)

        st.markdown("**Editar bandas**")
        nuevas_bandas_n = []
        import math
        for i, (lo, hi, grp) in enumerate(bandas_n):
            cols = st.columns([2, 2, 1])
            lo_val = float(lo) if math.isfinite(lo) else -1e6
            hi_val = float(hi) if math.isfinite(hi) else 1e6
            lo_new = cols[0].number_input(
                f"Mín banda {i+1}", value=lo_val, key=f"nect_{color_sel}_{periodo_sel}_{metrica_sel_n}_min_{i}"
            )
            hi_new = cols[1].number_input(
                f"Máx banda {i+1}", value=hi_val, key=f"nect_{color_sel}_{periodo_sel}_{metrica_sel_n}_max_{i}"
            )
            grp_new = cols[2].selectbox(
                f"Grupo banda {i+1}",
                options=[1, 2, 3, 4],
                index=int(grp) - 1 if not math.isnan(grp) else 0,
                key=f"nect_{color_sel}_{periodo_sel}_{metrica_sel_n}_grp_{i}"
            )
            nuevas_bandas_n.append((lo_new, hi_new, grp_new))

        if st.button("Agregar banda", key=f"add_nect_{color_sel}_{periodo_sel}_{metrica_sel_n}"):
            last_hi = nuevas_bandas_n[-1][1] if nuevas_bandas_n else 0
            nuevas_bandas_n.append((last_hi, last_hi + 1, 4))

        if st.button("Guardar cambios de regla", key=f"save_nect_{color_sel}_{periodo_sel}_{metrica_sel_n}"):
            current_nect_rules[color_sel][periodo_sel][metrica_sel_n] = nuevas_bandas_n
            st.session_state["nect_rules_df"] = nect_rules_to_df(current_nect_rules)
            st.success("Regla actualizada para nectarina.")

    st.info("Los cambios se guardan automáticamente en la sesión.")


if __name__ == "__main__":
    main()

