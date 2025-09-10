import streamlit as st
import pandas as pd
import numpy as np
from utils import show_logo
from segmentacion_base import (
    DEFAULT_PLUM_RULES, DEFAULT_NECT_RULES,
    plum_rules_to_df, df_to_plum_rules,
    nect_rules_to_df, df_to_nect_rules
)

# Configuración de la página
st.set_page_config(
    page_title="Métricas y Bandas por Especie", 
    page_icon="📊", 
    layout="wide"
)

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
        if st.button('Segmentación Ciruela 🍑'):
            st.switch_page('pages/segmentacion_ciruela.py')
        if st.button('Segmentación Nectarina 🍑'):
            st.switch_page('pages/segmentacion_nectarina.py')
        if st.button('Modelo de Clasificación'):
            st.switch_page('pages/Cluster_especies.py')
        if st.button('Análisis exploratorio'):
            st.switch_page('pages/analisis.py')
        if st.button('Métricas y Bandas 📊', type="primary"):
            st.switch_page('pages/metricas_bandas.py')
        if st.button('Detección Outliers 🎯'):
            st.switch_page('pages/outliers.py')

def main():
    generar_menu()
    
    st.title("📊 Configuración de Métricas y Bandas por Especie")
    
    st.markdown("""
    Esta página te permite configurar las **reglas de clasificación** para cada especie:
    - **Ciruela**: Configurar bandas para tipos Candy y Sugar
    - **Nectarina**: Configurar bandas por color (Amarilla/Blanca) y periodo de cosecha
    
    Las reglas definen cómo se clasifican los frutos en grupos 1-4 según sus métricas de calidad.
    """)
    
    # Inicializar reglas en session_state si no existen
    if "current_plum_rules" not in st.session_state:
        st.session_state["current_plum_rules"] = DEFAULT_PLUM_RULES.copy()
    if "current_nect_rules" not in st.session_state:
        st.session_state["current_nect_rules"] = DEFAULT_NECT_RULES.copy()
    
    # Tabs para cada especie
    tab1, tab2 = st.tabs(["🍑 Reglas Ciruela", "🍑 Reglas Nectarina"])
    
    with tab1:
        configurar_reglas_ciruela()
    
    with tab2:
        configurar_reglas_nectarina()

def configurar_reglas_ciruela():
    st.header("Configuración de Reglas para Ciruela")
    
    st.info("""
    **Ciruela** tiene dos subtipos principales:
    - **Candy**: Ciruelas más dulces con menor contenido de azúcar requerido
    - **Sugar**: Ciruelas que requieren mayor contenido de azúcar
    """)
    
    # Convertir reglas actuales a DataFrame para edición
    df_rules = plum_rules_to_df(st.session_state["current_plum_rules"])
    
    # Filtros para visualización
    col1, col2 = st.columns(2)
    with col1:
        subtype_filter = st.selectbox(
            "Filtrar por subtipo:", 
            ["Todos"] + list(st.session_state["current_plum_rules"].keys()),
            key="plum_subtype_filter"
        )
    with col2:
        metric_filter = st.selectbox(
            "Filtrar por métrica:",
            ["Todas"] + list(df_rules["metric"].unique()),
            key="plum_metric_filter"
        )
    
    # Aplicar filtros
    df_display = df_rules.copy()
    if subtype_filter != "Todos":
        df_display = df_display[df_display["subtype"] == subtype_filter]
    if metric_filter != "Todas":
        df_display = df_display[df_display["metric"] == metric_filter]
    
    # Editor de reglas
    st.subheader("Editor de Reglas")
    edited_df = st.data_editor(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "apply": st.column_config.CheckboxColumn("Aplicar", default=True),
            "subtype": st.column_config.SelectboxColumn("Subtipo", options=["candy", "sugar"]),
            "metric": st.column_config.TextColumn("Métrica"),
            "min": st.column_config.NumberColumn("Valor Mínimo", format="%.2f"),
            "max": st.column_config.NumberColumn("Valor Máximo", format="%.2f"),
            "group": st.column_config.SelectboxColumn("Grupo", options=[1, 2, 3, 4]),
        },
        key="plum_rules_editor"
    )
    
    # Botones de acción
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 Guardar Cambios", key="save_plum_rules"):
            try:
                # Actualizar las reglas filtradas en el DataFrame completo
                if subtype_filter != "Todos" or metric_filter != "Todas":
                    # Crear máscara para las filas editadas
                    mask = pd.Series(True, index=df_rules.index)
                    if subtype_filter != "Todos":
                        mask &= (df_rules["subtype"] == subtype_filter)
                    if metric_filter != "Todas":
                        mask &= (df_rules["metric"] == metric_filter)
                    
                    # Actualizar solo las filas filtradas
                    df_rules.loc[mask] = edited_df.values
                else:
                    df_rules = edited_df.copy()
                
                # Convertir de vuelta a formato de reglas
                st.session_state["current_plum_rules"] = df_to_plum_rules(df_rules)
                st.success("✅ Reglas de Ciruela guardadas correctamente")
            except Exception as e:
                st.error(f"❌ Error al guardar reglas: {e}")
    
    with col2:
        if st.button("🔄 Restaurar Valores por Defecto", key="reset_plum_rules"):
            st.session_state["current_plum_rules"] = DEFAULT_PLUM_RULES.copy()
            st.rerun()
    
    with col3:
        if st.button("➕ Agregar Nueva Regla", key="add_plum_rule"):
            st.session_state["show_add_plum_form"] = True
    
    # Formulario para agregar nueva regla
    if st.session_state.get("show_add_plum_form", False):
        with st.expander("➕ Agregar Nueva Regla de Ciruela", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                new_subtype = st.selectbox("Subtipo:", ["candy", "sugar"], key="new_plum_subtype")
                new_metric = st.text_input("Métrica:", key="new_plum_metric")
            with col2:
                new_min = st.number_input("Valor Mínimo:", value=0.0, key="new_plum_min")
                new_max = st.number_input("Valor Máximo:", value=100.0, key="new_plum_max")
            with col3:
                new_group = st.selectbox("Grupo:", [1, 2, 3, 4], key="new_plum_group")
                
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("✅ Crear Regla", key="create_plum_rule"):
                    if new_metric:
                        # Agregar la nueva regla
                        new_row = {
                            "apply": True,
                            "subtype": new_subtype,
                            "metric": new_metric,
                            "min": new_min,
                            "max": new_max,
                            "group": new_group
                        }
                        df_rules = pd.concat([df_rules, pd.DataFrame([new_row])], ignore_index=True)
                        st.session_state["current_plum_rules"] = df_to_plum_rules(df_rules)
                        st.session_state["show_add_plum_form"] = False
                        st.success("✅ Nueva regla creada")
                        st.rerun()
                    else:
                        st.error("❌ El nombre de la métrica es obligatorio")
            
            with col_btn2:
                if st.button("❌ Cancelar", key="cancel_plum_rule"):
                    st.session_state["show_add_plum_form"] = False
                    st.rerun()
    
    # Mostrar resumen de reglas actuales
    st.subheader("📈 Resumen de Reglas Actuales")
    for subtype, metrics in st.session_state["current_plum_rules"].items():
        with st.expander(f"**{subtype.upper()}** - {len(metrics)} métricas"):
            for metric, bands in metrics.items():
                st.write(f"**{metric}**: {len(bands)} bandas")
                for i, (min_val, max_val, group) in enumerate(bands):
                    min_str = f"{min_val:.2f}" if min_val != -np.inf else "-∞"
                    max_str = f"{max_val:.2f}" if max_val != np.inf else "+∞"
                    st.write(f"  - Banda {i+1}: [{min_str}, {max_str}] → Grupo {group}")

def configurar_reglas_nectarina():
    st.header("Configuración de Reglas para Nectarina")
    
    st.info("""
    **Nectarina** se clasifica por:
    - **Color de pulpa**: Amarilla / Blanca  
    - **Periodo de cosecha**: Muy temprana / Temprana / Tardía
    """)
    
    # Convertir reglas actuales a DataFrame para edición
    df_rules = nect_rules_to_df(st.session_state["current_nect_rules"])
    
    # Filtros para visualización
    col1, col2, col3 = st.columns(3)
    with col1:
        color_filter = st.selectbox(
            "Filtrar por color:", 
            ["Todos"] + list(st.session_state["current_nect_rules"].keys()),
            key="nect_color_filter"
        )
    with col2:
        period_options = []
        if color_filter != "Todos":
            period_options = list(st.session_state["current_nect_rules"][color_filter].keys())
        else:
            for color_rules in st.session_state["current_nect_rules"].values():
                period_options.extend(color_rules.keys())
            period_options = list(set(period_options))
        
        period_filter = st.selectbox(
            "Filtrar por periodo:",
            ["Todos"] + period_options,
            key="nect_period_filter"
        )
    with col3:
        metric_filter = st.selectbox(
            "Filtrar por métrica:",
            ["Todas"] + list(df_rules["metric"].unique()),
            key="nect_metric_filter"
        )
    
    # Aplicar filtros
    df_display = df_rules.copy()
    if color_filter != "Todos":
        df_display = df_display[df_display["color"] == color_filter]
    if period_filter != "Todos":
        df_display = df_display[df_display["period"] == period_filter]
    if metric_filter != "Todas":
        df_display = df_display[df_display["metric"] == metric_filter]
    
    # Editor de reglas
    st.subheader("Editor de Reglas")
    edited_df = st.data_editor(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "apply": st.column_config.CheckboxColumn("Aplicar", default=True),
            "color": st.column_config.SelectboxColumn("Color", options=["amarilla", "blanca"]),
            "period": st.column_config.SelectboxColumn("Periodo", options=["muy_temprana", "temprana", "tardia"]),
            "metric": st.column_config.TextColumn("Métrica"),
            "min": st.column_config.NumberColumn("Valor Mínimo", format="%.2f"),
            "max": st.column_config.NumberColumn("Valor Máximo", format="%.2f"),
            "group": st.column_config.SelectboxColumn("Grupo", options=[1, 2, 3, 4]),
        },
        key="nect_rules_editor"
    )
    
    # Botones de acción
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 Guardar Cambios", key="save_nect_rules"):
            try:
                # Actualizar las reglas filtradas en el DataFrame completo
                if color_filter != "Todos" or period_filter != "Todos" or metric_filter != "Todas":
                    # Crear máscara para las filas editadas
                    mask = pd.Series(True, index=df_rules.index)
                    if color_filter != "Todos":
                        mask &= (df_rules["color"] == color_filter)
                    if period_filter != "Todos":
                        mask &= (df_rules["period"] == period_filter)
                    if metric_filter != "Todas":
                        mask &= (df_rules["metric"] == metric_filter)
                    
                    # Actualizar solo las filas filtradas
                    df_rules.loc[mask] = edited_df.values
                else:
                    df_rules = edited_df.copy()
                
                # Convertir de vuelta a formato de reglas
                st.session_state["current_nect_rules"] = df_to_nect_rules(df_rules)
                st.success("✅ Reglas de Nectarina guardadas correctamente")
            except Exception as e:
                st.error(f"❌ Error al guardar reglas: {e}")
    
    with col2:
        if st.button("🔄 Restaurar Valores por Defecto", key="reset_nect_rules"):
            st.session_state["current_nect_rules"] = DEFAULT_NECT_RULES.copy()
            st.rerun()
    
    with col3:
        if st.button("➕ Agregar Nueva Regla", key="add_nect_rule"):
            st.session_state["show_add_nect_form"] = True
    
    # Formulario para agregar nueva regla
    if st.session_state.get("show_add_nect_form", False):
        with st.expander("➕ Agregar Nueva Regla de Nectarina", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                new_color = st.selectbox("Color:", ["amarilla", "blanca"], key="new_nect_color")
                new_period = st.selectbox("Periodo:", ["muy_temprana", "temprana", "tardia"], key="new_nect_period")
            with col2:
                new_metric = st.text_input("Métrica:", key="new_nect_metric")
                new_group = st.selectbox("Grupo:", [1, 2, 3, 4], key="new_nect_group")
            with col3:
                new_min = st.number_input("Valor Mínimo:", value=0.0, key="new_nect_min")
                new_max = st.number_input("Valor Máximo:", value=100.0, key="new_nect_max")
                
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("✅ Crear Regla", key="create_nect_rule"):
                    if new_metric:
                        # Agregar la nueva regla
                        new_row = {
                            "apply": True,
                            "color": new_color,
                            "period": new_period,
                            "metric": new_metric,
                            "min": new_min,
                            "max": new_max,
                            "group": new_group
                        }
                        df_rules = pd.concat([df_rules, pd.DataFrame([new_row])], ignore_index=True)
                        st.session_state["current_nect_rules"] = df_to_nect_rules(df_rules)
                        st.session_state["show_add_nect_form"] = False
                        st.success("✅ Nueva regla creada")
                        st.rerun()
                    else:
                        st.error("❌ El nombre de la métrica es obligatorio")
            
            with col_btn2:
                if st.button("❌ Cancelar", key="cancel_nect_rule"):
                    st.session_state["show_add_nect_form"] = False
                    st.rerun()
    
    # Mostrar resumen de reglas actuales
    st.subheader("📈 Resumen de Reglas Actuales")
    for color, periods in st.session_state["current_nect_rules"].items():
        with st.expander(f"**{color.upper()}** - {len(periods)} periodos"):
            for period, metrics in periods.items():
                st.write(f"**{period}**: {len(metrics)} métricas")
                for metric, bands in metrics.items():
                    st.write(f"  **{metric}**: {len(bands)} bandas")
                    for i, (min_val, max_val, group) in enumerate(bands):
                        min_str = f"{min_val:.2f}" if min_val != -np.inf else "-∞"
                        max_str = f"{max_val:.2f}" if max_val != np.inf else "+∞"
                        st.write(f"    - Banda {i+1}: [{min_str}, {max_str}] → Grupo {group}")

if __name__ == "__main__":
    main()