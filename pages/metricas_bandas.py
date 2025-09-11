import streamlit as st
import pandas as pd
import numpy as np
import math
from utils import show_logo
from segmentacion_base import (
    DEFAULT_PLUM_RULES,
    DEFAULT_NECT_RULES,
    plum_rules_to_df,
    nect_rules_to_df,
    df_to_plum_rules,
    df_to_nect_rules,
)

# Configuración de página
st.set_page_config(
    page_title="📊 Métricas y Bandas de Clasificación", 
    page_icon="📊", 
    layout="wide"
)

# Estilos CSS
st.markdown("""
    <style>
      [data-testid="stSidebar"] div.stButton > button {
        background-color: #D32F2F !important;
        color: white !important;
        border: none !important;
      }
      [data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #B71C1C !important;
      }
      
      .metric-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #D32F2F;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      }
    </style>
""", unsafe_allow_html=True)

def generarMenu():
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

generarMenu()

# ===== TÍTULO Y DESCRIPCIÓN =====
st.title("📊 Métricas y Bandas de Clasificación")

st.markdown("""
## 🎯 Configuración de Reglas de Segmentación

Esta página te permite configurar las **métricas y bandas de clasificación** para cada especie de fruto de carozo.
También puedes ajustar los **valores por defecto** utilizados cuando faltan datos en los registros.

### 📋 Funcionalidades:
- ✅ **Editar reglas** de clasificación por especie y período
- ✅ **Crear nuevas métricas** personalizadas
- ✅ **Modificar bandas** con límites mínimos y máximos
- ✅ **Configurar valores por defecto** para datos faltantes
- ✅ **Visualización con colores** por grupo de calidad

---
""")

# ===== INICIALIZACIÓN DE DATOS =====
# Inicializar dataframes de reglas en session_state
if "plum_rules_df" not in st.session_state:
    st.session_state["plum_rules_df"] = plum_rules_to_df(DEFAULT_PLUM_RULES)
if "nect_rules_df" not in st.session_state:
    st.session_state["nect_rules_df"] = nect_rules_to_df(DEFAULT_NECT_RULES)

# Inicializar valores por defecto
if "default_plum_subtype" not in st.session_state:
    st.session_state["default_plum_subtype"] = "sugar"
if "sugar_upper" not in st.session_state:
    st.session_state["sugar_upper"] = 60.0
if "default_color" not in st.session_state:
    st.session_state["default_color"] = "amarilla"
if "default_period" not in st.session_state:
    st.session_state["default_period"] = "tardia"

# Convertir dataframes a diccionarios para manipulación
current_plum_rules = df_to_plum_rules(st.session_state["plum_rules_df"])
current_nect_rules = df_to_nect_rules(st.session_state["nect_rules_df"])

# ===== COLORES PARA GRUPOS =====
group_colors = {
    1: '#a8e6cf',  # verde claro - Excelente
    2: '#ffd3b6',  # naranja claro - Bueno  
    3: '#ffaaa5',  # coral - Regular
    4: '#ff8b94',  # rojo rosado - Deficiente
}

# ===== PESTAÑAS PRINCIPALES =====
tab1, tab2, tab3 = st.tabs(["🍑 Reglas de Ciruela", "🍑 Reglas de Nectarina", "⚙️ Valores por Defecto"])

# ========================================
# TAB 1: REGLAS DE CIRUELA
# ========================================
with tab1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.subheader("🍑 Configuración de Reglas para Ciruela")
    st.markdown("Configura las bandas de clasificación para las diferentes métricas de calidad en ciruelas.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Selector de subtipo
    subtipo_sel = st.selectbox("📌 Subtipo de ciruela:", list(current_plum_rules.keys()), key="plum_subtype")
    metrica_sel = st.selectbox("📏 Métrica a configurar:", list(current_plum_rules[subtipo_sel].keys()), key="plum_metric")
    
    # Expander para agregar nueva métrica
    with st.expander("➕ Agregar nueva métrica para este subtipo", expanded=False):
        nueva_metric = st.text_input("Nombre de la nueva métrica:", key=f"new_metric_plum_{subtipo_sel}")
        if st.button("Crear métrica", key=f"create_metric_plum_{subtipo_sel}"):
            if nueva_metric:
                if nueva_metric not in current_plum_rules[subtipo_sel]:
                    # Bandas por defecto
                    default_bands = [(-np.inf, 0.0, 4), (0.0, 1.0, 3), (1.0, 2.0, 2), (2.0, np.inf, 1)]
                    current_plum_rules[subtipo_sel][nueva_metric] = default_bands
                    st.session_state["plum_rules_df"] = plum_rules_to_df(current_plum_rules)
                    st.success(f"✅ Métrica '{nueva_metric}' añadida correctamente.")
                    st.rerun()
                else:
                    st.warning("⚠️ La métrica ya existe.")
            else:
                st.warning("⚠️ Debes introducir un nombre para la nueva métrica.")
    
    # Mostrar bandas actuales con colores
    bandas = current_plum_rules[subtipo_sel][metrica_sel]
    bandas_df = pd.DataFrame(bandas, columns=["Mínimo", "Máximo", "Grupo"])
    
    def apply_colors_plum(row):
        return [f"background-color: {group_colors.get(int(row['Grupo']), '')}" for _ in row]
    
    st.markdown("#### 📊 Bandas actuales:")
    try:
        st.dataframe(bandas_df.style.apply(apply_colors_plum, axis=1), use_container_width=True)
    except Exception:
        st.dataframe(bandas_df, use_container_width=True)
    
    # Leyenda de colores
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div style="background-color: #a8e6cf; padding: 5px; text-align: center; border-radius: 5px;"><b>Grupo 1: Excelente</b></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="background-color: #ffd3b6; padding: 5px; text-align: center; border-radius: 5px;"><b>Grupo 2: Bueno</b></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div style="background-color: #ffaaa5; padding: 5px; text-align: center; border-radius: 5px;"><b>Grupo 3: Regular</b></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div style="background-color: #ff8b94; padding: 5px; text-align: center; border-radius: 5px;"><b>Grupo 4: Deficiente</b></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Editor de bandas
    st.markdown("#### ✏️ Editar bandas:")
    st.markdown("Modifica los límites de cada banda. Los valores infinitos se representan con números muy grandes/pequeños.")
    
    nuevas_bandas = []
    for i, (lo, hi, grp) in enumerate(bandas):
        cols = st.columns([2, 2, 1, 1])
        
        # Convertir infinitos a valores numéricos para la edición
        lo_val = float(lo) if math.isfinite(lo) else -1e6
        hi_val = float(hi) if math.isfinite(hi) else 1e6
        
        lo_new = cols[0].number_input(
            f"Mínimo banda {i+1}:", 
            value=lo_val, 
            key=f"plum_{subtipo_sel}_{metrica_sel}_min_{i}"
        )
        hi_new = cols[1].number_input(
            f"Máximo banda {i+1}:", 
            value=hi_val, 
            key=f"plum_{subtipo_sel}_{metrica_sel}_max_{i}"
        )
        grp_new = cols[2].selectbox(
            f"Grupo {i+1}:",
            options=[1, 2, 3, 4],
            index=int(grp) - 1 if not math.isnan(grp) else 0,
            key=f"plum_{subtipo_sel}_{metrica_sel}_grp_{i}"
        )
        
        # Convertir valores extremos de vuelta a infinito
        if lo_new <= -1e5:
            lo_new = -np.inf
        if hi_new >= 1e5:
            hi_new = np.inf
            
        nuevas_bandas.append((lo_new, hi_new, grp_new))
    
    # Botones de acción
    col_add, col_save, col_reset = st.columns([1, 1, 1])
    
    with col_add:
        if st.button("➕ Agregar banda", key=f"add_plum_{subtipo_sel}_{metrica_sel}"):
            last_hi = nuevas_bandas[-1][1] if nuevas_bandas else 0
            if math.isinf(last_hi):
                last_hi = 10
            nuevas_bandas.append((last_hi, last_hi + 5, 4))
            st.rerun()
    
    with col_save:
        if st.button("💾 Guardar cambios", key=f"save_plum_{subtipo_sel}_{metrica_sel}"):
            current_plum_rules[subtipo_sel][metrica_sel] = nuevas_bandas
            st.session_state["plum_rules_df"] = plum_rules_to_df(current_plum_rules)
            st.success(f"✅ Reglas actualizadas para {subtipo_sel} - {metrica_sel}")
    
    with col_reset:
        if st.button("🔄 Restaurar por defecto", key=f"reset_plum_{subtipo_sel}_{metrica_sel}"):
            # Restaurar a valores por defecto
            default_metric = DEFAULT_PLUM_RULES.get(subtipo_sel, {}).get(metrica_sel, [])
            if default_metric:
                current_plum_rules[subtipo_sel][metrica_sel] = default_metric
                st.session_state["plum_rules_df"] = plum_rules_to_df(current_plum_rules)
                st.success("✅ Reglas restauradas a valores por defecto")
                st.rerun()

# ========================================
# TAB 2: REGLAS DE NECTARINA
# ========================================
with tab2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.subheader("🍑 Configuración de Reglas para Nectarina")
    st.markdown("Configura las bandas de clasificación para las diferentes métricas de calidad en nectarinas.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Selectores de color y período
    color_sel = st.selectbox("🎨 Color de pulpa:", list(current_nect_rules.keys()), key="nect_color")
    periodo_sel = st.selectbox("📅 Período de cosecha:", list(current_nect_rules[color_sel].keys()), key="nect_period")
    metrica_sel_n = st.selectbox("📏 Métrica a configurar:", list(current_nect_rules[color_sel][periodo_sel].keys()), key="nect_metric")
    
    # Expander para agregar nueva métrica
    with st.expander("➕ Agregar nueva métrica para este color/período", expanded=False):
        nueva_metric_n = st.text_input(
            "Nombre de la nueva métrica:", 
            key=f"new_metric_nect_{color_sel}_{periodo_sel}"
        )
        if st.button("Crear métrica", key=f"create_metric_nect_{color_sel}_{periodo_sel}"):
            if nueva_metric_n:
                if nueva_metric_n not in current_nect_rules[color_sel][periodo_sel]:
                    default_bands_n = [(-np.inf, 0.0, 4), (0.0, 1.0, 3), (1.0, 2.0, 2), (2.0, np.inf, 1)]
                    current_nect_rules[color_sel][periodo_sel][nueva_metric_n] = default_bands_n
                    st.session_state["nect_rules_df"] = nect_rules_to_df(current_nect_rules)
                    st.success(f"✅ Métrica '{nueva_metric_n}' añadida correctamente.")
                    st.rerun()
                else:
                    st.warning("⚠️ La métrica ya existe.")
            else:
                st.warning("⚠️ Debes introducir un nombre para la nueva métrica.")
    
    # Mostrar bandas actuales
    bandas_n = current_nect_rules[color_sel][periodo_sel][metrica_sel_n]
    bandas_df_n = pd.DataFrame(bandas_n, columns=["Mínimo", "Máximo", "Grupo"])
    
    def apply_colors_nect(row):
        return [f"background-color: {group_colors.get(int(row['Grupo']), '')}" for _ in row]
    
    st.markdown("#### 📊 Bandas actuales:")
    try:
        st.dataframe(bandas_df_n.style.apply(apply_colors_nect, axis=1), use_container_width=True)
    except Exception:
        st.dataframe(bandas_df_n, use_container_width=True)
    
    # Leyenda de colores (igual que ciruela)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div style="background-color: #a8e6cf; padding: 5px; text-align: center; border-radius: 5px;"><b>Grupo 1: Excelente</b></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="background-color: #ffd3b6; padding: 5px; text-align: center; border-radius: 5px;"><b>Grupo 2: Bueno</b></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div style="background-color: #ffaaa5; padding: 5px; text-align: center; border-radius: 5px;"><b>Grupo 3: Regular</b></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div style="background-color: #ff8b94; padding: 5px; text-align: center; border-radius: 5px;"><b>Grupo 4: Deficiente</b></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Editor de bandas para nectarina
    st.markdown("#### ✏️ Editar bandas:")
    st.markdown("Modifica los límites de cada banda. Los valores infinitos se representan con números muy grandes/pequeños.")
    
    nuevas_bandas_n = []
    for i, (lo, hi, grp) in enumerate(bandas_n):
        cols = st.columns([2, 2, 1, 1])
        
        lo_val = float(lo) if math.isfinite(lo) else -1e6
        hi_val = float(hi) if math.isfinite(hi) else 1e6
        
        lo_new = cols[0].number_input(
            f"Mínimo banda {i+1}:", 
            value=lo_val, 
            key=f"nect_{color_sel}_{periodo_sel}_{metrica_sel_n}_min_{i}"
        )
        hi_new = cols[1].number_input(
            f"Máximo banda {i+1}:", 
            value=hi_val, 
            key=f"nect_{color_sel}_{periodo_sel}_{metrica_sel_n}_max_{i}"
        )
        grp_new = cols[2].selectbox(
            f"Grupo {i+1}:",
            options=[1, 2, 3, 4],
            index=int(grp) - 1 if not math.isnan(grp) else 0,
            key=f"nect_{color_sel}_{periodo_sel}_{metrica_sel_n}_grp_{i}"
        )
        
        # Convertir valores extremos de vuelta a infinito
        if lo_new <= -1e5:
            lo_new = -np.inf
        if hi_new >= 1e5:
            hi_new = np.inf
            
        nuevas_bandas_n.append((lo_new, hi_new, grp_new))
    
    # Botones de acción para nectarina
    col_add_n, col_save_n, col_reset_n = st.columns([1, 1, 1])
    
    with col_add_n:
        if st.button("➕ Agregar banda", key=f"add_nect_{color_sel}_{periodo_sel}_{metrica_sel_n}"):
            last_hi = nuevas_bandas_n[-1][1] if nuevas_bandas_n else 0
            if math.isinf(last_hi):
                last_hi = 10
            nuevas_bandas_n.append((last_hi, last_hi + 5, 4))
            st.rerun()
    
    with col_save_n:
        if st.button("💾 Guardar cambios", key=f"save_nect_{color_sel}_{periodo_sel}_{metrica_sel_n}"):
            current_nect_rules[color_sel][periodo_sel][metrica_sel_n] = nuevas_bandas_n
            st.session_state["nect_rules_df"] = nect_rules_to_df(current_nect_rules)
            st.success(f"✅ Reglas actualizadas para {color_sel} - {periodo_sel} - {metrica_sel_n}")
    
    with col_reset_n:
        if st.button("🔄 Restaurar por defecto", key=f"reset_nect_{color_sel}_{periodo_sel}_{metrica_sel_n}"):
            default_metric = DEFAULT_NECT_RULES.get(color_sel, {}).get(periodo_sel, {}).get(metrica_sel_n, [])
            if default_metric:
                current_nect_rules[color_sel][periodo_sel][metrica_sel_n] = default_metric
                st.session_state["nect_rules_df"] = nect_rules_to_df(current_nect_rules)
                st.success("✅ Reglas restauradas a valores por defecto")
                st.rerun()

# ========================================
# TAB 3: VALORES POR DEFECTO
# ========================================
with tab3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.subheader("⚙️ Valores por Defecto")
    st.markdown("Configura los valores que se utilizarán cuando falten datos en los registros originales.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🍑 Configuración para Ciruelas")
        
        # Tipo de ciruela por defecto
        st.selectbox(
            "Tipo de ciruela por defecto (si el peso no está disponible):",
            options=["sugar", "candy"],
            index=["sugar", "candy"].index(st.session_state["default_plum_subtype"]),
            key="default_plum_subtype",
            help="Cuando no hay información de peso para determinar el tipo de ciruela"
        )
        
        # Peso límite entre sugar y candy
        st.number_input(
            "Peso máximo para clasificar como 'sugar' (gramos):",
            min_value=10.0,
            max_value=200.0,
            value=float(st.session_state["sugar_upper"]),
            step=1.0,
            key="sugar_upper",
            help="Ciruelas con peso menor o igual se clasifican como 'sugar', mayores como 'candy'"
        )
    
    with col2:
        st.markdown("### 🍑 Configuración para Nectarinas")
        
        # Color de pulpa por defecto
        # Normalizar el valor por si hay inconsistencias de mayúsculas
        current_color = st.session_state.get("default_color", "amarilla")
        if isinstance(current_color, str):
            current_color = current_color.lower()
        if current_color not in ["amarilla", "blanca"]:
            current_color = "amarilla"
            
        # Determinar índice de forma segura
        try:
            color_index = ["amarilla", "blanca"].index(current_color)
        except ValueError:
            color_index = 0  # Default a "amarilla" si no se encuentra
            
        st.selectbox(
            "Color de pulpa por defecto (si falta la información):",
            options=["amarilla", "blanca"],
            index=color_index,
            key="default_color",
            help="Color de pulpa a usar cuando no está especificado en los datos"
        )
        
        # Período por defecto
        st.selectbox(
            "Período de cosecha por defecto (si falta fecha):",
            options=["muy_temprana", "temprana", "tardia", "sin_fecha"],
            index=["muy_temprana", "temprana", "tardia", "sin_fecha"].index(st.session_state["default_period"]),
            key="default_period",
            help="Período de cosecha a usar cuando no se puede determinar por la fecha"
        )
    
    st.markdown("---")
    
    # Información adicional
    with st.expander("ℹ️ Información sobre los valores por defecto", expanded=False):
        st.markdown("""
        ### 🔍 ¿Cuándo se usan estos valores?
        
        **Para Ciruelas:**
        - **Tipo por defecto:** Se usa cuando el campo "Peso (g)" está vacío o es nulo
        - **Peso límite:** Define el umbral para clasificar automáticamente entre 'sugar' (≤60g) y 'candy' (>60g)
        
        **Para Nectarinas:**
        - **Color por defecto:** Se usa cuando el campo "Color de pulpa" está vacío o es nulo  
        - **Período por defecto:** Se usa cuando no se puede determinar el período de cosecha por la fecha
        
        ### ⚠️ Importante:
        - Estos valores se aplican durante el procesamiento de segmentación
        - Los cambios se guardan automáticamente en la sesión
        - Para que los cambios tengan efecto, debes volver a ejecutar la segmentación
        """)
    
    st.success("✅ Los valores se guardan automáticamente. Los cambios se aplicarán en la próxima ejecución de segmentación.")

# ===== INFORMACIÓN FINAL =====
st.markdown("---")
st.info("""
💡 **Consejos de uso:**
- Los cambios se guardan automáticamente en la sesión actual
- Para aplicar las nuevas reglas, ejecuta la segmentación de nuevo en las páginas correspondientes
- Los colores representan niveles de calidad: Verde=Excelente, Naranja=Bueno, Coral=Regular, Rojo=Deficiente
- Los valores infinitos (±∞) se representan con números muy grandes/pequeños para facilitar la edición
""")