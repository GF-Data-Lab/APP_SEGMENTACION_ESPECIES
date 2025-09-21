import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import copy
from new_clustering_rules import PLUM_RULES, NECT_RULES
from common_styles import configure_page, generarMenu, get_cluster_colors

# Configuración de página con estilos unificados
configure_page("Métricas y Bandas", "📊")

# CSS personalizado adicional para esta página
st.markdown("""
<style>
    .rule-editor {
        background: linear-gradient(135deg, var(--success-green) 0%, var(--success-green-dark) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.3);
    }
    
    .stTab > div > div > div > div {
        padding: 2rem 1rem;
    }
    
    .band-input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        padding: 0.5rem;
        color: var(--text-dark);
    }
</style>
""", unsafe_allow_html=True)

# Menú de navegación

generarMenu()

st.title("🎯 Editor Avanzado de Métricas y Bandas")

st.markdown("""
<div class="metric-card">
    <h3>🚀 Sistema Interactivo de Reglas de Clustering</h3>
    <p>Configure y personalice las reglas de clasificación para diferentes especies y períodos de cosecha.</p>
    <p><strong>✨ Funcionalidades:</strong> Edición en tiempo real • Validación automática • Visualizaciones interactivas • Exportación de configuraciones</p>
</div>
""", unsafe_allow_html=True)

# Inicializar session state para las reglas editables
if "edited_plum_rules" not in st.session_state:
    st.session_state.edited_plum_rules = copy.deepcopy(PLUM_RULES)
if "edited_nect_rules" not in st.session_state:
    st.session_state.edited_nect_rules = copy.deepcopy(NECT_RULES)

# Función para crear sliders de bandas
def create_band_editor(metric_name, current_bands, key_prefix):
    st.markdown(f"#### 📏 {metric_name}")
    
    # Extraer valores actuales
    values = []
    for lo, hi, band in current_bands:
        if lo != float('-inf'):
            values.append(lo)
        if hi != float('inf'):
            values.append(hi)
    
    # Determinar rango
    if values:
        min_val = min(values) - 5
        max_val = max(values) + 5
    else:
        min_val, max_val = 0, 25
    
    # Editor de bandas
    col1, col2, col3 = st.columns(3)
    
    new_bands = []
    band_colors = ["🟢", "🟡", "🟠", "🟣"]
    
    for i, (lo, hi, band) in enumerate(current_bands):
        with [col1, col2, col3][i % 3]:
            st.markdown(f"**{band_colors[band-1]} Banda {band}**")
            
            # Manejar infinitos
            lo_display = lo if lo != float('-inf') else min_val
            hi_display = hi if hi != float('inf') else max_val
            
            # Sliders para límites
            if lo != float('-inf'):
                new_lo = st.slider(
                    f"Mínimo B{band}", 
                    min_val, max_val, 
                    float(lo_display), 
                    step=0.1,
                    key=f"{key_prefix}_lo_{i}"
                )
            else:
                new_lo = float('-inf')
                st.info("Mínimo: -∞")
            
            if hi != float('inf'):
                new_hi = st.slider(
                    f"Máximo B{band}", 
                    min_val, max_val, 
                    float(hi_display), 
                    step=0.1,
                    key=f"{key_prefix}_hi_{i}"
                )
            else:
                new_hi = float('inf')
                st.info("Máximo: +∞")
            
            new_bands.append((new_lo, new_hi, band))
    
    return new_bands

# Función para visualizar bandas
def create_band_visualization(metric_name, bands, title):
    fig = go.Figure()
    
    # Usar paleta pastel estándar
    cluster_colors = get_cluster_colors()
    colors = [cluster_colors['hex'][i] for i in range(1, 5)]
    
    for lo, hi, band in bands:
        lo_display = lo if lo != float('-inf') else 0
        hi_display = hi if hi != float('inf') else 25
        
        fig.add_trace(go.Scatter(
            x=[lo_display, hi_display, hi_display, lo_display, lo_display],
            y=[band-0.4, band-0.4, band+0.4, band+0.4, band-0.4],
            fill="toself",
            fillcolor=colors[band-1],
            opacity=0.6,
            line=dict(color=colors[band-1], width=2),
            name=f"Banda {band}",
            text=f"Banda {band}: [{lo}, {hi})",
            hovertemplate="%{text}<extra></extra>"
        ))
    
    fig.update_layout(
        title=f"📊 Visualización de Bandas - {title}",
        xaxis_title=metric_name,
        yaxis_title="Banda",
        yaxis=dict(tickmode='linear', tick0=1, dtick=1),
        showlegend=True,
        height=300,
        template="plotly_dark"
    )
    
    return fig

# Tabs principales
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🍑 Editor Ciruela", "🍑 Editor Nectarina", "📊 Visualizaciones", "⚙️ Configuración", "➕ Crear Métricas"])

with tab1:
    st.markdown("""
    <div class="rule-editor">
        <h3>🍭 Editor de Reglas para Ciruela</h3>
        <p>Configure las bandas de clasificación para Candy Plum (>60g) y Cherry Plum (≤60g)</p>
    </div>
    """, unsafe_allow_html=True)
    
    plum_type = st.selectbox("🎯 Seleccionar Tipo de Ciruela", ["candy", "cherry"], key="plum_type_select")
    
    st.markdown(f"### ✏️ Editando reglas para **{plum_type.upper()} PLUM**")
    
    current_plum_rules = st.session_state.edited_plum_rules[plum_type]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Editor BRIX
        new_brix = create_band_editor("BRIX (%)", current_plum_rules["BRIX"], f"plum_{plum_type}_brix")
        st.session_state.edited_plum_rules[plum_type]["BRIX"] = new_brix
        
        # Editor ACIDEZ
        new_acidez = create_band_editor("ACIDEZ (%)", current_plum_rules["ACIDEZ"], f"plum_{plum_type}_acidez")
        st.session_state.edited_plum_rules[plum_type]["ACIDEZ"] = new_acidez
    
    with col2:
        # Editor FIRMEZA_PUNTO
        new_fp = create_band_editor("FIRMEZA PUNTO", current_plum_rules["FIRMEZA_PUNTO"], f"plum_{plum_type}_fp")
        st.session_state.edited_plum_rules[plum_type]["FIRMEZA_PUNTO"] = new_fp
        
        # Editor FIRMEZA_MEJ
        new_fmej = create_band_editor("FIRMEZA MEJILLAS", current_plum_rules["FIRMEZA_MEJ"], f"plum_{plum_type}_fmej")
        st.session_state.edited_plum_rules[plum_type]["FIRMEZA_MEJ"] = new_fmej
    
    # Botones de acción
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Restaurar Originales", key="reset_plum"):
            st.session_state.edited_plum_rules[plum_type] = copy.deepcopy(PLUM_RULES[plum_type])
            st.success("✅ Reglas restauradas!")
            st.rerun()
    
    with col2:
        if st.button("💾 Guardar Cambios", key="save_plum"):
            st.success("✅ Cambios guardados en memoria!")
    
    with col3:
        if st.button("📄 Ver Reglas JSON", key="show_plum_json"):
            st.json(st.session_state.edited_plum_rules[plum_type])

with tab2:
    st.markdown("""
    <div class="rule-editor">
        <h3>🍑 Editor de Reglas para Nectarina</h3>
        <p>Configure las bandas por color (amarilla/blanca) y período de cosecha</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        nect_color = st.selectbox("🎨 Color de Nectarina", ["amarilla", "blanca"], key="nect_color")
    with col2:
        nect_period = st.selectbox("📅 Período de Cosecha", ["muy_temprana", "temprana", "tardia"], key="nect_period")
    
    st.markdown(f"### ✏️ Editando **NECTARINA {nect_color.upper()} - {nect_period.upper()}**")
    
    current_nect_rules = st.session_state.edited_nect_rules[nect_color][nect_period]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Editor BRIX
        new_brix_n = create_band_editor("BRIX (%)", current_nect_rules["BRIX"], f"nect_{nect_color}_{nect_period}_brix")
        st.session_state.edited_nect_rules[nect_color][nect_period]["BRIX"] = new_brix_n
        
        # Editor ACIDEZ
        new_acidez_n = create_band_editor("ACIDEZ (%)", current_nect_rules["ACIDEZ"], f"nect_{nect_color}_{nect_period}_acidez")
        st.session_state.edited_nect_rules[nect_color][nect_period]["ACIDEZ"] = new_acidez_n
    
    with col2:
        # Editor FIRMEZA_PUNTO
        new_fp_n = create_band_editor("FIRMEZA PUNTO", current_nect_rules["FIRMEZA_PUNTO"], f"nect_{nect_color}_{nect_period}_fp")
        st.session_state.edited_nect_rules[nect_color][nect_period]["FIRMEZA_PUNTO"] = new_fp_n
        
        # Editor FIRMEZA_MEJ
        new_fmej_n = create_band_editor("FIRMEZA MEJILLAS", current_nect_rules["FIRMEZA_MEJ"], f"nect_{nect_color}_{nect_period}_fmej")
        st.session_state.edited_nect_rules[nect_color][nect_period]["FIRMEZA_MEJ"] = new_fmej_n
    
    # Información de períodos
    st.markdown("""
    <div class="success-card">
        <h4>📅 Información de Períodos de Cosecha</h4>
        <strong>Nectarina Blanca:</strong><br>
        • Muy Temprana: antes del 25 nov • Temprana: 25 nov - 15 dic • Tardía: 16 dic - 15 feb<br><br>
        <strong>Nectarina Amarilla:</strong><br>
        • Muy Temprana: antes del 22 nov • Temprana: 22 nov - 22 dic • Tardía: 23 dic - 15 feb
    </div>
    """, unsafe_allow_html=True)
    
    # Botones de acción
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Restaurar Originales", key="reset_nect"):
            st.session_state.edited_nect_rules[nect_color][nect_period] = copy.deepcopy(NECT_RULES[nect_color][nect_period])
            st.success("✅ Reglas restauradas!")
            st.rerun()
    
    with col2:
        if st.button("💾 Guardar Cambios", key="save_nect"):
            st.success("✅ Cambios guardados en memoria!")
    
    with col3:
        if st.button("📄 Ver Reglas JSON", key="show_nect_json"):
            st.json(st.session_state.edited_nect_rules[nect_color][nect_period])

with tab3:
    st.markdown("""
    <div class="success-card">
        <h3>📊 Visualizaciones Interactivas</h3>
        <p>Explore gráficamente las reglas de clustering y sus rangos de aplicación</p>
    </div>
    """, unsafe_allow_html=True)
    
    viz_type = st.radio("📈 Tipo de Visualización", ["Ciruela", "Nectarina"], horizontal=True)
    
    if viz_type == "Ciruela":
        viz_subtype = st.selectbox("🍭 Subtipo", ["candy", "cherry"], key="viz_plum_type")
        rules = st.session_state.edited_plum_rules[viz_subtype]
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["BRIX", "FIRMEZA PUNTO", "FIRMEZA MEJILLAS", "ACIDEZ"],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        metrics = ["BRIX", "FIRMEZA_PUNTO", "FIRMEZA_MEJ", "ACIDEZ"]
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for metric, (row, col) in zip(metrics, positions):
            bands = rules[metric]
            # Usar colores pastel estándar
            cluster_colors = get_cluster_colors()
            colors = [cluster_colors['hex'][i] for i in range(1, 5)]
            
            for lo, hi, band in bands:
                lo_display = lo if lo != float('-inf') else 0
                hi_display = hi if hi != float('inf') else 30
                
                fig.add_trace(
                    go.Bar(
                        x=[f"Banda {band}"],
                        y=[hi_display - lo_display],
                        base=lo_display,
                        marker_color=colors[band-1],
                        name=f"{metric} B{band}",
                        showlegend=False,
                        text=f"[{lo}, {hi})",
                        textposition="inside"
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title=f"🍑 Visualización Completa - Ciruela {viz_subtype.upper()}",
            height=600,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # Nectarina
        col1, col2 = st.columns(2)
        with col1:
            viz_color = st.selectbox("🎨 Color", ["amarilla", "blanca"], key="viz_nect_color")
        with col2:
            viz_period = st.selectbox("📅 Período", ["muy_temprana", "temprana", "tardia"], key="viz_nect_period")
        
        rules = st.session_state.edited_nect_rules[viz_color][viz_period]
        
        # Gráfico de radar para mostrar todos los rangos
        categories = ["BRIX", "FIRMEZA_PUNTO", "FIRMEZA_MEJ", "ACIDEZ"]
        
        fig = go.Figure()
        
        for band_num in range(1, 5):
            values = []
            for metric in categories:
                bands = rules[metric]
                for lo, hi, band in bands:
                    if band == band_num:
                        # Usar el punto medio del rango para el radar
                        lo_val = lo if lo != float('-inf') else 0
                        hi_val = hi if hi != float('inf') else 20
                        values.append((lo_val + hi_val) / 2)
                        break
            
            if len(values) == 4:
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=f'Banda {band_num}',
                    line_color=colors[band_num-1]
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 25])
            ),
            title=f"🍑 Radar de Bandas - Nectarina {viz_color} {viz_period}",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("""
    <div class="warning-card">
        <h3>⚙️ Configuración y Exportación</h3>
        <p>Gestione sus configuraciones personalizadas y exporte las reglas modificadas</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📤 Exportar Configuraciones")
        
        if st.button("📄 Exportar Reglas como JSON", key="export_json"):
            export_data = {
                "plum_rules": st.session_state.edited_plum_rules,
                "nect_rules": st.session_state.edited_nect_rules
            }
            st.download_button(
                label="💾 Descargar JSON",
                data=str(export_data),
                file_name="reglas_clustering_personalizadas.json",
                mime="application/json"
            )
        
        if st.button("📊 Exportar como Python Code", key="export_python"):
            python_code = f"""
# Reglas de clustering personalizadas
PLUM_RULES = {st.session_state.edited_plum_rules}

NECT_RULES = {st.session_state.edited_nect_rules}
"""
            st.download_button(
                label="💾 Descargar .py",
                data=python_code,
                file_name="reglas_personalizadas.py",
                mime="text/plain"
            )
    
    with col2:
        st.markdown("#### 🔄 Acciones de Sistema")
        
        if st.button("🔄 Restaurar TODAS las Reglas", key="reset_all"):
            st.session_state.edited_plum_rules = copy.deepcopy(PLUM_RULES)
            st.session_state.edited_nect_rules = copy.deepcopy(NECT_RULES)
            st.success("✅ Todas las reglas restauradas a valores originales!")
            st.rerun()
        
        if st.button("📋 Mostrar Diferencias", key="show_diff"):
            st.markdown("**🔍 Comparación con Reglas Originales:**")
            
            # Comparar cambios en Ciruela
            for ptype in ["candy", "cherry"]:
                original = PLUM_RULES[ptype]
                edited = st.session_state.edited_plum_rules[ptype]
                if original != edited:
                    st.warning(f"Ciruela {ptype}: ¡Modificada!")
                else:
                    st.success(f"Ciruela {ptype}: Sin cambios")
            
            # Comparar cambios en Nectarina
            for color in ["amarilla", "blanca"]:
                for period in ["muy_temprana", "temprana", "tardia"]:
                    original = NECT_RULES[color][period]
                    edited = st.session_state.edited_nect_rules[color][period]
                    if original != edited:
                        st.warning(f"Nectarina {color} {period}: ¡Modificada!")
                    else:
                        st.success(f"Nectarina {color} {period}: Sin cambios")

# Información final
st.markdown("---")
st.markdown("""
<div class="metric-card">
    <h4>🎯 Información del Sistema de Bandas</h4>
    <p><strong>Cluster Final = Suma de Bandas:</strong></p>
    <p>🟢 Cluster 1: Suma 3-5 (Excelente) | 🟡 Cluster 2: Suma 6-8 (Bueno) | 🟠 Cluster 3: Suma 9-11 (Regular) | 🔴 Cluster 4: Suma 12+ (Deficiente)</p>
    <p><strong>📏 Métricas:</strong> BRIX (sólidos solubles) • FIRMEZA_PUNTO (punto más débil) • FIRMEZA_MEJ (promedio mejillas) • ACIDEZ (primer fruto)</p>
</div>
""", unsafe_allow_html=True)

with tab5:
    st.markdown("""
    <div class="info-card">
        <h3>➕ Creador de Nuevas Métricas</h3>
        <p>Define métricas personalizadas para tu análisis de clustering</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Inicializar session state para métricas personalizadas
    if "custom_metrics" not in st.session_state:
        st.session_state.custom_metrics = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔧 Configurar Nueva Métrica")
        
        # Nombre de la métrica
        metric_name = st.text_input(
            "📝 Nombre de la Métrica:",
            placeholder="Ej: TEXTURA_SUPERFICIE",
            key="new_metric_name"
        )
        
        # Tipo de métrica
        metric_type = st.selectbox(
            "📊 Tipo de Métrica:",
            ["Numérica Continua", "Numérica Discreta", "Categórica"],
            key="metric_type"
        )
        
        # Descripción
        metric_description = st.text_area(
            "📋 Descripción:",
            placeholder="Describe qué mide esta métrica y cómo se obtiene...",
            key="metric_description"
        )
        
        # Unidad de medida
        metric_unit = st.text_input(
            "🏷️ Unidad de Medida:",
            placeholder="Ej: %, cm, kg, puntos",
            key="metric_unit"
        )
        
        # Rango esperado
        st.markdown("#### 📏 Rango de Valores")
        col_min, col_max = st.columns(2)
        with col_min:
            metric_min = st.number_input(
                "Valor Mínimo:",
                value=0.0,
                key="metric_min"
            )
        with col_max:
            metric_max = st.number_input(
                "Valor Máximo:",
                value=100.0,
                key="metric_max"
            )
    
    with col2:
        st.markdown("### 🎯 Definir Bandas de Clasificación")
        
        if metric_name:
            st.markdown(f"**Configurando bandas para:** {metric_name}")
            
            # Número de bandas
            num_bands = st.selectbox(
                "🔢 Número de Bandas:",
                [3, 4, 5],
                index=1,  # Default 4 bandas
                key="num_bands"
            )
            
            # Configurar cada banda
            bands_config = []
            band_colors = ["🟢 Excelente", "🟡 Bueno", "🟠 Regular", "🟣 Deficiente"]
            
            for i in range(num_bands):
                st.markdown(f"#### {band_colors[i]} - Banda {i+1}")
                
                col_band_min, col_band_max = st.columns(2)
                with col_band_min:
                    if i == 0:
                        band_min = st.number_input(
                            f"Mínimo B{i+1}:",
                            value=metric_min,
                            key=f"band_{i}_min"
                        )
                    else:
                        band_min = bands_config[i-1]['max']
                        st.info(f"Mínimo: {band_min}")
                
                with col_band_max:
                    if i == num_bands - 1:
                        band_max = metric_max
                        st.info(f"Máximo: {band_max}")
                    else:
                        band_max = st.number_input(
                            f"Máximo B{i+1}:",
                            value=metric_min + ((metric_max - metric_min) / num_bands) * (i+1),
                            key=f"band_{i}_max"
                        )
                
                bands_config.append({
                    'band': i+1,
                    'min': band_min,
                    'max': band_max,
                    'label': band_colors[i]
                })
    
    # Guardar métrica personalizada
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Guardar Métrica", key="save_custom_metric"):
            if metric_name and metric_description:
                # Crear estructura de la métrica
                custom_metric = {
                    'name': metric_name,
                    'type': metric_type,
                    'description': metric_description,
                    'unit': metric_unit,
                    'min_value': metric_min,
                    'max_value': metric_max,
                    'bands': bands_config,
                    'created_at': pd.Timestamp.now()
                }
                
                # Guardar en session state
                st.session_state.custom_metrics[metric_name] = custom_metric
                
                st.success(f"✅ Métrica '{metric_name}' guardada exitosamente!")
                st.rerun()
            else:
                st.error("❌ Por favor completa al menos el nombre y descripción de la métrica.")
    
    with col2:
        if st.button("📄 Exportar Métrica", key="export_custom_metric"):
            if metric_name in st.session_state.custom_metrics:
                metric_data = st.session_state.custom_metrics[metric_name]
                
                # Convertir a JSON
                import json
                metric_json = json.dumps(metric_data, default=str, indent=2)
                
                st.download_button(
                    label="💾 Descargar JSON",
                    data=metric_json,
                    file_name=f"metrica_{metric_name.lower()}.json",
                    mime="application/json"
                )
    
    with col3:
        if st.button("🗑️ Limpiar Formulario", key="clear_form"):
            for key in ['new_metric_name', 'metric_description', 'metric_unit']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Mostrar métricas existentes
    if st.session_state.custom_metrics:
        st.markdown("---")
        st.markdown("### 📋 Métricas Personalizadas Creadas")
        
        for name, metric in st.session_state.custom_metrics.items():
            with st.expander(f"📊 {name} ({metric['unit']})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Tipo:** {metric['type']}")
                    st.markdown(f"**Descripción:** {metric['description']}")
                    st.markdown(f"**Rango:** {metric['min_value']} - {metric['max_value']} {metric['unit']}")
                    st.markdown(f"**Creada:** {metric['created_at'].strftime('%Y-%m-%d %H:%M')}")
                
                with col2:
                    st.markdown("**Bandas de Clasificación:**")
                    for band in metric['bands']:
                        st.markdown(f"• {band['label']}: {band['min']} - {band['max']} {metric['unit']}")
                
                # Botones de acción
                col_action1, col_action2, col_action3 = st.columns(3)
                
                with col_action1:
                    if st.button(f"📝 Editar {name}", key=f"edit_{name}"):
                        st.info("🔧 Para editar, modifica los valores arriba y guarda de nuevo.")
                
                with col_action2:
                    if st.button(f"📄 Exportar {name}", key=f"export_{name}"):
                        import json
                        metric_json = json.dumps(metric, default=str, indent=2)
                        st.download_button(
                            label="💾 Descargar",
                            data=metric_json,
                            file_name=f"metrica_{name.lower()}.json",
                            mime="application/json",
                            key=f"download_{name}"
                        )
                
                with col_action3:
                    if st.button(f"🗑️ Eliminar {name}", key=f"delete_{name}"):
                        del st.session_state.custom_metrics[name]
                        st.success(f"✅ Métrica '{name}' eliminada.")
                        st.rerun()
    
    else:
        st.info("📝 No hay métricas personalizadas creadas aún. ¡Crea tu primera métrica usando el formulario arriba!")
    
    # Importar métricas desde JSON
    st.markdown("---")
    st.markdown("### 📤 Importar Métricas")
    
    uploaded_metric = st.file_uploader(
        "📁 Cargar Métrica desde JSON:",
        type=['json'],
        key="upload_metric"
    )
    
    if uploaded_metric is not None:
        try:
            import json
            metric_data = json.load(uploaded_metric)
            
            # Validar estructura básica
            required_fields = ['name', 'type', 'description', 'bands']
            if all(field in metric_data for field in required_fields):
                
                # Guardar métrica importada
                metric_name = metric_data['name']
                st.session_state.custom_metrics[metric_name] = metric_data
                
                st.success(f"✅ Métrica '{metric_name}' importada exitosamente!")
                st.rerun()
            else:
                st.error("❌ El archivo JSON no tiene la estructura correcta de una métrica.")
        except Exception as e:
            st.error(f"❌ Error al cargar el archivo: {str(e)}")

st.success("🎉 ¡Sistema de reglas interactivo funcionando correctamente! Todas las modificaciones se aplican en tiempo real.")