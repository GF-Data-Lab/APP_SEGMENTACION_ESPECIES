# 🎨 Referencia de Colores Pastel Estándar

## 📋 Paleta de Colores Unificada

Esta guía documenta la paleta de colores pastel estándar utilizada en toda la aplicación para garantizar consistencia visual y accesibilidad.

### 🎯 **Clusters de Calidad**

| Cluster | Nombre | Color Sólido | Hex | Background | Text | Descripción |
|---------|--------|--------------|-----|------------|------|-------------|
| **1** | 🟢 Excelente | ![#90EE90](https://via.placeholder.com/20/90EE90/000000?text=+) | `#90EE90` | `#E8F5E8` | `#2D5016` | Verde claro pastel |
| **2** | 🟡 Bueno | ![#FFB347](https://via.placeholder.com/20/FFB347/000000?text=+) | `#FFB347` | `#FFF4E6` | `#8B4000` | Naranja durazno pastel |
| **3** | 🟠 Regular | ![#FFA07A](https://via.placeholder.com/20/FFA07A/000000?text=+) | `#FFA07A` | `#FFE4E1` | `#8B0000` | Salmón claro pastel |
| **4** | 🟣 Deficiente | ![#DDA0DD](https://via.placeholder.com/20/DDA0DD/000000?text=+) | `#DDA0DD` | `#F0E6FF` | `#4B0082` | Ciruela pastel |

### 📊 **Colores para Gráficos Plotly**

| Cluster | Color Plotly | Hex | Uso |
|---------|--------------|-----|-----|
| **1** | ![#7FB069](https://via.placeholder.com/20/7FB069/000000?text=+) | `#7FB069` | Verde sage para scatter plots |
| **2** | ![#FF9F40](https://via.placeholder.com/20/FF9F40/000000?text=+) | `#FF9F40` | Naranja cálido para visualizaciones |
| **3** | ![#FF6B6B](https://via.placeholder.com/20/FF6B6B/000000?text=+) | `#FF6B6B` | Coral para datos regulares |
| **4** | ![#A8A8FF](https://via.placeholder.com/20/A8A8FF/000000?text=+) | `#A8A8FF` | Lavanda suave para deficientes |

### 🖱️ **Colores de Hover/Interacción**

| Cluster | Color Hover | Hex | Descripción |
|---------|-------------|-----|-------------|
| **1** | ![#98FB98](https://via.placeholder.com/20/98FB98/000000?text=+) | `#98FB98` | Verde pálido para hover |
| **2** | ![#FFDAB9](https://via.placeholder.com/20/FFDAB9/000000?text=+) | `#FFDAB9` | Durazno suave para hover |
| **3** | ![#FFB6C1](https://via.placeholder.com/20/FFB6C1/000000?text=+) | `#FFB6C1` | Rosa claro para hover |
| **4** | ![#E6E6FA](https://via.placeholder.com/20/E6E6FA/000000?text=+) | `#E6E6FA` | Lavanda para hover |

## 🔧 **Implementación en el Código**

### Importar Colores
```python
from common_styles import get_cluster_colors, get_plotly_color_map, get_plotly_color_sequence
```

### Usar en DataFrames
```python
# Para colorear tablas
style_function = get_cluster_style_function()
df.style.map(style_function, subset=['cluster'])
```

### Usar en Plotly
```python
# Para gráficos con color por cluster
color_map = get_plotly_color_map()
fig = px.scatter(df, x='x', y='y', color='cluster', color_discrete_map=color_map)

# Para secuencia de colores
colors = get_plotly_color_sequence()
```

### Usar Colores Individuales
```python
colors = get_cluster_colors()

# Color sólido para elementos destacados
solid_color = colors['solid'][1]  # Verde pastel

# Color de fondo para containers
bg_color = colors['background'][1]  # Verde muy suave

# Color de texto contrastante
text_color = colors['text'][1]  # Verde oscuro
```

## 📋 **Archivos Actualizados**

### ✅ Archivos con Colores Estandarizados:

1. **`common_styles.py`** - Paleta central y funciones helper
2. **`segmentacion_base.py`** - Gráficos PCA y visualizaciones principales
3. **`pages/verificar_calculos.py`** - Colores de bandas y clusters
4. **`pages/metricas_bandas.py`** - Editor de métricas y visualizaciones
5. **`pages/evolucion_variedad.py`** - Análisis temporal y comparativas
6. **`pages/outliers.py`** - Gráficos de distribución y detección

### 🎨 **Consistencia Visual Garantizada:**

- ✅ **Tablas:** Colores de fondo suaves con texto contrastante
- ✅ **Gráficos Plotly:** Paleta armoniosa y accesible
- ✅ **Métricas:** Indicadores visuales consistentes
- ✅ **Bandas:** Colores representativos de calidad
- ✅ **Hover States:** Interacciones suaves y naturales

## 🌈 **Principios de Diseño**

### 🎯 **Accesibilidad**
- Contraste suficiente para lectura (WCAG 2.1 AA)
- Colores distinguibles para personas con daltonismo
- Significado no dependiente únicamente del color

### 🎨 **Armonía Visual**
- Paleta basada en tonos pastel suaves
- Transiciones naturales entre clusters de calidad
- Coherencia con identidad visual científica

### 📊 **Funcionalidad**
- Colores semánticamente correctos (verde=bueno, rojo=malo)
- Escalabilidad para diferentes tipos de gráficos
- Facilidad de identificación en visualizaciones complejas

## 🔄 **Migración de Colores Anteriores**

### Mapeo de Colores Antiguos → Nuevos:

| Anterior | Nuevo | Mejora |
|----------|-------|--------|
| `#10b981` → | `#90EE90` | Más suave, mejor contraste |
| `#f59e0b` → | `#FFB347` | Tono más cálido y accesible |
| `#ef4444` → | `#FFA07A` | Menos agresivo, más profesional |
| `#8b5cf6` → | `#DDA0DD` | Más distinguible, mejor lectura |

## 🚀 **Uso Recomendado**

### Para Desarrolladores:
1. **Siempre** usar funciones de `common_styles.py`
2. **Nunca** hardcodear colores hex directamente
3. **Probar** accesibilidad con herramientas de contraste
4. **Verificar** consistencia en diferentes dispositivos

### Para Usuarios:
- Los colores ahora son más suaves y profesionales
- Mejor legibilidad en pantallas diversas
- Consistencia visual en toda la aplicación
- Experiencia más agradable y menos fatiga visual

---

**Actualización:** Septiembre 2024
**Responsable:** Equipo de Desarrollo - Garces Data Analytics
**Versión:** 2.1 - Paleta Pastel Unificada