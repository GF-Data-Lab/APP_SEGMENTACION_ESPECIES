# 🍑 Sistema de Segmentación y Análisis de Especies de Carozos

Esta aplicación avanzada de **Streamlit** está diseñada para procesar y analizar datos de calidad de frutos de carozo (ciruelas, nectarinas, duraznos, etc.), implementando algoritmos de **segmentación automática**, **detección de outliers** y **clustering inteligente** para la clasificación de variedades.

## 🎯 Características Principales

- ✅ **Análisis multivariado** de firmeza, acidez, sólidos solubles y color
- ✅ **Clustering automático** con algoritmos de k-means y PCA  
- ✅ **Detección de outliers** usando Z-Score e IQR
- ✅ **Reglas configurables** para diferentes especies y periodos
- ✅ **Filtrado inteligente** de datos atípicos
- ✅ **Visualizaciones interactivas** con gráficos dinámicos
- ✅ **Navegación intuitiva** con guía paso a paso
- ✅ **Exportación de resultados** en Excel/CSV

## 📋 Estructura del Proyecto

### Páginas Principales:
- **🏚️ Página de Inicio:** Descripción del proyecto y guía de navegación
- **📁 Carga de Archivos:** Importar datos de Excel
- **🎯 Detección de Outliers:** Identificar y filtrar datos atípicos  
- **🍑 Segmentación Ciruela:** Análisis específico para ciruelas
- **🍑 Segmentación Nectarina:** Análisis específico para nectarinas
- **🤖 Modelo de Clasificación:** Modelos predictivos de machine learning
- **🔍 Análisis Exploratorio:** PCA, clustering avanzado y visualizaciones
- **📊 Métricas y Bandas:** Configuración de reglas de clasificación

### Archivos Técnicos:
- `app.py` - Página principal y navegación
- `segmentacion_base.py` - Lógica central de procesamiento
- `utils.py` - Utilidades compartidas (logo, etc.)
- `requirements.txt` - Dependencias de Python
- `pages/` - Directorio con todas las páginas de la aplicación

## 🚀 Instalación

### 1. Clonar/Descargar el Proyecto
```bash
git clone [URL_DEL_REPOSITORIO]
cd APP_SEGMENTACION_ESPECIES
```

### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 3. Problemas Comunes de Instalación

Si `scikit-learn-extra` falla durante la instalación:

```bash
# Opción 1: Instalar dependencias principales sin scikit-learn-extra
pip install streamlit pandas numpy plotly openpyxl altair xlsxwriter jinja2 scikit-learn

# Opción 2: Si hay problemas con otros paquetes
pip install --no-cache-dir -r requirements.txt
```

**Soluciones adicionales:**
1. Cierra cualquier proceso de Python/IDE en ejecución
2. Elimina carpetas temporales de pip si aparecen errores
3. Desactiva temporalmente el antivirus durante la instalación
4. Reinicia la máquina si persisten los problemas

## 🎮 Uso de la Aplicación

### Iniciar el Servidor
```bash
streamlit run app.py
```

La aplicación estará disponible en: **http://localhost:8501**

### 📊 Flujo de Trabajo Recomendado

#### Para Usuarios Nuevos:
1. **📁 Carga de Archivos** → Subir "MAESTRO CAROZOS FINAL COMPLETO CG.xlsx"
2. **🎯 Detección Outliers** → Limpiar datos atípicos
3. **🍑 Segmentación [Especie]** → Procesar y analizar
4. **📊 Métricas y Bandas** → Ajustar reglas (opcional)
5. **🔍 Análisis Exploratorio** → Explorar resultados

#### Para Usuarios Avanzados:
1. **📊 Métricas y Bandas** → Configurar reglas personalizadas
2. **📁 Carga de Archivos** → Subir datos
3. **🎯 Detección Outliers** → Configurar filtros específicos
4. **🍑 Segmentación [Especie]** → Ejecutar con configuración avanzada
5. **🤖 Modelo de Clasificación** → Entrenar modelos predictivos
6. **🔍 Análisis Exploratorio** → Clustering y PCA avanzado

## 📁 Formato de Datos

### Archivo de Entrada: `MAESTRO CAROZOS FINAL COMPLETO CG.xlsx`

**Hoja requerida:** `CAROZOS`  
**Columnas esperadas (A:AP):**
- Especie, Variedad, PMG, Localidad, Temporada
- Fecha cosecha, Fecha evaluación, Periodo de almacenaje
- Datos de firmeza: Punta, Quilla, Hombro, Mejilla 1, Mejilla 2
- Sólidos solubles (%), Acidez (%)
- Color de pulpa, Peso (g), etc.

**Especies soportadas:**
- Ciruela (tipos: Candy, Sugar)
- Nectarina (colores: Amarilla, Blanca)
- Durazno, Paraguayo, Damasco

## 🔧 Configuración Técnica

### Parámetros de Clustering:
- **Método individual:** Suma o media de métricas de calidad
- **Método grupal:** Media o moda por grupo
- **Clusters:** 4 niveles de calidad (1=Excelente, 4=Deficiente)

### Detección de Outliers:
- **Z-Score:** Umbral configurable (1.0-4.0)
- **IQR:** Factor configurable (1.0-3.0)
- **Filtrado:** Por variable individual o combinado

### Reglas de Clasificación:
- **Configurables por especie** y periodo de cosecha
- **Métricas:** BRIX, firmeza (punto/mejillas), acidez
- **Bandas personalizables** con umbrales min/max

## 📊 Interpretación de Resultados

### Clusters (1-4):
- **Cluster 1:** Calidad excelente (valores óptimos)
- **Cluster 2:** Calidad buena (valores aceptables)
- **Cluster 3:** Calidad regular (valores por debajo del estándar)
- **Cluster 4:** Calidad deficiente (valores problemáticos)

### Visualizaciones:
- **Gráficos PCA:** Muestran agrupaciones naturales en 2D
- **Boxplots:** Identifican outliers y distribuciones
- **Histogramas:** Distribución de valores por variable
- **Scatter plots:** Correlaciones entre variables

### Exportación:
- **Excel:** Resultados completos con múltiples hojas
- **CSV:** Datos filtrados para análisis externos
- **Configuraciones:** Reglas y parámetros guardados automáticamente

## 🛠️ Desarrollo y Personalización

### Agregar Nuevas Especies:
1. Actualizar `DEFAULT_PLUM_RULES` o `DEFAULT_NECT_RULES` en `segmentacion_base.py`
2. Modificar función `_classify_row()` para incluir nueva lógica
3. Añadir opciones en la página de configuración

### Modificar Algoritmos:
- **Clustering:** Editar funciones de discretización en `segmentacion_base.py`
- **Outliers:** Ampliar métodos en `pages/outliers.py`
- **Visualizaciones:** Modificar gráficos en `pages/analisis.py`

## 📞 Soporte

### Logs y Debugging:
- Los logs de Streamlit aparecen en la consola
- Errores comunes se muestran en la interfaz web
- Estado del sistema visible en la página de inicio

### Archivos de Configuración:
- Las configuraciones se guardan en `st.session_state`
- Las reglas personalizadas persisten durante la sesión
- Los datos filtrados se mantienen entre páginas

---

## 🎓 Créditos y Licencia

**Desarrollado para:** Análisis de calidad de frutos de carozo  
**Tecnologías:** Streamlit, pandas, scikit-learn, plotly  
**Versión:** 2.0.0 (Sistema completo con navegación mejorada)

---

🚀 **¡Inicia tu análisis!** Ejecuta `streamlit run app.py` y navega a la página de inicio para comenzar.

