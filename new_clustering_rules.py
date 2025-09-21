"""
Módulo de reglas de clustering optimizadas para segmentación de carozos.

Este módulo implementa las reglas de clasificación específicas para diferentes
especies de frutos de carozo (ciruelas y nectarinas), basadas en parámetros
de calidad como Brix, firmeza y acidez.

Las reglas están organizadas por:
- Especie (Ciruela, Nectarina)
- Tipo/Color (Candy/Cherry, Amarilla/Blanca)
- Periodo de cosecha (Muy temprana, Temprana, Tardía)

Cada regla define umbrales para clasificar frutos en 4 clusters de calidad:
1. Excelente
2. Bueno
3. Regular
4. Deficiente

Autor: Equipo de Análisis de Datos
Fecha: Septiembre 2024
Versión: 2.0 - Reglas optimizadas basadas en validación
"""

import numpy as np
import pandas as pd
from data_columns import COL_FECHA_EVALUACION, COL_PORTAINJERTO, COL_CAMPO

# ============================================================================
# REGLAS DE CLASIFICACIÓN POR BANDAS PARA CIRUELAS
# ============================================================================

# Las reglas definen umbrales para clasificar frutos en 4 clusters de calidad.
# Formato: [(límite_inferior, límite_superior, cluster), ...]
# Los clusters van de 1 (excelente) a 4 (deficiente)

PLUM_RULES = {
    # Reglas para Candy Plum (ciruelas grandes, peso > 60g)
    "candy": {
        # Sólidos solubles - Indicador de dulzor y madurez
        "BRIX": [
            (18.0, float('inf'), 1),  # Excelente: ≥ 18%
            (16.0, 18.0, 2),          # Bueno: 16-18%
            (14.0, 16.0, 3),          # Regular: 14-16%
            (float('-inf'), 14.0, 4)   # Deficiente: < 14%
        ],
        # Firmeza en zona apical - Resistencia al transporte
        "FIRMEZA_PUNTO": [
            (7.0, float('inf'), 1),   # Excelente: ≥ 7N
            (5.0, 7.0, 2),            # Bueno: 5-7N
            (4.0, 5.0, 3),            # Regular: 4-5N
            (float('-inf'), 4.0, 4)    # Deficiente: < 4N
        ],
        # Firmeza en mejilla - Indicador de vida postcosecha
        "FIRMEZA_MEJ": [
            (9.0, float('inf'), 1),   # Excelente: ≥ 9N
            (7.0, 9.0, 2),            # Bueno: 7-9N
            (6.0, 7.0, 3),            # Regular: 6-7N
            (float('-inf'), 6.0, 4)    # Deficiente: < 6N
        ],
        # Acidez - Balance de sabor y conservación
        "ACIDEZ": [
            (float('-inf'), 0.60, 1), # Excelente: ≤ 0.60%
            (0.60, 0.81, 2),          # Bueno: 0.60-0.81%
            (0.81, 1.00, 3),          # Regular: 0.81-1.00%
            (1.00, float('inf'), 4)    # Deficiente: > 1.00%
        ],
    },
    # Reglas para Cherry Plum (ciruelas pequeñas, peso ≤ 60g)
    "cherry": {
        # Sólidos solubles - Umbrales más altos para cherry
        "BRIX": [
            (21.0, float('inf'), 1),  # Excelente: ≥ 21%
            (18.0, 21.0, 2),          # Bueno: 18-21%
            (15.0, 18.0, 3),          # Regular: 15-18%
            (float('-inf'), 15.0, 4)   # Deficiente: < 15%
        ],
        # Firmeza punta - Ligeramente menor que candy
        "FIRMEZA_PUNTO": [
            (6.0, float('inf'), 1),   # Excelente: ≥ 6N
            (4.5, 6.0, 2),            # Bueno: 4.5-6N
            (3.0, 4.5, 3),            # Regular: 3-4.5N
            (float('-inf'), 3.0, 4)    # Deficiente: < 3N
        ],
        # Firmeza mejilla - Ajustada para tamaño menor
        "FIRMEZA_MEJ": [
            (6.0, float('inf'), 1),   # Excelente: ≥ 6N
            (5.0, 6.0, 2),            # Bueno: 5-6N
            (4.0, 5.0, 3),            # Regular: 4-5N
            (float('-inf'), 4.0, 4)    # Deficiente: < 4N
        ],
        # Acidez - Mismos umbrales que candy
        "ACIDEZ": [
            (float('-inf'), 0.60, 1), # Excelente: ≤ 0.60%
            (0.60, 0.81, 2),          # Bueno: 0.60-0.81%
            (0.81, 1.00, 3),          # Regular: 0.81-1.00%
            (1.00, float('inf'), 4)    # Deficiente: > 1.00%
        ],
    },
}

def mk_nect(b1: float, b2: float, b3: float, m1: float, m2: float) -> dict:
    """
    Genera reglas de clasificación para nectarinas con parámetros personalizados.

    Esta función helper crea un diccionario de reglas de clasificación
    para nectarinas usando umbrales específicos de Brix y firmeza.

    Args:
        b1 (float): Umbral superior de Brix para cluster 1
        b2 (float): Umbral medio-alto de Brix para cluster 2
        b3 (float): Umbral medio-bajo de Brix para cluster 3
        m1 (float): Umbral superior de firmeza mejilla para cluster 1
        m2 (float): Umbral medio de firmeza mejilla para cluster 2

    Returns:
        dict: Diccionario con reglas de clasificación por parámetro

    Example:
        >>> rules = mk_nect(14.0, 12.0, 10.0, 14.0, 12.0)
        >>> # Genera reglas para nectarina tardía amarilla
    """
    return {
        # Sólidos solubles - Parámetros variables según periodo
        "BRIX": [
            (b1, float("inf"), 1),    # Excelente: ≥ b1%
            (b2, b1, 2),              # Bueno: b2-b1%
            (b3, b2, 3),              # Regular: b3-b2%
            (float("-inf"), b3, 4)    # Deficiente: < b3%
        ],
        # Firmeza punta - Estándar para todas las nectarinas
        "FIRMEZA_PUNTO": [
            (9.0, float("inf"), 1),   # Excelente: ≥ 9N
            (8.0, 9.0, 2),            # Bueno: 8-9N
            (7.0, 8.0, 3),            # Regular: 7-8N
            (float("-inf"), 7.0, 4)   # Deficiente: < 7N
        ],
        # Firmeza mejilla - Variable según color y periodo
        "FIRMEZA_MEJ": [
            (m1, float("inf"), 1),    # Excelente: ≥ m1N
            (m2, m1, 2),              # Bueno: m2-m1N
            (9.0, m2, 3),             # Regular: 9-m2N
            (float("-inf"), 9.0, 4)   # Deficiente: < 9N
        ],
        # Acidez - Estándar para todas las nectarinas
        "ACIDEZ": [
            (float("-inf"), 0.60, 1), # Excelente: ≤ 0.60%
            (0.60, 0.81, 2),          # Bueno: 0.60-0.81%
            (0.81, 1.00, 3),          # Regular: 0.81-1.00%
            (1.00, float("inf"), 4)   # Deficiente: > 1.00%
        ],
    }

# ============================================================================
# REGLAS DE CLASIFICACIÓN POR BANDAS PARA NECTARINAS
# ============================================================================

# Las nectarinas se clasifican por:
# 1. Color de pulpa: Amarilla vs Blanca
# 2. Periodo de cosecha: Muy temprana, Temprana, Tardía
#
# Los parámetros varían según estas categorías debido a:
# - Diferencias varietales en acumulación de azúcares
# - Variación estacional en firmeza
# - Características de color específicas

NECT_RULES = {
    # Nectarinas de pulpa amarilla
    "amarilla": {
        # Muy temprana (antes 22 nov) - Menor acumulación de Brix
        "muy_temprana": mk_nect(
            b1=13.0, b2=10.0, b3=9.0,    # Brix: 13/10/9%
            m1=14.0, m2=12.0              # Firmeza mejilla: 14/12N
        ),
        # Temprana (22 nov - 22 dic) - Acumulación intermedia
        "temprana": mk_nect(
            b1=13.0, b2=11.0, b3=9.0,    # Brix: 13/11/9%
            m1=14.0, m2=12.0              # Firmeza mejilla: 14/12N
        ),
        # Tardía (23 dic - 15 feb) - Máxima acumulación
        "tardia": mk_nect(
            b1=14.0, b2=12.0, b3=10.0,   # Brix: 14/12/10%
            m1=14.0, m2=12.0              # Firmeza mejilla: 14/12N
        ),
    },
    # Nectarinas de pulpa blanca - Firmeza generalmente menor
    "blanca": {
        # Muy temprana - Umbrales ajustados para pulpa blanca
        "muy_temprana": mk_nect(
            b1=13.0, b2=10.0, b3=9.0,    # Brix: igual que amarilla
            m1=13.0, m2=11.0              # Firmeza mejilla: menor
        ),
        # Temprana - Características intermedias
        "temprana": mk_nect(
            b1=13.0, b2=11.0, b3=9.0,    # Brix: igual que amarilla
            m1=13.0, m2=11.0              # Firmeza mejilla: menor
        ),
        # Tardía - Máxima diferenciación
        "tardia": mk_nect(
            b1=14.0, b2=12.0, b3=10.0,   # Brix: igual que amarilla
            m1=13.0, m2=11.0              # Firmeza mejilla: menor
        ),
    },
}

def clasificar_valor(val: float, bands: list) -> int:
    """
    Clasifica un valor numérico según bandas de umbrales predefinidas.

    Esta función toma un valor y lo evalúa contra una lista de bandas
    para determinar en qué cluster de calidad debe clasificarse.

    Args:
        val (float): Valor a clasificar (ej: Brix, firmeza, acidez)
        bands (list): Lista de tuplas (límite_inferior, límite_superior, cluster)
                     donde cluster va de 1 (mejor) a 4 (peor)

    Returns:
        int: Número de cluster (1-4) o np.nan si no aplica

    Example:
        >>> bands = [(18.0, float('inf'), 1), (16.0, 18.0, 2),
        ...          (14.0, 16.0, 3), (float('-inf'), 14.0, 4)]
        >>> clasificar_valor(19.5, bands)  # Retorna 1 (excelente)
        >>> clasificar_valor(15.2, bands)  # Retorna 3 (regular)

    Note:
        - La función evalúa val >= lo and val < hi
        - Si el valor no cae en ninguna banda, retorna np.nan
        - Los valores NaN de entrada retornan np.nan
    """
    # Validar entrada
    if pd.isna(val) or not bands:
        return np.nan

    # Evaluar contra cada banda en orden
    for lo, hi, grp in bands:
        if lo <= val < hi:
            return grp

    # Si no cae en ninguna banda
    return np.nan

def determinar_harvest_period(color: str, fecha) -> str:
    """
    Determina el período de cosecha para nectarinas según color y fecha.

    Las nectarinas se clasifican en tres períodos de cosecha basados en
    la fecha de evaluación y el color de pulpa, ya que las variedades
    blancas y amarillas tienen calendarios ligeramente diferentes.

    Args:
        color (str): Color de pulpa ('blanca', 'amarilla', u otro)
        fecha: Fecha de cosecha/evaluación (cualquier formato pandas-compatible)

    Returns:
        str: Período de cosecha:
             - 'muy_temprana': Inicio de temporada
             - 'temprana': Temporada media-temprana
             - 'tardia': Temporada tardía (incluye casos sin fecha)

    Note:
        Calendario para pulpa blanca:
        - Muy temprana: antes del 25 noviembre
        - Temprana: 25 nov - 15 dic
        - Tardía: 16 dic - 15 feb

        Calendario para pulpa amarilla (y otros colores):
        - Muy temprana: antes del 22 noviembre
        - Temprana: 22 nov - 22 dic
        - Tardía: 23 dic - 15 feb

    Example:
        >>> determinar_harvest_period('blanca', '2023-11-20')
        'muy_temprana'
        >>> determinar_harvest_period('amarilla', '2023-12-25')
        'tardia'
    """
    # Convertir fecha a datetime, manejar errores
    fecha = pd.to_datetime(fecha, errors='coerce')
    if pd.isna(fecha):
        return "tardia"  # Default para fechas inválidas

    # Extraer mes y día para evaluación
    m, d = fecha.month, fecha.day

    # Calendario específico para pulpa blanca
    if str(color).lower().strip() == "blanca":
        if (m, d) < (11, 25):
            return "muy_temprana"
        elif (11, 25) <= (m, d) <= (12, 15):
            return "temprana"
        elif (12, 16) <= (m, d) <= (2, 15):
            return "tardia"
    else:
        # Calendario para pulpa amarilla y otros
        if (m, d) < (11, 22):
            return "muy_temprana"
        elif (11, 22) <= (m, d) <= (12, 22):
            return "temprana"
        elif (12, 23) <= (m, d) <= (2, 15):
            return "tardia"

    # Default para fechas fuera de rango
    return "tardia"

def determinar_categoria_ciruela(especie: str, peso_promedio: float) -> tuple:
    """
    Determina la categoría de ciruela basada en el peso promedio del fruto.

    Las ciruelas se clasifican en dos categorías principales según su peso:
    - Candy plum: Frutos grandes (> 60g) con características comerciales premium
    - Cherry plum: Frutos pequeños (≤ 60g) con diferentes estándares de calidad

    Args:
        especie (str): Nombre de la especie (debe ser 'ciruela')
        peso_promedio (float): Peso promedio del fruto en gramos

    Returns:
        tuple: (nombre_categoria, codigo_categoria) donde:
               - nombre_categoria: Descripción completa
               - codigo_categoria: Código para lookup de reglas ('candy'/'cherry')

    Example:
        >>> determinar_categoria_ciruela('ciruela', 75.5)
        ('Candy plum', 'candy')
        >>> determinar_categoria_ciruela('ciruela', 45.2)
        ('Cherry plum', 'cherry')
        >>> determinar_categoria_ciruela('nectarina', 50.0)
        (nan, nan)

    Note:
        - Solo procesa especies que sean exactamente 'ciruela'
        - El umbral de 60g es un estándar de la industria
        - Si no hay peso disponible, asume Cherry plum por defecto
    """
    # Verificar que sea efectivamente una ciruela
    if str(especie).strip().lower() == "ciruela":
        # Si hay peso disponible, clasificar según umbral
        if pd.notna(peso_promedio):
            if peso_promedio > 60.0:
                return ("Candy plum", "candy")
            else:
                return ("Cherry plum", "cherry")
        else:
            # Sin peso, asumir Cherry por defecto (categoría más común)
            return ("Cherry plum (sin peso)", "cherry")

    # No es ciruela, retornar valores nulos
    return (np.nan, np.nan)

def first_non_null(s: pd.Series):
    """
    Obtiene el primer valor no nulo de una serie de pandas.

    Esta función es útil para extraer un valor representativo de un grupo
    cuando se espera que todos los valores sean iguales (ej: nombre de variedad
    en un grupo de frutos de la misma variedad).

    Args:
        s (pd.Series): Serie de pandas que puede contener valores nulos

    Returns:
        Primer valor no nulo encontrado, o np.nan si todos son nulos

    Example:
        >>> serie = pd.Series([np.nan, np.nan, 'Valor', 'Otro'])
        >>> first_non_null(serie)
        'Valor'

        >>> serie_vacia = pd.Series([np.nan, np.nan, np.nan])
        >>> first_non_null(serie_vacia)
        nan
    """
    # Encontrar el índice del primer valor válido
    idx = s.first_valid_index()

    # Retornar el valor si existe, sino np.nan
    return s.loc[idx] if idx is not None else np.nan

def aplicar_nuevas_reglas_clustering(df_processed, COL_BRIX, COL_ACIDEZ, ESPECIE_COLUMN, VAR_COLUMN, FRUTO_COLUMN):
    """
    Aplica las nuevas reglas de clustering siguiendo el script de validación
    """
    
    # 1. Determinar columnas de agrupación
    season_col = 'temporada' if 'temporada' in df_processed.columns else 'harvest_period'
    
    # Agregar portainjerto si no existe
    if COL_PORTAINJERTO not in df_processed.columns:
        df_processed[COL_PORTAINJERTO] = 'NA'
    
    # Columnas de agrupación (incluye portainjerto y temporada)
    grp_keys = [ESPECIE_COLUMN, VAR_COLUMN, season_col, COL_CAMPO, COL_PORTAINJERTO, COL_FECHA_EVALUACION]
    
    # Limpiar claves de agrupación que no existen
    grp_keys = [col for col in grp_keys if col in df_processed.columns]
    
    # 3. Crear agregación usando múltiples pasos para evitar problemas
    grouped = df_processed.groupby(grp_keys, dropna=False)
    
    # Crear DataFrame base con las claves de agrupación
    resumen = grouped.size().reset_index(name='n_registros')
    
    # Agregar métricas una por una
    if COL_BRIX in df_processed.columns:
        brix_mean = grouped[COL_BRIX].mean().reset_index()
        brix_mean = brix_mean.rename(columns={COL_BRIX: 'brix_promedio'})
        resumen = resumen.merge(brix_mean, on=grp_keys, how='left')
    
    # Acidez del primer fruto (usar first en lugar de función custom)
    if COL_ACIDEZ in df_processed.columns:
        acidez_first = grouped[COL_ACIDEZ].first().reset_index()
        acidez_first = acidez_first.rename(columns={COL_ACIDEZ: 'acidez_primer_fruto'})
        resumen = resumen.merge(acidez_first, on=grp_keys, how='left')
    
    # Promedio de mejillas
    if "avg_mejillas" in df_processed.columns:
        mejillas_mean = grouped["avg_mejillas"].mean().reset_index()
        mejillas_mean = mejillas_mean.rename(columns={"avg_mejillas": "mejillas_promedio"})
        resumen = resumen.merge(mejillas_mean, on=grp_keys, how='left')
    
    # Firmeza por puntos
    firmeza_mapping = {
        "Quilla": "quilla_promedio",
        "Hombro": "hombro_promedio", 
        "Punta": "punta_promedio"
    }
    
    for punto, nuevo_nombre in firmeza_mapping.items():
        if punto in df_processed.columns:
            firmeza_mean = grouped[punto].mean().reset_index()
            firmeza_mean = firmeza_mean.rename(columns={punto: nuevo_nombre})
            resumen = resumen.merge(firmeza_mean, on=grp_keys, how='left')
    
    # Peso promedio
    for col in ["Peso (g)", "Peso", "Calibre"]:
        if col in df_processed.columns:
            peso_mean = grouped[col].mean().reset_index()
            peso_mean = peso_mean.rename(columns={col: "peso_promedio"})
            resumen = resumen.merge(peso_mean, on=grp_keys, how='left')
            break
    
    # 5. Crear columnas faltantes con valores por defecto
    columns_default = {
        "brix_promedio": np.nan,
        "acidez_primer_fruto": np.nan, 
        "mejillas_promedio": np.nan,
        "quilla_promedio": np.nan,
        "hombro_promedio": np.nan,
        "punta_promedio": np.nan,
        "peso_promedio": np.nan
    }
    
    for col, default_val in columns_default.items():
        if col not in resumen.columns:
            resumen[col] = default_val
    
    # 6. Calcular punto más débil
    puntos_cols = ["quilla_promedio", "hombro_promedio", "punta_promedio"]
    resumen["firmeza_punto_debil"] = resumen[puntos_cols].min(axis=1, skipna=True)
    
    # 7. Determinar categoría de ciruela basada en peso
    def aplicar_categoria_ciruela(row):
        if str(row[ESPECIE_COLUMN]).strip().lower() == "ciruela":
            peso = row.get("peso_promedio", np.nan)
            if pd.notna(peso):
                return ("Candy plum", "candy") if peso > 60.0 else ("Cherry plum", "cherry")
            else:
                return ("Cherry plum (sin peso)", "cherry")
        return (np.nan, np.nan)
    
    categorias = resumen.apply(aplicar_categoria_ciruela, axis=1, result_type="expand")
    categorias.columns = ["categoria_ciruela", "subtipo_ciruela"]
    resumen = pd.concat([resumen, categorias], axis=1)
    
    # 8. Aplicar bandas según las reglas específicas
    def aplicar_bandas(resumen_df):
        bandas_brix, bandas_mejillas, bandas_punto, bandas_acidez = [], [], [], []
        
        for _, row in resumen_df.iterrows():
            if str(row[ESPECIE_COLUMN]).strip().lower() == "ciruela":
                # Usar reglas de ciruela
                subtipo = row.get("subtipo_ciruela", "cherry")
                rules = PLUM_RULES.get(subtipo, PLUM_RULES["cherry"])
            elif str(row[ESPECIE_COLUMN]).strip().lower() in ["nectarina", "nectarin"]:
                # Detectar color y período para nectarina
                fecha = row.get(COL_FECHA_EVALUACION, row.get(season_col))
                
                # Intentar determinar color desde los datos
                color = "amarilla"  # Default
                if 'Color de pulpa' in df_processed.columns:
                    # Buscar datos del grupo específico
                    filtro = (df_processed[ESPECIE_COLUMN] == row[ESPECIE_COLUMN]) & \
                            (df_processed[VAR_COLUMN] == row[VAR_COLUMN])
                    if season_col in df_processed.columns:
                        filtro = filtro & (df_processed[season_col] == row[season_col])
                    
                    grupo_datos = df_processed[filtro]
                    if not grupo_datos.empty:
                        color_vals = grupo_datos['Color de pulpa'].dropna().astype(str)
                        if not color_vals.empty:
                            color_str = color_vals.iloc[0].lower()
                            color = "blanca" if "blanc" in color_str else "amarilla"
                
                periodo = determinar_harvest_period(color, fecha)
                rules = NECT_RULES.get(color, {}).get(periodo, {})
            else:
                rules = {}
            
            # Aplicar clasificación para cada métrica (con validación de columnas)
            bandas_brix.append(clasificar_valor(row.get("brix_promedio", np.nan), rules.get("BRIX", [])))
            bandas_mejillas.append(clasificar_valor(row.get("mejillas_promedio", np.nan), rules.get("FIRMEZA_MEJ", [])))
            bandas_punto.append(clasificar_valor(row.get("firmeza_punto_debil", np.nan), rules.get("FIRMEZA_PUNTO", [])))
            bandas_acidez.append(clasificar_valor(row.get("acidez_primer_fruto", np.nan), rules.get("ACIDEZ", [])))
        
        resumen_df["banda_brix"] = bandas_brix
        resumen_df["banda_mejillas"] = bandas_mejillas
        resumen_df["banda_firmeza_punto"] = bandas_punto
        resumen_df["banda_acidez"] = bandas_acidez
        
        # Calcular suma de bandas
        resumen_df["suma_bandas"] = (
            resumen_df[["banda_brix", "banda_mejillas", "banda_firmeza_punto", "banda_acidez"]]
            .sum(axis=1, min_count=1)
        )
        
        # Asignar clusters basados en suma de bandas
        def asignar_cluster(suma):
            if pd.isna(suma):
                return np.nan
            elif 3 <= suma <= 5:
                return 1
            elif 6 <= suma <= 8:
                return 2
            elif 9 <= suma <= 11:
                return 3
            else:  # 12+
                return 4
        
        resumen_df["cluster_grp"] = resumen_df["suma_bandas"].apply(asignar_cluster)

        # Build group identifiers for downstream analysis
        key_cols = [col for col in [ESPECIE_COLUMN, VAR_COLUMN, season_col, COL_CAMPO, COL_PORTAINJERTO] if col in resumen_df.columns]
        if key_cols:
            def _normalize_key_value(val):
                if pd.isna(val):
                    return 'NA'
                return str(val).strip()
            resumen_df['grupo_key'] = resumen_df[key_cols].apply(lambda row: '|'.join(_normalize_key_value(v) for v in row), axis=1)
            grupo_id = pd.factorize(resumen_df['grupo_key'])[0] + 1
            grupo_id = np.where(grupo_id == 0, np.nan, grupo_id)
            resumen_df['grupo_id'] = grupo_id
        else:
            resumen_df['grupo_key'] = np.nan
            resumen_df['grupo_id'] = np.nan

        fecha_col = COL_FECHA_EVALUACION if COL_FECHA_EVALUACION in resumen_df.columns else None
        if fecha_col:
            base_vals = resumen_df['grupo_key'].fillna('NA').astype(str)
            fecha_vals = resumen_df[fecha_col].fillna('NA').astype(str)
            resumen_df['grupo_key_detalle'] = base_vals + '|' + fecha_vals
            resumen_df.loc[resumen_df['grupo_key'].isna(), 'grupo_key_detalle'] = np.nan
        else:
            resumen_df['grupo_key_detalle'] = resumen_df['grupo_key']

        return resumen_df
    
    # 9. Aplicar el cálculo de bandas
    resumen_con_bandas = aplicar_bandas(resumen.copy())
    
    return resumen_con_bandas