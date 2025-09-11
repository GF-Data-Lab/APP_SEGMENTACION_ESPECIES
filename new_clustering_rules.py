# Nuevas reglas de clustering basadas en el script de validación
import numpy as np
import pandas as pd

# Reglas de bandas para Ciruela
PLUM_RULES = {
    "candy": {
        "BRIX": [(18.0, float('inf'), 1), (16.0, 18.0, 2), (14.0, 16.0, 3), (float('-inf'), 14.0, 4)],
        "FIRMEZA_PUNTO": [(7.0, float('inf'), 1), (5.0, 7.0, 2), (4.0, 5.0, 3), (float('-inf'), 4.0, 4)],
        "FIRMEZA_MEJ": [(9.0, float('inf'), 1), (7.0, 9.0, 2), (6.0, 7.0, 3), (float('-inf'), 6.0, 4)],
        "ACIDEZ": [(float('-inf'), 0.60, 1), (0.60, 0.81, 2), (0.81, 1.00, 3), (1.00, float('inf'), 4)],
    },
    "sugar": {
        "BRIX": [(21.0, float('inf'), 1), (18.0, 21.0, 2), (15.0, 18.0, 3), (float('-inf'), 15.0, 4)],
        "FIRMEZA_PUNTO": [(6.0, float('inf'), 1), (4.5, 6.0, 2), (3.0, 4.5, 3), (float('-inf'), 3.0, 4)],
        "FIRMEZA_MEJ": [(6.0, float('inf'), 1), (5.0, 6.0, 2), (4.0, 5.0, 3), (float('-inf'), 4.0, 4)],
        "ACIDEZ": [(float('-inf'), 0.60, 1), (0.60, 0.81, 2), (0.81, 1.00, 3), (1.00, float('inf'), 4)],
    },
}

def mk_nect(b1, b2, b3, m1, m2):
    return {
        "BRIX": [(b1, float("inf"), 1), (b2, b1, 2), (b3, b2, 3), (float("-inf"), b3, 4)],
        "FIRMEZA_PUNTO": [(9.0, float("inf"), 1), (8.0, 9.0, 2), (7.0, 8.0, 3), (float("-inf"), 7.0, 4)],
        "FIRMEZA_MEJ": [(m1, float("inf"), 1), (m2, m1, 2), (9.0, m2, 3), (float("-inf"), 9.0, 4)],
        "ACIDEZ": [(float("-inf"), 0.60, 1), (0.60, 0.81, 2), (0.81, 1.00, 3), (1.00, float("inf"), 4)],
    }

# Reglas de bandas para Nectarina
NECT_RULES = {
    "amarilla": {
        "muy_temprana": mk_nect(13.0, 10.0, 9.0, 14.0, 12.0),
        "temprana": mk_nect(13.0, 11.0, 9.0, 14.0, 12.0),
        "tardia": mk_nect(14.0, 12.0, 10.0, 14.0, 12.0),
    },
    "blanca": {
        "muy_temprana": mk_nect(13.0, 10.0, 9.0, 13.0, 11.0),
        "temprana": mk_nect(13.0, 11.0, 9.0, 13.0, 11.0),
        "tardia": mk_nect(14.0, 12.0, 10.0, 13.0, 11.0),
    },
}

def clasificar_valor(val, bands):
    """Clasifica un valor según las bandas definidas"""
    if pd.isna(val) or not bands:
        return np.nan
    for lo, hi, grp in bands:
        if lo <= val < hi:
            return grp
    return np.nan

def determinar_harvest_period(color, fecha):
    """Determina el período de cosecha según color y fecha"""
    fecha = pd.to_datetime(fecha, errors='coerce')
    if pd.isna(fecha):
        return "tardia"
    m, d = fecha.month, fecha.day
    if color == "blanca":
        if (m, d) < (11,25): return "muy_temprana"
        if (11,25) <= (m, d) <= (12,15): return "temprana"
        if (12,16) <= (m, d) <= (2,15): return "tardia"
    else:
        if (m, d) < (11,22): return "muy_temprana"
        if (11,22) <= (m, d) <= (12,22): return "temprana"
        if (12,23) <= (m, d) <= (2,15): return "tardia"
    return "tardia"

def determinar_categoria_ciruela(especie, peso_promedio):
    """Determina la categoría de ciruela basada en el peso"""
    if str(especie).strip().lower() == "ciruela":
        if pd.notna(peso_promedio):
            return ("Candy plum", "candy") if peso_promedio > 60.0 else ("Cherry plum", "sugar")
        else:
            return ("Cherry plum (sin peso)", "sugar")
    return (np.nan, np.nan)

def first_non_null(s):
    """Obtiene el primer valor no nulo de una serie"""
    idx = s.first_valid_index()
    return s.loc[idx] if idx is not None else np.nan

def aplicar_nuevas_reglas_clustering(df_processed, COL_BRIX, COL_ACIDEZ, ESPECIE_COLUMN, VAR_COLUMN, FRUTO_COLUMN):
    """
    Aplica las nuevas reglas de clustering siguiendo el script de validación
    """
    
    # 1. Determinar columnas de agrupación
    season_col = 'temporada' if 'temporada' in df_processed.columns else 'harvest_period'
    
    # Agregar portainjerto si no existe
    if 'portainjerto' not in df_processed.columns:
        df_processed['portainjerto'] = 'NA'
    
    # Columnas de agrupación (incluye portainjerto y temporada)
    grp_keys = [ESPECIE_COLUMN, VAR_COLUMN, season_col, 'Campo', 'portainjerto', 'Fecha de evaluación']
    
    # Limpiar claves de agrupación que no existen
    grp_keys = [col for col in grp_keys if col in df_processed.columns]
    
    # 2. Función para obtener primer valor no nulo
    def first_non_null(s):
        idx = s.first_valid_index()
        return s.loc[idx] if idx is not None else np.nan
    
    # 3. Crear agregación según nuevas reglas
    agg_dict = {
        COL_BRIX: (COL_BRIX, "mean"),  # BRIX promedio
        COL_ACIDEZ: (COL_ACIDEZ, first_non_null),  # Acidez del primer fruto
        "avg_mejillas": ("avg_mejillas", "mean"),  # Promedio de mejillas
        "n_registros": (COL_BRIX, "size"),  # Número de registros
    }
    
    # Agregar columnas de firmeza por punto si existen
    for punto in ["Firmeza quilla", "Firmeza hombro", "Firmeza punta"]:
        if punto in df_processed.columns:
            agg_dict[punto] = (punto, "mean")
    
    # Peso si existe
    for col in ["Peso (g)", "Peso", "Calibre"]:
        if col in df_processed.columns:
            agg_dict["peso_promedio"] = (col, "mean")
            break
    
    # 4. Crear resumen por grupo
    resumen = (
        df_processed.groupby(grp_keys, dropna=False)
        .agg(**agg_dict)
        .reset_index()
        .rename(columns={
            COL_BRIX: "brix_promedio",
            COL_ACIDEZ: "acidez_primer_fruto",
            "avg_mejillas": "mejillas_promedio",
            "Firmeza quilla": "quilla_promedio",
            "Firmeza hombro": "hombro_promedio", 
            "Firmeza punta": "punta_promedio"
        })
    )
    
    # 5. Crear columnas de firmeza por punto si no existen
    for col in ["quilla_promedio", "hombro_promedio", "punta_promedio"]:
        if col not in resumen.columns:
            resumen[col] = np.nan
    
    # 6. Calcular punto más débil
    puntos_cols = ["quilla_promedio", "hombro_promedio", "punta_promedio"]
    resumen["firmeza_punto_debil"] = resumen[puntos_cols].min(axis=1, skipna=True)
    
    # 7. Determinar categoría de ciruela basada en peso
    def aplicar_categoria_ciruela(row):
        if str(row[ESPECIE_COLUMN]).strip().lower() == "ciruela":
            peso = row.get("peso_promedio", np.nan)
            if pd.notna(peso):
                return ("Candy plum", "candy") if peso > 60.0 else ("Cherry plum", "sugar")
            else:
                return ("Cherry plum (sin peso)", "sugar")
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
                subtipo = row.get("subtipo_ciruela", "sugar")
                rules = PLUM_RULES.get(subtipo, PLUM_RULES["sugar"])
            elif str(row[ESPECIE_COLUMN]).strip().lower() in ["nectarina", "nectarin"]:
                # Detectar color y período para nectarina
                fecha = row.get('Fecha de evaluación', row.get(season_col))
                
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
            
            # Aplicar clasificación para cada métrica
            bandas_brix.append(clasificar_valor(row["brix_promedio"], rules.get("BRIX", [])))
            bandas_mejillas.append(clasificar_valor(row["mejillas_promedio"], rules.get("FIRMEZA_MEJ", [])))
            bandas_punto.append(clasificar_valor(row["firmeza_punto_debil"], rules.get("FIRMEZA_PUNTO", [])))
            bandas_acidez.append(clasificar_valor(row["acidez_primer_fruto"], rules.get("ACIDEZ", [])))
        
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
        
        return resumen_df
    
    # 9. Aplicar el cálculo de bandas
    resumen_con_bandas = aplicar_bandas(resumen.copy())
    
    return resumen_con_bandas