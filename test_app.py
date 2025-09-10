import pandas as pd
import numpy as np
from segmentacion_base import process_carozos, DEFAULT_PLUM_RULES, DEFAULT_NECT_RULES
import streamlit as st

print("=== INICIANDO PRUEBAS DE LA APLICACIÓN ===\n")

# Simular session state
class MockSessionState:
    def __init__(self):
        self.data = {}
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __contains__(self, key):
        return key in self.data

# Crear mock session state
mock_state = MockSessionState()
mock_state["default_period"] = "tardia"
mock_state["default_color"] = "amarilla"
mock_state["rules_plum"] = DEFAULT_PLUM_RULES
mock_state["rules_nect"] = DEFAULT_NECT_RULES

# Reemplazar session state
st.session_state = mock_state

try:
    # Prueba 1: Cargar el archivo Excel
    print("1. Probando carga del archivo Excel...")
    file_path = "MAESTRO CAROZOS FINAL COMPLETO CG.xlsx"
    df_raw = pd.read_excel(file_path, sheet_name="CAROZOS", usecols="A:AP")
    print(f"   ✓ Archivo cargado: {len(df_raw)} filas")
    
    # Prueba 2: Procesar datos
    print("\n2. Probando procesamiento de datos...")
    result = process_carozos(
        df_raw,
        rules_plum=DEFAULT_PLUM_RULES,
        rules_nect=DEFAULT_NECT_RULES
    )
    
    if result and "df_processed" in result:
        df_processed = result["df_processed"]
        print(f"   ✓ Datos procesados: {len(df_processed)} filas")
        
        # Verificar columnas críticas
        required_cols = ["harvest_period", "Especie", "Variedad", "Fruto"]
        for col in required_cols:
            if col in df_processed.columns:
                print(f"   ✓ Columna '{col}' presente")
                # Verificar formato de harvest_period
                if col == "harvest_period":
                    sample_values = df_processed[col].dropna().head(5).tolist()
                    print(f"     Valores de muestra: {sample_values}")
                    # Verificar que empiecen con "Period"
                    for val in sample_values:
                        if isinstance(val, str) and not val.startswith("Period"):
                            print(f"     ⚠ Valor sin 'Period': {val}")
            else:
                print(f"   ✗ Columna '{col}' faltante")
    else:
        print("   ✗ Error en el procesamiento")
        
    # Prueba 3: Verificar datos agregados
    print("\n3. Verificando datos agregados...")
    if "agg_groups_plum" in result:
        agg_plum = result["agg_groups_plum"]
        print(f"   ✓ Datos agregados ciruela: {len(agg_plum)} grupos")
    if "agg_groups_nect" in result:
        agg_nect = result["agg_groups_nect"]
        print(f"   ✓ Datos agregados nectarina: {len(agg_nect)} grupos")
        
    # Prueba 4: Verificar PCA
    print("\n4. Probando análisis PCA...")
    from sklearn.decomposition import PCA
    
    numeric_cols = ["Promedio de firmeza mejillas", "BRIX", "Acidez (%)"]
    available_cols = [col for col in numeric_cols if col in df_processed.columns]
    
    if len(available_cols) >= 2:
        X = df_processed[available_cols].fillna(0)
        pca = PCA(n_components=2)
        pc = pca.fit_transform(X)
        print(f"   ✓ PCA calculado: shape {pc.shape}")
    else:
        print(f"   ✗ Columnas insuficientes para PCA: {available_cols}")
        
    # Prueba 5: Verificar fechas
    print("\n5. Verificando manejo de fechas...")
    if "Fecha evaluación" in df_processed.columns:
        fechas = df_processed["Fecha evaluación"].dropna().head(5)
        print(f"   Tipos de fecha encontrados: {fechas.apply(type).value_counts().to_dict()}")
        for idx, fecha in fechas.items():
            print(f"   - {fecha} (tipo: {type(fecha).__name__})")
    
except FileNotFoundError:
    print("✗ No se encontró el archivo Excel. Asegúrate de que existe.")
except Exception as e:
    print(f"✗ Error durante las pruebas: {e}")
    import traceback
    traceback.print_exc()

print("\n=== PRUEBAS COMPLETADAS ===")