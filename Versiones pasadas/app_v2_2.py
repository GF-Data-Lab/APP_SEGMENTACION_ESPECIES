# -*- coding: utf-8 -*-
"""
Streamlit multipage: Procesador de Carozos + Análisis de Clúster
Versión 1 – 11-jul-2025
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Sequence
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import io

# --------------------------------------------------
# Tu pipeline principal
# --------------------------------------------------

from pages import process_carozos, process_file  # ajusta el import a tu estructura


# --------------------------------------------------
# Navegación entre páginas
# --------------------------------------------------
page = st.sidebar.radio("📂 Selecciona página:",
                        ["Procesador de Carozos", "Segmentación por Clúster"])

if page == "Procesador de Carozos":
    st.title("🛠️ Procesador de Carozos CG")
    st.write("Sube tu archivo Excel 'MAESTRO CAROZOS...' y obtén clusters y clasificaciones.")
    uploaded = st.file_uploader("🔄 Sube tu archivo Excel", type=["xls", "xlsx"])
    if uploaded:
        df = process_file(uploaded)
        if df is not None:
            st.success("¡Procesamiento completado! 🎉")
            st.dataframe(df, use_container_width=True)
            # Botón de descarga
            towrite = io.BytesIO()
            with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Carozos')
            towrite.seek(0)
            st.download_button(
                "📥 Descargar resultados",
                data=towrite.getvalue(),
                file_name="carozos_procesados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("Esperando que subas un archivo para procesar…")

# --------------------------------------------------
# Página de Clusterización
# --------------------------------------------------
else:
    st.title("📊 Segmentación de Carozos por Clúster")
    st.write("Aprovecha el DataFrame ya procesado para generar segmentos con K-Means y PCA.")
    
    # Primero, cargamos o reusamos el DataFrame procesado
    df = None
    if 'df_carozos' not in st.session_state:
        uploaded = st.file_uploader("🔄 Sube el Excel ya procesado (o súbelo arriba en 'Procesador')", type=["xls", "xlsx"])
        if uploaded:
            df = pd.read_excel(uploaded)
            st.session_state.df_carozos = df
    else:
        df = st.session_state.df_carozos
    
    if df is None:
        st.warning("Necesitamos el DataFrame procesado para clúster. Sube el archivo o ve a la página de Procesador.")
        st.stop()
    
    # Selección de variables numéricas
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    st.subheader("🧮 Variables numéricas para clúster")
    seleccion = st.multiselect("Elige al menos 2 columnas:", numeric_cols, default=numeric_cols[:4])
    if len(seleccion) < 2:
        st.error("Selecciona al menos dos variables numéricas.")
        st.stop()
    
    # Ponderaciones
    st.subheader("⚖️ Ponderaciones (%)")
    default_w = round(100.0 / len(seleccion), 1)
    pesos = {}
    cols_w = st.columns(len(seleccion))
    for i,col in enumerate(seleccion):
        pesos[col] = cols_w[i].number_input(col, 0.0, 100.0, default_w, step=1.0, key=f"w_{col}")
    total = sum(pesos.values())
    st.write(f"Total asignado: {total:.1f}%. Falta {100-total:.1f}%")
    if st.button("🔄 Normalizar a 100%"):
        for col in seleccion:
            pesos[col] = round(pesos[col] * 100/ total, 1) if total else default_w
            st.session_state[f"w_{col}"] = pesos[col]
        st.experimental_rerun()
    weights = np.array([pesos[c] for c in seleccion])
    weights = weights / weights.sum()
    
    # Parámetros de K-Means y PCA
    st.sidebar.header("⚙️ Parámetros de Modelo")
    n_clusters   = st.sidebar.slider("Número de clusters (K)", 2, 12, 4)
    rand_state   = st.sidebar.number_input("Random state", 0, 9999, 42, 1)
    do_pca       = st.sidebar.checkbox("Calcular PCA 3D", True)
    show_labels  = st.sidebar.checkbox("Mostrar etiquetas PCA", False)
    
    # Estandarización y clustering
    X = df[seleccion].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xw = Xs * weights
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=rand_state)
    clusters = kmeans.fit_predict(Xw)
    df['Cluster'] = clusters.astype(int)
    df['Cluster_str'] = df['Cluster'].astype(str)
    
    # PCA
    if do_pca:
        pca = PCA(n_components=3, random_state=rand_state)
        df[['PCA1','PCA2','PCA3']] = pca.fit_transform(Xw)
    
    # Visualizaciones
    st.subheader("📊 Conteo por cluster")
    fig_bar = px.bar(df['Cluster'].value_counts().sort_index(),
                     labels={'value':'# Registros','index':'Cluster'},
                     text_auto=True)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    if do_pca:
        st.subheader("🔍 PCA 3D")
        fig3d = px.scatter_3d(df, x='PCA1', y='PCA2', z='PCA3',
                              color='Cluster_str',
                              hover_data=seleccion,
                              text=df.index if show_labels else None)
        fig3d.update_traces(marker=dict(size=4))
        st.plotly_chart(fig3d, use_container_width=True)
    
    st.subheader("📈 Estadísticos por cluster")
    stats = df.groupby('Cluster')[seleccion].agg(['mean','median']).round(2)
    st.dataframe(stats, use_container_width=True)
    
    # Botón de descarga
    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Clusters')
        stats.to_excel(writer, sheet_name='Estadísticos')
    towrite.seek(0)
    st.download_button("📥 Descargar resultados de cluster",
                       data=towrite.getvalue(),
                       file_name="carozos_clusters.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    st.success("✅ Segmentación completa. ¡A explorar esos clusters!")
