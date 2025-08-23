import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.figure_factory as ff
from utils import show_logo

st.set_page_config(page_title="Modelo y Clustering", page_icon="🧠", layout="wide")

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

generar_menu()

st.title("🧪 Modelo de Clasificación y Clustering")
st.markdown(
    """
    Entrena un modelo de clasificación y explora agrupamientos K‑Means
    utilizando los datos que subiste en la sección de **Carga de archivos**.
    """
)

# 1. Obtener datos cargados
if "carozos_df" not in st.session_state:
    st.error("❗ Primero ve a 'Carga de archivos' y sube tu Excel.")
    st.stop()

df = st.session_state["carozos_df"].copy()
st.success("✅ Datos cargados desde archivo")
st.dataframe(df.head(), use_container_width=True)

# 2. Selección de variables
st.subheader("⚙️ Selección de etiqueta y características")
cat_cols = [c for c in df.columns if df[c].dtype == "object"]
if not cat_cols:
    st.error("No hay columnas categóricas para usar como etiqueta")
    st.stop()

default_target = "Variedad" if "Variedad" in cat_cols else cat_cols[0]
# target variable
target_col = st.selectbox(
    "Etiqueta (target) a predecir",
    options=cat_cols,
    index=cat_cols.index(default_target)
)
# grouping column
group_col = st.selectbox(
    "Columna de agrupación",
    options=cat_cols,
    index=cat_cols.index(default_target)
)

numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
features = st.multiselect(
    "Variables predictoras (features)",
    options=numeric_cols,
    default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols,
)
if len(features) < 2:
    st.error("Selecciona al menos **2** variables predictoras.")
    st.stop()

# 3. División de datos
st.subheader("📊 División Train / Test")
test_size = st.slider("Tamaño del test (%)", 10, 50, 25, step=5)

data = df[features + [target_col, group_col]].dropna()
X = data[features].values
y = data[target_col].astype(str).values
groups = data[group_col].astype(str).values
try:
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, groups,
        test_size=test_size / 100,
        random_state=42,
        stratify=y,
    )
except ValueError:
    st.warning("⚠️ Stratify falló. Se realiza división sin estratificar")
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, groups,
        test_size=test_size / 100,
        random_state=42,
    )

st.write(f"- 👨‍🏫 Entrenamiento: {len(y_train)} muestras")
st.write(f"- 🧪 Test: {len(y_test)} muestras")

# 4. Hiperparámetros
st.subheader("🔧 Hiperparámetros Random Forest")
n_estimators = st.slider("Número de árboles", 10, 200, 50, step=10)
max_depth = st.slider("Profundidad máxima (None=sin límite)", 1, 20, 5)
if max_depth == 1:
    max_depth = None

# 5. Entrenamiento y métricas
if st.button("🚀 Entrenar modelo"):
    with st.spinner("Entrenando Random Forest…"):
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    st.success("✅ Modelo entrenado")

    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy (test)", f"{acc*100:.2f}%")

    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).T, use_container_width=True)

    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=list(clf.classes_),
        y=list(clf.classes_),
        colorscale="Blues",
        showscale=True,
    )
    fig_cm.update_layout(xaxis_title="Predicho", yaxis_title="Verdadero")
    st.plotly_chart(fig_cm, use_container_width=True)

    # accuracy por grupo
    st.markdown("**Accuracy por grupo**")
    df_res = pd.DataFrame({target_col: y_test, "Predicho": y_pred, group_col: g_test})
    acc_group = df_res.groupby(group_col).apply(
        lambda d: accuracy_score(d[target_col], d["Predicho"])
    )
    fig_acc = px.bar(
        acc_group,
        x=acc_group.index,
        y=acc_group.values,
        labels={"x": group_col, "y": "Accuracy"}
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    # importancia de variables
    st.markdown("**Importancia de Variables**")
    imp = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    fig_imp = px.bar(
        imp,
        x=imp.values,
        y=imp.index,
        orientation="h",
        labels={"x": "Importancia", "y": "Variable"}
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # Clustering
    st.subheader("Agrupamiento KMeans")
    n_clusters = st.slider("Número de clusters", 2, 10, 3)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    km_labels = km.fit_predict(df[features].fillna(0))
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(df[features].fillna(0))
    plot_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    plot_df["cluster"] = km_labels
    plot_df[group_col] = df[group_col].values
    fig_cluster = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="cluster",
        symbol=group_col,
        title="KMeans vs grupo",
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    cross = pd.crosstab(km_labels, df[group_col])
    st.markdown("**Comparación cluster vs grupo**")
    st.dataframe(cross)

    majority = cross.idxmax(axis=1)
    df_cluster = df.copy()
    df_cluster["cluster"] = km_labels
    df_cluster["cluster_mayor"] = df_cluster["cluster"].map(majority)
    mismatch = df_cluster[df_cluster[group_col] != df_cluster["cluster_mayor"]]
    st.markdown("**Registros con cluster distinto al grupo mayoritario**")
    st.dataframe(mismatch[[group_col, "cluster"] + features])
else:
    st.info("Ajusta los parámetros y dale a **Entrenar modelo** 😎")
