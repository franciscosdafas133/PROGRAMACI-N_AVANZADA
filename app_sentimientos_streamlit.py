import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(layout="wide", page_title="Análisis de Sentimientos sobre Vacunas COVID-19")

st.title("🧬 Análisis Geolocalizado de Sentimientos sobre Vacunas COVID-19 (2020–2022)")

# --- Cargar datos ---
@st.cache_data
def cargar_datos():
    df = pd.read_csv("tweets_con_sentimiento_y_score_nuevo.csv")
    df = df.rename(columns={
        "sentimiento": "Sentimiento",
        "sentimiento_score": "Score",
        "pais": "País",
        "texto_nuevo": "Texto"
    })
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df = df.dropna(subset=["Score", "Sentimiento", "País"])
    return df

df = cargar_datos()

# --- Filtro: países con más de 1000 registros ---
conteo_ubicaciones = df["País"].value_counts()
ubicaciones_frecuentes = conteo_ubicaciones[conteo_ubicaciones > 1000].index.to_numpy()
df_top_paises = df[df["País"].isin(ubicaciones_frecuentes)]

# --- Filtro: eliminación de outliers por país ---
data_sin_outliers = pd.DataFrame()
for pais, grupo in df_top_paises.groupby("País"):
    Q1 = grupo["Score"].quantile(0.25)
    Q3 = grupo["Score"].quantile(0.75)
    IQR = Q3 - Q1
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR
    filtro = (grupo["Score"] >= limite_inf) & (grupo["Score"] <= limite_sup)
    data_filtrada = grupo.loc[filtro]
    data_sin_outliers = pd.concat([data_filtrada, data_sin_outliers], axis=0)

# --- Filtros interactivos ---
st.sidebar.header("🔍 Filtros")
paises = sorted(data_sin_outliers["País"].unique())
pais_seleccionado = st.sidebar.selectbox("Selecciona un país", ["Todos"] + paises)

# Filtrar data
df_filtrado = data_sin_outliers.copy()
if pais_seleccionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado["País"] == pais_seleccionado]

# --- Gráfico de barras de sentimiento por país ---
st.subheader("📊 Distribución de Sentimientos")
fig = px.histogram(df_filtrado, x="Sentimiento", color="Sentimiento",
                   category_orders={"Sentimiento": ["Negativo", "Neutral", "Positivo"]},
                   title=f"Distribución de Sentimientos en {'todos los países' if pais_seleccionado == 'Todos' else pais_seleccionado}")
st.plotly_chart(fig, use_container_width=True)
