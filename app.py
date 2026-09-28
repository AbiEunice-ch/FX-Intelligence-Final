import streamlit as st

import pandas as pd

import plotly.express as px

import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 

# =========================================================
# CONFIGURACIÓN
# =========================================================
st.set_page_config(
   page_title="FX Intelligence",
   page_icon="💱",
   layout="wide"
)

# =========================================================
# CARGA DE DATOS
# =========================================================
@st.cache_data
def cargar_datos():
    BASE_DIR = Path(__file__).resolve().parent
    df = pd.read_csv(BASE_DIR / "TC_FX_Intelligence.csv")
    df["tmb_FechaCarga"] = pd.to_datetime(df["tmb_FechaCarga"],errors="coerce")
    df = df.sort_values(
       ["tmb_MonedaOrig", "tmb_FechaCarga"]
    )
    return df

TC = cargar_datos()

# =========================================================
# TÍTULO
# =========================================================
st.title("💱 FX Intelligence")
st.markdown(
   """
   ### Plataforma de análisis inteligente del mercado cambiario
   Exploración histórica, volatilidad, regímenes de mercado
   y pronóstico de tipos de cambio.
   """
)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Configuración")
monedas = sorted(
   TC["tmb_MonedaOrig"].dropna().unique()
)
moneda = st.sidebar.selectbox(
   "Selecciona una moneda",
   monedas
)

# =========================================================
# FILTRO
# =========================================================
df_moneda = TC[
   TC["tmb_MonedaOrig"] == moneda
].copy()

# =========================================================
# INDICADORES
# =========================================================
ultimo = df_moneda.iloc[-1]
precio_actual = ultimo["tmb_PrecioLimpio"]
volatilidad = ultimo["Vola20"]
regimen = ultimo["Regimen_Volatilidad"]

col1, col2, col3 = st.columns(3)

with col1:
   st.metric(
       "Último tipo de cambio",
       f"{precio_actual:.4f}"
   )

with col2:
   if pd.notna(volatilidad):
       st.metric(
           "Volatilidad 20 días",
           f"{volatilidad:.2%}"
       )
   else:
       st.metric(
           "Volatilidad 20 días",
           "N/D"
       )

with col3:
   st.metric(
       "Régimen de volatilidad",
       str(regimen)
   )

# =========================================================
# EVOLUCIÓN HISTÓRICA
# =========================================================
st.subheader(
   f"📈 Evolución histórica — {moneda}/MXN"
)
fig_precio = px.line(
   df_moneda,
   x="tmb_FechaCarga",
   y="tmb_PrecioLimpio",
   title=f"Tipo de cambio histórico {moneda}/MXN",
   labels={
       "tmb_FechaCarga": "Fecha",
       "tmb_PrecioLimpio": "Tipo de cambio"
   }
)
fig_precio.update_layout(
   hovermode="x unified"
)
st.plotly_chart(
   fig_precio,
   use_container_width=True
)

# =========================================================

# FORECAST USD/MXN

# =========================================================

st.subheader("🔮 Pronóstico USD/MXN")

if moneda == "USD":

    from statsmodels.tsa.arima.model import ARIMA

    import numpy as np

    # Serie USD

    usd_forecast = (

        df_moneda[

            ["tmb_FechaCarga", "tmb_PrecioLimpio"]

        ]

        .dropna()

        .sort_values("tmb_FechaCarga")

        .drop_duplicates("tmb_FechaCarga")

        .set_index("tmb_FechaCarga")

    )

    y = usd_forecast["tmb_PrecioLimpio"]

    # Horizonte de pronóstico

    horizonte = st.slider(

        "Horizonte de pronóstico (días)",

        min_value=1,

        max_value=20,

        value=5

    )

    # Modelo ARIMA(0,1,0)

    modelo_app = ARIMA(

        y,

        order=(0, 1, 0)

    )

    resultado_app = modelo_app.fit()

    # Pronóstico

    forecast = resultado_app.forecast(

        steps=horizonte

    )

    # Fechas futuras aproximadas de días hábiles

    fechas_futuras = pd.bdate_range(

        start=y.index[-1] + pd.Timedelta(days=1),

        periods=horizonte

    )

    forecast_df = pd.DataFrame({

        "Fecha": fechas_futuras,

        "Pronóstico": forecast.to_numpy()

    })

    # Gráfica

    historico_reciente = usd_forecast.tail(120).reset_index()

    fig_forecast = px.line(

        historico_reciente,

        x="tmb_FechaCarga",

        y="tmb_PrecioLimpio",

        title="USD/MXN — Histórico y pronóstico",

        labels={

            "tmb_FechaCarga": "Fecha",

            "tmb_PrecioLimpio": "Tipo de cambio"

        }

    )

    fig_forecast.add_scatter(

        x=forecast_df["Fecha"],

        y=forecast_df["Pronóstico"],

        mode="lines+markers",

        name="Pronóstico"

    )

    fig_forecast.update_layout(

        hovermode="x unified"

    )

    st.plotly_chart(

        fig_forecast,

        use_container_width=True

    )

    # Métricas previamente obtenidas en el conjunto de prueba

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:

        st.metric(

            "Modelo",

            "ARIMA(0,1,0)"

        )

    with col_f2:

        st.metric(

            "MAE",

            "0.0911 MXN"

        )

    with col_f3:

        st.metric(

            "RMSE",

            "0.1307 MXN"

        )

    st.caption(

        "Las métricas corresponden a la evaluación fuera de muestra "

        "realizada previamente sobre el conjunto de prueba."

    )

    st.dataframe(

        forecast_df,

        use_container_width=True

    )

else:
    st.info(

        "El módulo de pronóstico está implementado inicialmente "

        "para USD/MXN. Posteriormente se extenderá a EUR y GBP."

    )
 

# =========================================================

# MACHINE LEARNING - RANDOM FOREST

# =========================================================

st.subheader("🤖 Machine Learning — Random Forest")

if moneda == "USD":

    # -----------------------------------------

    # Preparación de datos

    # -----------------------------------------

    ml_app = (

        TC[TC["tmb_MonedaOrig"] == "USD"]

        [

            [

                "tmb_FechaCarga",

                "tmb_PrecioLimpio",

                "Rendimiento",

                "Vola20",

                "Media_Movil_20",

                "Media_Movil_60",

                "Lag_1",

                "Lag_5",

                "Lag_20"

            ]

        ]

        .copy()

    )

    ml_app = (

        ml_app

        .sort_values("tmb_FechaCarga")

        .dropna()

        .reset_index(drop=True)

    )

    # Objetivo: precio del siguiente día

    ml_app["Target_1D"] = (

        ml_app["tmb_PrecioLimpio"].shift(-1)

    )

    ml_app = ml_app.dropna().reset_index(drop=True)

    features_ml = [

        "tmb_PrecioLimpio",

        "Rendimiento",

        "Vola20",

        "Media_Movil_20",

        "Media_Movil_60",

        "Lag_1",

        "Lag_5",

        "Lag_20"

    ]

    X_ml = ml_app[features_ml]

    y_ml = ml_app["Target_1D"]

    # -----------------------------------------

    # División temporal 80 / 20

    # -----------------------------------------

    split_ml = int(len(ml_app) * 0.80)

    X_train_ml = X_ml.iloc[:split_ml]

    X_test_ml = X_ml.iloc[split_ml:]

    y_train_ml = y_ml.iloc[:split_ml]

    y_test_ml = y_ml.iloc[split_ml:]

    # -----------------------------------------

    # Entrenamiento

    # -----------------------------------------

    rf_app = RandomForestRegressor(

        n_estimators=200,

        max_depth=10,

        min_samples_leaf=5,

        random_state=42,

        n_jobs=-1

    )

    rf_app.fit(

        X_train_ml,

        y_train_ml

    )

    # -----------------------------------------

    # Evaluación

    # -----------------------------------------

    pred_test_ml = rf_app.predict(X_test_ml)

    mae_app = mean_absolute_error(

        y_test_ml,

        pred_test_ml

    )

    rmse_app = np.sqrt(

        mean_squared_error(

            y_test_ml,

            pred_test_ml

        )

    )

    r2_app = r2_score(

        y_test_ml,

        pred_test_ml

    )

    # -----------------------------------------

    # Predicción del siguiente día

    # -----------------------------------------

    ultima_fila = ml_app.iloc[-1:]

    X_actual = ultima_fila[features_ml]

    prediccion_siguiente = rf_app.predict(

        X_actual

    )[0]

    precio_actual_ml = ultima_fila[

        "tmb_PrecioLimpio"

    ].iloc[0]

    # -----------------------------------------

    # Indicadores

    # -----------------------------------------

    st.markdown(

        """

        El modelo Random Forest utiliza variables históricas, volatilidad, medias móviles y rezagos para estimar el tipo de cambio USD/MXN del siguiente día hábil.

        """

    )

    col_ml1, col_ml2, col_ml3, col_ml4 = st.columns(4)

    with col_ml1:

        st.metric(

            "Precio actual",

            f"{precio_actual_ml:.4f}"

        )

    with col_ml2:

        st.metric(

            "Predicción +1 día",

            f"{prediccion_siguiente:.4f}"

        )

    with col_ml3:

        st.metric(

            "MAE",

            f"{mae_app:.4f}"

        )

    with col_ml4:

        st.metric(

            "RMSE",

            f"{rmse_app:.4f}"

        )

    st.metric(

        "R²",

        f"{r2_app:.4f}"

    )

    # -----------------------------------------

    # Comparación real vs predicho

    # -----------------------------------------

    resultados_ml = pd.DataFrame({

        "Fecha": ml_app[

            "tmb_FechaCarga"

        ].iloc[split_ml:],

        "Real": y_test_ml.to_numpy(),

        "Predicción": pred_test_ml

    })

    resultados_ml = resultados_ml.tail(120)

    fig_ml = px.line(

        resultados_ml,

        x="Fecha",

        y=["Real", "Predicción"],

        title="USD/MXN — Real vs Predicción Random Forest",

        labels={

            "value": "Tipo de cambio",

            "variable": "Serie"

        }

    )

    fig_ml.update_layout(

        hovermode="x unified"

    )

    st.plotly_chart(

        fig_ml,

        use_container_width=True

    )

    # -----------------------------------------

    # Importancia de variables

    # -----------------------------------------

    importancia_app = pd.DataFrame({

        "Variable": features_ml,

        "Importancia": rf_app.feature_importances_

    }).sort_values(

        "Importancia",

        ascending=False

    )

    st.subheader(

        "📊 Importancia de variables"

    )

    fig_importancia = px.bar(

        importancia_app,

        x="Importancia",

        y="Variable",

        orientation="h",

        title="Importancia de variables — Random Forest"

    )

    st.plotly_chart(

        fig_importancia,

        use_container_width=True

    )

    st.dataframe(

        importancia_app,

        use_container_width=True

    )

else:
    st.info(

        "El módulo de Machine Learning está implementado "

        "inicialmente para USD/MXN."

    )
 


# =========================================================
# VOLATILIDAD
# =========================================================
st.subheader(
   f"⚠️ Volatilidad — {moneda}"
)
fig_vol = px.line(
   df_moneda,
   x="tmb_FechaCarga",
   y="Vola20",
   title="Volatilidad móvil de 20 días",
   labels={
       "tmb_FechaCarga": "Fecha",
       "Vola20": "Volatilidad"
   }
)
fig_vol.update_layout(
   hovermode="x unified"
)
st.plotly_chart(
   fig_vol,
   use_container_width=True
)

# =========================================================
# RÉGIMEN DE VOLATILIDAD
# =========================================================
st.subheader(
   "🔎 Régimen de volatilidad"
)
regimen_counts = (
   df_moneda["Regimen_Volatilidad"]
   .value_counts()
   .reset_index()
)
regimen_counts.columns = [
   "Regimen",
   "Observaciones"
]
fig_regimen = px.bar(
   regimen_counts,
   x="Regimen",
   y="Observaciones",
   title="Distribución de regímenes de volatilidad",
   labels={
       "Regimen": "Régimen",
       "Observaciones": "Número de observaciones"
   }
)
st.plotly_chart(
   fig_regimen,
   use_container_width=True
)

# =========================================================
# INFORMACIÓN DEL DATASET
# =========================================================
st.subheader("📊 Información del análisis")
c1, c2, c3 = st.columns(3)
with c1:
   st.metric(
       "Observaciones",
       f"{len(df_moneda):,}"
   )
with c2:
   fecha_inicio = df_moneda["tmb_FechaCarga"].min()
   st.metric(
       "Fecha inicial",
       fecha_inicio.strftime("%d/%m/%Y")
   )
with c3:
   fecha_fin = df_moneda["tmb_FechaCarga"].max()
   st.metric(
       "Fecha final",
       fecha_fin.strftime("%d/%m/%Y")
   )

# =========================================================
# TABLA RECIENTE
# =========================================================
st.subheader("📋 Últimas observaciones")
columnas_mostrar = [
   "tmb_FechaCarga",
   "tmb_PrecioLimpio",
   "Rendimiento",
   "Vola20",
   "Regimen_Volatilidad"
]
columnas_disponibles = [
   c for c in columnas_mostrar
   if c in df_moneda.columns
]
st.dataframe(
   df_moneda[
       columnas_disponibles
   ].tail(10).sort_values(
       "tmb_FechaCarga",
       ascending=False
   ),
   use_container_width=True
)

# =========================================================
# PIE
# =========================================================
st.markdown("---")
st.caption(
   "FX Intelligence | Análisis histórico de tipos de cambio MXP/USD/EUR/GBP"
)

# =========================================================

# RAG + AGENTE FX INTELLIGENCE

# =========================================================

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity


# =========================================================

# CORPUS DEL PROYECTO

# =========================================================

documentos_fx = [

    {

        "titulo": "Datos y EDA",

        "texto": """

        FX Intelligence analiza tipos de cambio históricos entre el peso mexicano

        y las monedas USD, EUR y GBP. El dataset contiene precios, rendimientos,

        volatilidad de 20 días, medias móviles y variables lag.

        """

    },

    {

        "titulo": "Estadística",

        "texto": """

        Se calcularon estadísticas descriptivas, intervalos de confianza,

        correlaciones y pruebas de hipótesis. La prueba de Friedman obtuvo

        estadístico 1.3037 y p-value 0.5211.

        """

    },

    {

        "titulo": "Series temporales",

        "texto": """

        El proyecto utiliza medias móviles de 20 y 60 días, volatilidad móvil

        y variables Lag_1, Lag_5 y Lag_20. También se analizaron autocorrelación,

        estacionalidad descriptiva y regímenes de volatilidad.

        """

    },

    {

        "titulo": "Forecasting USD/MXN",

        "texto": """

        Para USD/MXN se utilizó una separación temporal 80/20.

        El modelo naive obtuvo MAE 0.0911 y RMSE 0.1307.

        ARIMA(0,1,0) obtuvo MAE 0.0911 y RMSE 0.1307.

        """

    },

    {

        "titulo": "Machine Learning",

        "texto": """

        Random Forest utiliza precio actual, rendimiento, volatilidad,

        medias móviles y variables lag.

        Obtuvo MAE 0.0258, RMSE 0.0504 y R2 0.9980.

        """

    },

    {

        "titulo": "Deep Learning",

        "texto": """

        Se implementó un MLPRegressor con capas ocultas 32 y 16.

        Obtuvo MAE 0.0501, RMSE 0.0724 y R2 0.9958.

        """

    },

    {

        "titulo": "PCA y Clustering",

        "texto": """

        PCA produjo 28.74% de varianza explicada en PC1 y 17.16%

        en PC2, con aproximadamente 45.90% acumulado.

        KMeans generó tres grupos.

        """

    },

    {

        "titulo": "Fourier",

        "texto": """

        El análisis de Fourier mediante FFT permite identificar

        frecuencias con mayor potencia en los rendimientos.

        """

    },

    {

        "titulo": "Wavelets",

        "texto": """

        Se utilizó una descomposición Wavelet Daubechies db4 de nivel 4

        para estudiar la distribución de energía entre diferentes escalas.

        """

    }

]

df_docs_app = pd.DataFrame(documentos_fx)


# =========================================================

# RETRIEVER TF-IDF

# =========================================================

vectorizador_app = TfidfVectorizer(

    ngram_range=(1, 2)

)

matriz_docs_app = vectorizador_app.fit_transform(

    df_docs_app["texto"]

)


def recuperar_documentos(pregunta, k=3):

    vector_pregunta = vectorizador_app.transform(

        [pregunta]

    )

    similitudes = cosine_similarity(

        vector_pregunta,

        matriz_docs_app

    )[0]

    resultados = df_docs_app.copy()

    resultados["similitud"] = similitudes

    resultados = resultados.sort_values(

        "similitud",

        ascending=False

    ).head(k)

    return resultados


# =========================================================

# TOOLS DEL AGENTE

# =========================================================

def tool_metricas_rf_app():

    return {

        "modelo": "Random Forest",

        "MAE": 0.0258,

        "RMSE": 0.0504,

        "R2": 0.9980

    }


def tool_forecast_arima_app():

    return {

        "modelo": "ARIMA(0,1,0)",

        "MAE": 0.0911,

        "RMSE": 0.1307

    }


def tool_pca_app():

    return {

        "PC1": 0.2874,

        "PC2": 0.1716,

        "Varianza_acumulada": 0.4590

    }


def tool_volatilidad_app():

    return {

        "variable": "Vola20",

        "descripcion": "Volatilidad móvil de 20 días"

    }


# =========================================================

# AGENTE / ROUTER

# =========================================================

def ejecutar_agente(pregunta):

    pregunta_lower = pregunta.lower()

    if "random forest" in pregunta_lower or "mae" in pregunta_lower:

        return (

            "tool_metricas_rf",

            tool_metricas_rf_app()

        )

    elif "arima" in pregunta_lower or "forecast" in pregunta_lower:

        return (

            "tool_forecast_arima",

            tool_forecast_arima_app()

        )

    elif "pca" in pregunta_lower or "componente" in pregunta_lower:

        return (

            "tool_pca",

            tool_pca_app()

        )

    elif "volatilidad" in pregunta_lower:

        return (

            "tool_volatilidad",

            tool_volatilidad_app()

        )

    return (

        "ninguna",

        None

    )


# =========================================================

# INTERFAZ RAG

# =========================================================

st.divider()

st.header("🧠 RAG + Agente FX Intelligence")

st.write(

    "Consulta los resultados documentados del proyecto mediante "

    "búsqueda semántica y herramientas analíticas."

)

pregunta_fx = st.text_input(

    "Escribe tu pregunta sobre FX Intelligence:",

    placeholder="Ej. ¿Qué métricas obtuvo Random Forest para USD/MXN?"

)


if pregunta_fx:

    # -----------------------------------------------------

    # RETRIEVAL

    # -----------------------------------------------------

    resultados_rag = recuperar_documentos(

        pregunta_fx,

        k=3

    )

    st.markdown("### 🔎 Fuentes recuperadas")

    st.dataframe(

        resultados_rag[

            ["titulo", "similitud"]

        ],

        use_container_width=True

    )

    # -----------------------------------------------------

    # CONTEXTO

    # -----------------------------------------------------

    contexto_rag = "\n\n".join(

        [

            f"FUENTE: {fila['titulo']}\n{fila['texto']}"

            for _, fila in resultados_rag.iterrows()

        ]

    )

    with st.expander("📚 Ver contexto recuperado"):

        st.write(contexto_rag)

    # -----------------------------------------------------

    # AGENTE

    # -----------------------------------------------------

    herramienta, resultado_tool = ejecutar_agente(

        pregunta_fx

    )

    if resultado_tool is not None:

        st.markdown("### 🛠️ Tool utilizada")

        st.code(herramienta)

        st.markdown("### 📊 Resultado de la herramienta")

        st.json(resultado_tool)

        st.markdown("### 💬 Respuesta")

        if herramienta == "tool_metricas_rf":

            st.write(

                f"Random Forest obtuvo MAE de "

                f"{resultado_tool['MAE']:.4f}, RMSE de "

                f"{resultado_tool['RMSE']:.4f} y R² de "

                f"{resultado_tool['R2']:.4f}."

            )

        elif herramienta == "tool_forecast_arima":

            st.write(

                f"ARIMA(0,1,0) obtuvo MAE de "

                f"{resultado_tool['MAE']:.4f} y RMSE de "

                f"{resultado_tool['RMSE']:.4f}."

            )

        elif herramienta == "tool_pca":

            st.write(

                f"PC1 explica {resultado_tool['PC1']:.2%} "

                f"y PC2 explica {resultado_tool['PC2']:.2%}. "

                f"La varianza acumulada es aproximadamente "

                f"{resultado_tool['Varianza_acumulada']:.2%}."

            )

        elif herramienta == "tool_volatilidad":

            st.write(

                "El proyecto utiliza Vola20 como medida "

                "de volatilidad móvil de 20 días."

            )

    else:

        st.markdown("### 💬 Respuesta basada en RAG")

        st.write(

            "La información más relevante encontrada en "

            "las fuentes del proyecto se muestra arriba. "

            "Revisa el contexto recuperado para consultar "

            "los resultados documentados."

        )


# =========================================================

# MÉTRICA DEL RETRIEVER

# =========================================================

with st.expander("📏 Evaluación del Retriever"):

    st.metric(

        "Recall@3",

        "0.80"

    )

    st.caption(

        "En la evaluación realizada en el notebook, "

        "el Retriever recuperó la fuente esperada en "

        "4 de 5 consultas usando Top-3."

    )
 