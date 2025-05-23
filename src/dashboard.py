import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
import joblib
import os

# Configuración de la página
st.set_page_config(
    page_title="AVAL Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Título principal
st.title("📊 AVAL Stock Analysis Dashboard")

# Definir rutas relativas
DATA_PATH = os.path.join('src', 'static', 'data', 'enriched_historical.csv')
MODEL_PATH = os.path.join('src', 'static', 'models', 'model.pkl')
METRICS_PATH = os.path.join('src', 'static', 'models', 'metrics.csv')

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()

# Sidebar con filtros
st.sidebar.header("Filtros")
year_range = st.sidebar.slider(
    "Seleccionar Rango de Años",
    min_value=int(df['Year'].min()),
    max_value=int(df['Year'].max()),
    value=(int(df['Year'].min()), int(df['Year'].max()))
)

# Filtrar datos por año
mask = (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])
filtered_df = df[mask]

# KPIs en la parte superior
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Precio Actual",
        f"${filtered_df['Adj Close AVAL'].iloc[-1]:.2f}",
        f"{filtered_df['Daily_Return'].iloc[-1]*100:.2f}%"
    )

with col2:
    volatility = filtered_df['Volatility_7'].iloc[-1]
    st.metric(
        "Volatilidad (7d)",
        f"{volatility:.2f}",
        f"{(volatility - filtered_df['Volatility_7'].iloc[-2]):.2f}"
    )

with col3:
    current_sma = filtered_df['SMA_21'].iloc[-1]
    st.metric(
        "Media Móvil (21d)",
        f"${current_sma:.2f}",
        f"{(current_sma - filtered_df['Adj Close AVAL'].iloc[-1]):.2f}"
    )

with col4:
    cumulative_return = (filtered_df['Cumulative_Return'].iloc[-1] - 1) * 100
    st.metric(
        "Retorno Acumulado",
        f"{cumulative_return:.2f}%",
        f"{filtered_df['Daily_Return'].iloc[-1]*100:.2f}%"
    )

with col5:
    rsi = filtered_df['RSI'].iloc[-1]
    st.metric(
        "RSI",
        f"{rsi:.2f}",
        "Sobrecomprado" if rsi > 70 else "Sobrevendido" if rsi < 30 else "Normal"
    )

# Gráficos
st.subheader("Análisis de Precio y Volumen")
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=filtered_df['Date'],
    open=filtered_df['Open AVAL'],
    high=filtered_df['High AVAL'],
    low=filtered_df['Low AVAL'],
    close=filtered_df['Close AVAL'],
    name='OHLC'
))

fig.add_trace(go.Scatter(
    x=filtered_df['Date'],
    y=filtered_df['SMA_21'],
    name='SMA 21',
    line=dict(color='orange')
))

fig.update_layout(
    title='Precio AVAL y Media Móvil 21 días',
    yaxis_title='Precio',
    xaxis_title='Fecha',
    template='plotly_dark'
)

st.plotly_chart(fig, use_container_width=True)

# Gráfico de volumen
volume_fig = px.bar(
    filtered_df,
    x='Date',
    y='Volume AVAL',
    title='Volumen de Negociación'
)
st.plotly_chart(volume_fig, use_container_width=True)

# Indicadores técnicos
col1, col2 = st.columns(2)

with col1:
    # RSI
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['RSI'],
        name='RSI'
    ))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    fig_rsi.update_layout(title='Relative Strength Index (RSI)')
    st.plotly_chart(fig_rsi, use_container_width=True)

with col2:
    # Bandas de Bollinger
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['BB_upper'],
        name='Banda Superior',
        line=dict(color='gray', dash='dash')
    ))
    fig_bb.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['BB_middle'],
        name='Media Móvil',
        line=dict(color='blue')
    ))
    fig_bb.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df['BB_lower'],
        name='Banda Inferior',
        line=dict(color='gray', dash='dash'),
        fill='tonexty'
    ))
    fig_bb.update_layout(title='Bandas de Bollinger')
    st.plotly_chart(fig_bb, use_container_width=True)

# Predicción
st.subheader("Predicción del Modelo")
try:
    model = joblib.load(MODEL_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    
    last_data = df[['SMA_7', 'SMA_21', 'Volatility_7', 'RSI', 'Momentum', 
                    'BB_upper', 'BB_lower', 'Month', 'Quarter']].iloc[-1:]
    prediction = model.predict(last_data)[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Predicción Próximo Día",
            f"${prediction:.2f}",
            f"{(prediction - df['Adj Close AVAL'].iloc[-1]):.2f}"
        )
    with col2:
        st.metric("RMSE del Modelo", f"{metrics['RMSE'].iloc[0]:.4f}")
    with col3:
        st.metric("R² del Modelo", f"{metrics['R2'].iloc[0]:.4f}")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

# Información adicional
with st.expander("ℹ️ Información del Dataset"):
    st.write("Estadísticas Descriptivas:")
    st.dataframe(df.describe())
    
    st.write("Últimos Registros:")
    st.dataframe(df.tail())