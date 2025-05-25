import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import joblib
import os

# Configuración de la página
st.set_page_config(
    page_title="AVAL Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📊 AVAL Stock Analysis Dashboard")

# Definir rutas relativas
DATA_PATH = os.path.join('src', 'static', 'data', 'enriched_historical.csv')
MODEL_PATH = os.path.join('src', 'static', 'models', 'model.pkl')
METRICS_PATH = os.path.join('src', 'static', 'models', 'metrics.csv')
SCALER_PATH = os.path.join('src', 'static', 'models', 'scaler.pkl')
SELECTOR_PATH = os.path.join('src', 'static', 'models', 'feature_selector.pkl')
FEATURES_PATH = os.path.join('src', 'static', 'models', 'selected_features.csv')

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.floor('ms')
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

# Selección de medias móviles a visualizar
ma_options = ['SMA_21', 'SMA_50', 'SMA_100', 'SMA_200']
selected_mas = st.sidebar.multiselect(
    "Medias Móviles a Visualizar",
    options=ma_options,
    default=['SMA_21', 'SMA_50', 'SMA_200']
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

# Gráficos principales
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

# Agregar medias móviles seleccionadas
ma_colors = {
    'SMA_21': 'orange',
    'SMA_50': 'green',
    'SMA_100': 'blue',
    'SMA_200': 'purple'
}
for ma in selected_mas:
    fig.add_trace(go.Scatter(
        x=filtered_df['Date'],
        y=filtered_df[ma],
        name=ma,
        line=dict(color=ma_colors.get(ma, 'gray'), dash='solid')
    ))

# Visualización de cruces de medias móviles
cross_5_20 = filtered_df[filtered_df['SMA_Cross_5_20'] == 1]
cross_10_50 = filtered_df[filtered_df['SMA_Cross_10_50'] == 1]
fig.add_trace(go.Scatter(
    x=cross_5_20['Date'],
    y=cross_5_20['Adj Close AVAL'],
    mode='markers',
    marker=dict(color='red', size=8, symbol='triangle-up'),
    name='Cruz EMA5>EMA20'
))
fig.add_trace(go.Scatter(
    x=cross_10_50['Date'],
    y=cross_10_50['Adj Close AVAL'],
    mode='markers',
    marker=dict(color='cyan', size=8, symbol='star'),
    name='Cruz EMA10>SMA50'
))

fig.update_layout(
    title='Precio AVAL y Medias Móviles',
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

# ======================
# Predicción con el pipeline completo
# ======================
st.subheader("Predicción del Modelo Mejorado")
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    selector = joblib.load(SELECTOR_PATH)
    selected_features = pd.read_csv(FEATURES_PATH)['0'].tolist()
    metrics = pd.read_csv(METRICS_PATH)

    # Preprocesar la última fila igual que en el entrenamiento
    last_data = df.copy().iloc[[-1]]

    # Verificar que todas las features seleccionadas estén presentes
    missing_features = [col for col in selected_features if col not in last_data.columns]
    if missing_features:
        st.warning(f"Faltan las siguientes features en los datos: {missing_features}")
        for col in missing_features:
            last_data[col] = 0  # Valor por defecto

    # Usar SOLO las features seleccionadas
    X_last = last_data[selected_features]
    X_last_scaled = scaler.transform(X_last)
    prediction = model.predict(X_last_scaled)[0]  # No necesitas usar selector.transform aquí

    # Señal de trading
    last_value = last_data['Adj Close AVAL'].values[0]
    percent_change = ((prediction - last_value) / last_value) * 100
    if percent_change > 0:
        signal = f"COMPRA (↑ {percent_change:.2f}%)"
    elif percent_change < 0:
        signal = f"VENTA (↓ {abs(percent_change):.2f}%)"
    else:
        signal = "MANTENER (sin cambio)"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Predicción Próximo Día",
            f"${prediction:.2f}",
            f"{(prediction - last_value):.2f}"
        )
        st.write(f"**Señal:** {signal}")
    with col2:
        st.metric("RMSE del Modelo", f"{metrics['RMSE'].iloc[0]:.4f}")
    with col3:
        st.metric("R² del Modelo", f"{metrics['R2'].iloc[0]:.4f}")

    # Importancia de features (si la guardaste)
    st.subheader("Importancia de las Features Seleccionadas")
    if hasattr(model, 'coef_'):
        importances = pd.Series(model.coef_, index=selected_features)
        importances = importances.abs().sort_values(ascending=False)
        st.bar_chart(importances)
    else:
        st.info("El modelo no tiene coeficientes de importancia disponibles.")

except Exception as e:
    st.error(f"Error al cargar el modelo mejorado: {e}")

# Información adicional
with st.expander("ℹ️ Información del Dataset"):
    st.write("Estadísticas Descriptivas:")
    stats_df = df.drop(columns=['Date']).describe()
    st.dataframe(stats_df)

    st.write("Últimos Registros:")
    display_df = df.copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    st.dataframe(display_df.tail())