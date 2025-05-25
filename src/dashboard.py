import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import joblib
import os

# Importa el modelo ARIMA
from arima_model import ejecutar_arima_completo

# Configuración de la página
st.set_page_config(
    page_title="AVAL Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📊 AVAL Stock Analysis Dashboard")

# Definir rutas relativas
DATA_PATH = os.path.join('src', 'static', 'data', 'enriched_historical.csv')
HISTORICAL_PATH = os.path.join('src', 'static', 'data', 'historical.csv')  # Ajusta si tu archivo está en otra ruta
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

# Sidebar con filtros y mejoras
st.sidebar.header("Filtros")

# Selector de modelo
model_selector = st.sidebar.radio(
    "Selecciona el modelo a visualizar:",
    options=["ML Mejorado", "ARIMA", "Ambos"],
    index=0
)

# Botón para resetear filtros
if st.sidebar.button("Resetear filtros"):
    st.experimental_rerun()

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

# Selector de indicadores técnicos
indicator_options = ['RSI', 'Bandas de Bollinger', 'Volatilidad', 'Momentum']
selected_indicators = st.sidebar.multiselect(
    "Indicadores Técnicos a Visualizar",
    options=indicator_options,
    default=['RSI', 'Bandas de Bollinger']
)

# Filtrar datos por año
mask = (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])
filtered_df = df[mask].copy()

# ======================
# KPIs en la parte superior
# ======================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Precio Actual",
        f"${filtered_df['Adj Close AVAL'].iloc[-1]:.2f}",
        f"{filtered_df['Daily_Return'].iloc[-1]*100:.2f}%"
    )
    with st.expander("¿Qué es el Precio Actual?"):
        st.write("El precio ajustado de cierre del último día disponible en el rango seleccionado.")

with col2:
    volatility = filtered_df['Volatility_7'].iloc[-1]
    st.metric(
        "Volatilidad (7d)",
        f"{volatility:.2f}",
        f"{(volatility - filtered_df['Volatility_7'].iloc[-2]):.2f}"
    )
    with st.expander("¿Qué es la Volatilidad?"):
        st.write("La volatilidad mide la variabilidad de los precios en los últimos 7 días.")

with col3:
    current_sma = filtered_df['SMA_21'].iloc[-1]
    st.metric(
        "Media Móvil (21d)",
        f"${current_sma:.2f}",
        f"{(current_sma - filtered_df['Adj Close AVAL'].iloc[-1]):.2f}"
    )
    with st.expander("¿Qué es la Media Móvil?"):
        st.write("La media móvil suaviza el precio para identificar tendencias.")

with col4:
    cumulative_return = (filtered_df['Cumulative_Return'].iloc[-1] - 1) * 100
    st.metric(
        "Retorno Acumulado",
        f"{cumulative_return:.2f}%",
        f"{filtered_df['Daily_Return'].iloc[-1]*100:.2f}%"
    )
    with st.expander("¿Qué es el Retorno Acumulado?"):
        st.write("El retorno acumulado muestra la ganancia o pérdida total desde el inicio del periodo.")

with col5:
    rsi = filtered_df['RSI'].iloc[-1]
    st.metric(
        "RSI",
        f"{rsi:.2f}",
        "Sobrecomprado" if rsi > 70 else "Sobrevendido" if rsi < 30 else "Normal"
    )
    with st.expander("¿Qué es el RSI?"):
        st.write("El RSI (Relative Strength Index) es un indicador de momentum que mide la fuerza de los movimientos de precio.")

# ======================
# Gráficos principales con zoom y señales
# ======================
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

# Señales de trading (si tienes una columna 'Señal' en tu dataframe)
if 'Señal' in filtered_df.columns:
    buy_signals = filtered_df[filtered_df['Señal'] == 'COMPRA']
    sell_signals = filtered_df[filtered_df['Señal'] == 'VENTA']
    fig.add_trace(go.Scatter(
        x=buy_signals['Date'],
        y=buy_signals['Adj Close AVAL'],
        mode='markers',
        marker=dict(color='green', size=10, symbol='triangle-up'),
        name='Señal de Compra'
    ))
    fig.add_trace(go.Scatter(
        x=sell_signals['Date'],
        y=sell_signals['Adj Close AVAL'],
        mode='markers',
        marker=dict(color='red', size=10, symbol='triangle-down'),
        name='Señal de Venta'
    ))

# Zoom y rango interactivo
fig.update_layout(
    title='Precio AVAL y Medias Móviles',
    yaxis_title='Precio',
    xaxis_title='Fecha',
    template='plotly_dark',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)

st.plotly_chart(fig, use_container_width=True)

# ======================
# Gráfico de volumen con media móvil
# ======================
filtered_df['Volume_MA_21'] = filtered_df['Volume AVAL'].rolling(window=21).mean()
volume_fig = go.Figure()
volume_fig.add_trace(go.Bar(
    x=filtered_df['Date'],
    y=filtered_df['Volume AVAL'],
    name='Volumen'
))
volume_fig.add_trace(go.Scatter(
    x=filtered_df['Date'],
    y=filtered_df['Volume_MA_21'],
    name='Volumen MA 21d',
    line=dict(color='orange', dash='dot')
))
volume_fig.update_layout(
    title='Volumen de Negociación',
    yaxis_title='Volumen',
    xaxis_title='Fecha',
    template='plotly_dark',
    xaxis=dict(
        rangeslider=dict(visible=True),
        type="date"
    )
)
st.plotly_chart(volume_fig, use_container_width=True)

# ======================
# Indicadores técnicos seleccionados
# ======================
col1, col2 = st.columns(2)

if 'RSI' in selected_indicators:
    with col1:
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

if 'Bandas de Bollinger' in selected_indicators:
    with col2:
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

if 'Volatilidad' in selected_indicators:
    with col1:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Volatility_7'],
            name='Volatilidad 7d'
        ))
        fig_vol.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Volatility_14'],
            name='Volatilidad 14d'
        ))
        fig_vol.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Volatility_30'],
            name='Volatilidad 30d'
        ))
        fig_vol.update_layout(title='Volatilidad')
        st.plotly_chart(fig_vol, use_container_width=True)

if 'Momentum' in selected_indicators:
    with col2:
        fig_mom = go.Figure()
        fig_mom.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Momentum'],
            name='Momentum'
        ))
        fig_mom.update_layout(title='Momentum')
        st.plotly_chart(fig_mom, use_container_width=True)

# ======================
# Selector de modelo y predicción
# ======================
st.subheader("Comparación de Modelos de Predicción")

if model_selector in ["ML Mejorado", "Ambos"]:
    st.markdown("### Predicción del Modelo ML Mejorado")
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        selector = joblib.load(SELECTOR_PATH)
        selected_features = pd.read_csv(FEATURES_PATH)['0'].tolist()
        metrics = pd.read_csv(METRICS_PATH)

        # Preprocesar la última fila igual que en el entrenamiento
        last_data = df.copy().iloc[[-1]]
        all_features = [
            'High AVAL', 'Low AVAL', 'Open AVAL', 'Volume AVAL',
            'Month', 'Year', 'Quarter', 'SMA_7', 'SMA_21',
            'SMA_50', 'SMA_100', 'SMA_200',
            'Volatility_7', 'Daily_Return', 'RSI', 'Momentum',
            'BB_middle', 'BB_upper', 'BB_lower', 'Day_of_Week_Num',
            'Month_Sin', 'Month_Cos', 'Day_of_Week_Sin', 'Day_of_Week_Cos',
            'Price_Ratio', 'High_Low_Ratio', 'Volume_Change',
            'Volatility_14', 'Volatility_30', 'ROC_5', 'ROC_10', 'ROC_20',
            'EMA_5', 'EMA_10', 'EMA_20', 'SMA_EMA_5_Diff', 'SMA_EMA_10_Diff',
            'SMA_Cross_5_20', 'SMA_Cross_10_50', 'Volatility_Ratio_7_30'
        ]
        for col in all_features:
            if col not in last_data.columns:
                last_data[col] = 0  # Valor por defecto

        available_features = [col for col in all_features if col in last_data.columns]
        X_last = last_data[available_features]

        # Escalar y seleccionar
        X_last_scaled = scaler.transform(X_last)
        X_last_selected = selector.transform(X_last_scaled)
        prediction = model.predict(X_last_selected)[0]

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

if model_selector in ["ARIMA", "Ambos"]:
    st.markdown("### Predicción y Métricas del Modelo ARIMA")
    try:
        arima_result = ejecutar_arima_completo(
            ruta_archivo=HISTORICAL_PATH,
            order=(3,1,1),
            graficar=False
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE ARIMA", f"{arima_result['mae']:.4f}")
        with col2:
            st.metric("RMSE ARIMA", f"{arima_result['rmse']:.4f}")
        with col3:
            st.metric("MAPE ARIMA", f"{arima_result['mape']:.2f}%")
        with col4:
            st.metric("R² ARIMA", f"{arima_result['r2']:.4f}")

        st.write(f"📅 **Predicción ARIMA para el siguiente día ({arima_result['next_date'].date()}):** {arima_result['forecast'].values[0]:.4f}")

        # Mostrar gráfico ARIMA (matplotlib)
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12,5))
        plt.plot(arima_result['serie'], label='Precio Real', color='blue')
        plt.plot(arima_result['pred'], label='Predicción ARIMA', color='orange', linestyle='--')
        plt.scatter(arima_result['next_date'], arima_result['forecast'].values[0], color='red', label='Predicción siguiente día', zorder=5)
        plt.title(f'ARIMA(3,1,1) - R² = {arima_result["r2"]:.4f}')
        plt.xlabel('Fecha')
        plt.ylabel('Adj Close AVAL')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error al ejecutar el modelo ARIMA: {e}")

# ======================
# Historial de señales y exportación
# ======================
if 'Señal' in filtered_df.columns:
    st.subheader("Historial de Señales de Trading")
    st.dataframe(filtered_df[['Date', 'Adj Close AVAL', 'Señal']].tail(10))
    st.download_button(
        label="Descargar historial de señales",
        data=filtered_df[['Date', 'Adj Close AVAL', 'Señal']].to_csv(index=False),
        file_name='historial_senales.csv',
        mime='text/csv'
    )

# ======================
# Información adicional
# ======================
with st.expander("ℹ️ Información del Dataset"):
    st.write("Estadísticas Descriptivas:")
    stats_df = df.drop(columns=['Date']).describe()
    st.dataframe(stats_df)

    st.write("Últimos Registros:")
    display_df = df.copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    st.dataframe(display_df.tail())