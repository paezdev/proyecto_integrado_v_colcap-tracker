import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# --- Funciones principales ---

def cargar_datos(ruta_archivo='historical.csv'):
    """Carga y prepara los datos desde un archivo CSV."""
    df = pd.read_csv(ruta_archivo)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def obtener_serie(df, columna='Adj Close AVAL'):
    """Extrae la serie temporal de interés."""
    return df[columna].dropna()

def entrenar_arima(serie, order=(3,1,1)):
    """Entrena un modelo ARIMA con los parámetros especificados."""
    model = ARIMA(serie, order=order)
    return model.fit()

def predecir_arima(fit, serie):
    """Realiza predicciones dentro de muestra y para el siguiente día."""
    pred = fit.predict(start=serie.index[1], end=serie.index[-1], typ='levels')
    pred.index = serie.index[1:]
    forecast = fit.forecast(steps=1)
    next_date = serie.index[-1] + pd.Timedelta(days=1)
    return pred, forecast, next_date

def calcular_metricas(y_true, y_pred):
    """Calcula MAE, RMSE, MAPE y R²."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, mape, r2

def graficar_arima(serie, pred, forecast, next_date, order=(3,1,1)):
    """Grafica los resultados del modelo ARIMA."""
    plt.figure(figsize=(16,6))
    plt.plot(serie, label='Precio Real', color='blue')
    plt.plot(pred, label='Predicción ARIMA', color='orange', linestyle='--')
    plt.scatter(next_date, forecast.values[0], color='red', label=f'Predicción siguiente día ({next_date.date()})', zorder=5)
    plt.axvline(x=serie.index[-1], color='gray', linestyle=':', label='Último dato')
    plt.title(f'Precio Real vs Ajuste ARIMA{order} y Predicción del Siguiente Día')
    plt.xlabel('Fecha')
    plt.ylabel('Adj Close AVAL')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def ejecutar_arima_completo(ruta_archivo='historical.csv', columna='Adj Close AVAL', order=(3,1,1), graficar=True):
    """Ejecuta el flujo completo de ARIMA y retorna resultados y métricas."""
    df = cargar_datos(ruta_archivo)
    serie = obtener_serie(df, columna)
    modelo = entrenar_arima(serie, order)
    pred, forecast, next_date = predecir_arima(modelo, serie)
    mae, rmse, mape, r2 = calcular_metricas(serie[1:], pred)
    if graficar:
        graficar_arima(serie, pred, forecast, next_date, order)
    print("📈 Métricas de Evaluación:")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAPE = {mape:.2f}%")
    print(f"R²   = {r2:.4f}")
    print(f"\n📅 Predicción para el siguiente día ({next_date.date()}): {forecast.values[0]:.4f}")
    return {
        'serie': serie,
        'pred': pred,
        'forecast': forecast,
        'next_date': next_date,
        'modelo': modelo,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }

# Si ejecutas este archivo directamente, muestra el resultado
if __name__ == "__main__":
    ejecutar_arima_completo()