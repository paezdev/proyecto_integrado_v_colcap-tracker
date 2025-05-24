import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import joblib
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns

class StockPredictor:
    def __init__(self, data_file):
        # Definir rutas relativas para los datos de entrada
        self.data_path = os.path.join('src', 'static', 'data', data_file)
        self.df = pd.read_csv(self.data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None

        # Definir todas las características disponibles
        self.all_features = [
            'High AVAL', 'Low AVAL', 'Open AVAL', 'Volume AVAL',
            'Month', 'Year', 'Quarter', 'SMA_7', 'SMA_21',
            'Volatility_7', 'Daily_Return', 'RSI', 'Momentum',
            'BB_middle', 'BB_upper', 'BB_lower', 'Day_of_Week_Num',
            'Month_Sin', 'Month_Cos', 'Day_of_Week_Sin', 'Day_of_Week_Cos',
            'Price_Ratio', 'High_Low_Ratio', 'Volume_Change',
            'Volatility_14', 'Volatility_30', 'ROC_5', 'ROC_10', 'ROC_20',
            'EMA_5', 'EMA_10', 'EMA_20', 'SMA_EMA_5_Diff', 'SMA_EMA_10_Diff'
        ]

    def prepare_data(self):
        """Prepara los datos para el entrenamiento"""
        # Eliminar filas con valores NaN
        self.df = self.df.dropna()

        # Definir características y objetivo
        target_col = 'Adj Close AVAL'
        date_cols = ['Date']
        drop_cols = ['Close AVAL', 'Dividends AVAL', 'Stock Splits AVAL', 'Cumulative_Return']

        # Filtrar columnas disponibles
        available_features = [col for col in self.all_features if col in self.df.columns]

        # Características (X) y objetivo (y)
        X = self.df[available_features]
        y = self.df[target_col]

        # División temporal (los últimos 20% para prueba)
        train_size = int(len(self.df) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

        return X_train, X_test, y_train, y_test, available_features

    def train(self, model_dir='src/static/models'):
        """Entrena el modelo y guarda el artefacto"""
        # Crear directorio si no existe
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        selector_path = os.path.join(model_dir, 'feature_selector.pkl')
        features_path = os.path.join(model_dir, 'selected_features.csv')
        metrics_path = os.path.join(model_dir, 'metrics.csv')

        X_train, X_test, y_train, y_test, available_features = self.prepare_data()

        # Escalar características
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Selección de características
        selector_model = RandomForestRegressor(n_estimators=100, random_state=42)
        selector_model.fit(X_train_scaled, y_train)

        # Seleccionar características importantes
        self.feature_selector = SelectFromModel(selector_model, threshold='0.5*mean')
        self.feature_selector.fit(X_train_scaled, y_train)

        X_train_selected = self.feature_selector.transform(X_train_scaled)
        X_test_selected = self.feature_selector.transform(X_test_scaled)

        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_features = [available_features[i] for i in selected_indices]

        print(f"Características seleccionadas: {self.selected_features}")
        print(f"Número de características seleccionadas: {len(self.selected_features)}")

        # Entrenar modelo ElasticNet
        self.model = ElasticNet(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=10000,
            random_state=42
        )
        self.model.fit(X_train_selected, y_train)

        # Guardar modelo y componentes
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_selector, selector_path)
        pd.Series(self.selected_features).to_csv(features_path, index=False)

        # Evaluar modelo
        y_pred = self.model.predict(X_test_selected)

        # Calcular métricas
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }

        # Imprimir resultados
        print("\nResultados de la evaluación del modelo:")
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"MAE: {metrics['MAE']:.4f}")
        print(f"R2: {metrics['R2']:.4f}")

        # Guardar métricas
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

        # Generar gráficos
        self.generate_plots(y_test, y_pred, model_dir)

        return metrics

    def generate_plots(self, y_test, y_pred, model_dir):
        """Genera gráficos para visualizar el rendimiento del modelo"""
        # Crear directorio para gráficos
        plots_dir = os.path.join(model_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Gráfico de predicciones vs valores reales
        plt.figure(figsize=(12, 6))
        test_dates = self.df['Date'].iloc[-len(y_test):]
        plt.plot(test_dates, y_test, label='Valores reales', color='blue')
        plt.plot(test_dates, y_pred, label='Predicciones', color='red', linestyle='--')
        plt.title('Precio ajustado: Predicción vs Valor real')
        plt.xlabel('Fecha')
        plt.ylabel('Adj Close AVAL')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'predictions_vs_actual.png'))
        plt.close()

        # Gráfico de distribución de errores
        errors = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.title('Distribución de errores')
        plt.xlabel('Error')
        plt.ylabel('Frecuencia')
        plt.savefig(os.path.join(plots_dir, 'error_distribution.png'))
        plt.close()

    def predict_next_day(self, model_dir='src/static/models'):
        """Predice el valor para el siguiente día"""
        # Cargar modelo y componentes si no están cargados
        if self.model is None:
            model_path = os.path.join(model_dir, 'model.pkl')
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            selector_path = os.path.join(model_dir, 'feature_selector.pkl')
            features_path = os.path.join(model_dir, 'selected_features.csv')

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_selector = joblib.load(selector_path)
            self.selected_features = pd.read_csv(features_path)['0'].tolist()

        # Obtener la última fila de datos
        last_row = self.df.iloc[-1:].copy()

        # Obtener la fecha más reciente y calcular la siguiente
        last_date = last_row['Date'].iloc[0]
        next_date = last_date + timedelta(days=1)

        # Filtrar columnas disponibles
        available_features = [col for col in self.all_features if col in self.df.columns]

        # Preparar datos para predicción
        X_last = last_row[available_features]

        # Escalar y seleccionar características
        X_last_scaled = self.scaler.transform(X_last)
        X_last_selected = self.feature_selector.transform(X_last_scaled)

        # Predecir
        prediction = self.model.predict(X_last_selected)[0]

        # Obtener el último valor real
        last_value = last_row['Adj Close AVAL'].values[0]

        # Calcular el cambio porcentual
        percent_change = ((prediction - last_value) / last_value) * 100

        # Determinar señal
        if percent_change > 0:
            signal = f"COMPRA (se espera un aumento de {percent_change:.2f}%)"
        elif percent_change < 0:
            signal = f"VENTA (se espera una disminución de {abs(percent_change):.2f}%)"
        else:
            signal = "MANTENER (no se espera cambio significativo)"

        # Crear DataFrame con la predicción
        prediction_df = pd.DataFrame({
            'Fecha': [next_date],
            'Último valor conocido': [last_value],
            'Predicción': [prediction],
            'Cambio porcentual': [percent_change],
            'Señal': [signal.split(' ')[0]]  # Solo guardar COMPRA, VENTA o MANTENER
        })

        # Guardar predicción
        predictions_dir = os.path.join('src', 'static', 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        prediction_path = os.path.join(predictions_dir, 'next_day_prediction.csv')
        prediction_df.to_csv(prediction_path, index=False)

        return {
            'last_date': last_date,
            'next_date': next_date,
            'last_value': last_value,
            'prediction': prediction,
            'percent_change': percent_change,
            'signal': signal
        }

def main():
    try:
        # Instanciar y entrenar el modelo
        predictor = StockPredictor('enriched_historical.csv')
        metrics = predictor.train()

        # Realizar una predicción para el siguiente día
        prediction_result = predictor.predict_next_day()

        print("\nPredicción para el siguiente día:")
        print(f"Fecha de la última observación: {prediction_result['last_date'].strftime('%Y-%m-%d')}")
        print(f"Fecha de la predicción: {prediction_result['next_date'].strftime('%Y-%m-%d')}")
        print(f"Último valor conocido: {prediction_result['last_value']:.4f}")
        print(f"Predicción: {prediction_result['prediction']:.4f}")
        print(f"Cambio porcentual: {prediction_result['percent_change']:.2f}%")
        print(f"Señal: {prediction_result['signal']}")

        print("\nProceso de modelado completado exitosamente.")

    except Exception as e:
        print(f"\nError durante el proceso de modelado: {e}")

if __name__ == "__main__":
    main()