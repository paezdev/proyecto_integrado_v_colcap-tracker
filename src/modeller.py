import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from datetime import datetime, timedelta
import os

class StockPredictor:
    def __init__(self, data_file):
        # Definir rutas relativas para los datos de entrada
        self.data_path = os.path.join('src', 'static', 'data', data_file)
        self.df = pd.read_csv(self.data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.model = None
        self.feature_columns = ['SMA_7', 'SMA_21', 'Volatility_7', 'RSI', 'Momentum',
                              'BB_upper', 'BB_lower', 'Month', 'Quarter']

    def prepare_data(self):
        """Prepara los datos para el entrenamiento"""
        # Eliminar filas con valores NaN
        self.df = self.df.dropna(subset=self.feature_columns + ['Adj Close AVAL'])

        # Características (X) y objetivo (y)
        X = self.df[self.feature_columns]
        y = self.df['Adj Close AVAL']

        # División temporal (los últimos 20% para prueba)
        train_size = int(len(self.df) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

        return X_train, X_test, y_train, y_test

    def train(self, model_dir='src/static/models'):
        """Entrena el modelo y guarda el artefacto"""
        # Crear directorio si no existe
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'model.pkl')
        metrics_path = os.path.join(model_dir, 'metrics.csv')

        X_train, X_test, y_train, y_test = self.prepare_data()

        # Entrenar modelo
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Guardar modelo
        joblib.dump(self.model, model_path)

        # Evaluar modelo
        y_pred = self.model.predict(X_test)

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

        return metrics

    def predict(self, model_dir='src/static/models'):
        """Carga el modelo y realiza predicciones"""
        model_path = os.path.join(model_dir, 'model.pkl')

        if self.model is None:
            self.model = joblib.load(model_path)

        # Usar los últimos datos disponibles para predecir
        latest_data = self.df[self.feature_columns].iloc[-1:]
        prediction = self.model.predict(latest_data)[0]

        return prediction

def main():
    try:
        # Instanciar y entrenar el modelo
        predictor = StockPredictor('enriched_historical.csv')
        metrics = predictor.train()

        # Realizar una predicción
        next_day_prediction = predictor.predict()
        print(f"\nPredicción para el siguiente día: {next_day_prediction:.4f}")

        # Mostrar importancia de características
        feature_importance = pd.DataFrame({
            'feature': predictor.feature_columns,
            'importance': predictor.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nImportancia de características:")
        print(feature_importance)

        print("\nProceso de modelado completado exitosamente.")

    except Exception as e:
        print(f"\nError durante el proceso de modelado: {e}")

if __name__ == "__main__":
    main()