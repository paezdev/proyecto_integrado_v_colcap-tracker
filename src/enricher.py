import pandas as pd
import numpy as np
from datetime import datetime
import os

class DataEnricher:
    def __init__(self, input_file):
        # Definir rutas relativas
        self.input_path = os.path.join('src', 'static', 'data', input_file)
        self.df = pd.read_csv(self.input_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])

    def add_temporal_features(self):
        """Añade características temporales"""
        self.df['Day_of_Week'] = self.df['Date'].dt.day_name()
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Quarter'] = self.df['Date'].dt.quarter

    def add_technical_indicators(self):
        """Añade indicadores técnicos"""
        # Medias móviles
        self.df['SMA_7'] = self.df['Adj Close AVAL'].rolling(window=7).mean()
        self.df['SMA_21'] = self.df['Adj Close AVAL'].rolling(window=21).mean()

        # Volatilidad
        self.df['Volatility_7'] = self.df['Adj Close AVAL'].rolling(window=7).std()

        # Retornos
        self.df['Daily_Return'] = self.df['Adj Close AVAL'].pct_change()
        self.df['Cumulative_Return'] = (1 + self.df['Daily_Return']).cumprod()

        # RSI
        delta = self.df['Adj Close AVAL'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

        # Momentum
        self.df['Momentum'] = self.df['Adj Close AVAL'].diff(periods=7)

        # Bandas de Bollinger
        self.df['BB_middle'] = self.df['Adj Close AVAL'].rolling(window=20).mean()
        std_dev = self.df['Adj Close AVAL'].rolling(window=20).std()
        self.df['BB_upper'] = self.df['BB_middle'] + (std_dev * 2)
        self.df['BB_lower'] = self.df['BB_middle'] - (std_dev * 2)

    def enrich_data(self, output_file):
        """Ejecuta todo el proceso de enriquecimiento"""
        self.add_temporal_features()
        self.add_technical_indicators()

        # Crear el directorio si no existe
        output_dir = os.path.join('src', 'static', 'data')
        os.makedirs(output_dir, exist_ok=True)

        # Guarda los datos enriquecidos
        output_path = os.path.join(output_dir, output_file)
        self.df.to_csv(output_path, index=False)
        return self.df

def main():
    try:
        # Ejecutar el enriquecimiento
        enricher = DataEnricher('historical.csv')
        enriched_df = enricher.enrich_data('enriched_historical.csv')

        # Mostrar las nuevas columnas y primeras filas
        print("\nColumnas en el dataset enriquecido:")
        print(enriched_df.columns.tolist())
        print("\nPrimeras filas del dataset enriquecido:")
        print(enriched_df.head())

        # Mostrar estadísticas básicas de los nuevos indicadores
        print("\nEstadísticas de los nuevos indicadores:")
        new_indicators = ['SMA_7', 'SMA_21', 'Volatility_7', 'Daily_Return', 'RSI', 'Momentum']
        print(enriched_df[new_indicators].describe())

        print("\nProceso de enriquecimiento completado exitosamente.")

    except Exception as e:
        print(f"\nError durante el proceso de enriquecimiento: {e}")

if __name__ == "__main__":
    main()