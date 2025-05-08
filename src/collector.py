import yfinance as yf
import pandas as pd
import os
from logger import setup_logger

class DataCollector:
    def __init__(self, symbol, filepath):
        self.symbol = symbol
        self.filepath = filepath
        self.logger = setup_logger()

    def fetch_data(self):
        self.logger.info(f"Descargando datos para {self.symbol}")
        df = yf.download(self.symbol, progress=False)
        df.reset_index(inplace=True)
        return df

    def save_data(self, df):
        if os.path.exists(self.filepath):
            old_df = pd.read_csv(self.filepath, parse_dates=["Date"])
            merged_df = pd.concat([old_df, df]).drop_duplicates(subset="Date").sort_values("Date")
        else:
            merged_df = df
        merged_df.to_csv(self.filepath, index=False)
        self.logger.info(f"Datos guardados en {self.filepath}")
