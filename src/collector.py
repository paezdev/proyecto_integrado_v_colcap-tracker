import yfinance as yf
import pandas as pd
import os
from logger import setup_logger
from datetime import datetime

class DataCollector:
    def __init__(self, symbol, filepath):
        self.symbol = symbol
        self.filepath = filepath
        self.logger = setup_logger()

    def fetch_data(self):
        self.logger.info(f"Descargando datos para {self.symbol}")
        df = yf.download(self.symbol, progress=False, auto_adjust=False, actions=True)
        df.reset_index(inplace=True)
        return df

    def save_data(self, df):
        # Crear el directorio si no existe
        dir_path = os.path.dirname(self.filepath)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        downloaded_count = len(df)

        if os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0:
            old_df = pd.read_csv(self.filepath, parse_dates=["Date"])
            before_merge_count = len(old_df)
            merged_df = pd.concat([old_df, df]).drop_duplicates(subset="Date").sort_values("Date")
            after_merge_count = len(merged_df)
            new_rows_added = after_merge_count - before_merge_count
        else:
            merged_df = df
            new_rows_added = len(df)

        merged_df.to_csv(self.filepath, index=False)
        self.logger.info(f"Datos guardados en {self.filepath}")
        self.logger.info(f"Registros descargados: {downloaded_count}")
        self.logger.info(f"Nuevos registros agregados: {new_rows_added}")
        self.logger.info(f"Total de registros en el archivo: {len(merged_df)}")

        log_data_path = "log_data.csv"

        log_entry = {
            "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Símbolo": self.symbol,
            "Registros_descargados": downloaded_count,
            "Registros_agregados": new_rows_added,
            "Total_en_archivo": len(merged_df)
        }

        log_df = pd.DataFrame([log_entry])

        # Si el archivo no existe, crearlo con encabezado; si existe, agregar sin encabezado
        if not os.path.exists(log_data_path):
            log_df.to_csv(log_data_path, index=False)
        else:
            log_df.to_csv(log_data_path, mode='a', header=False, index=False)


if __name__ == "__main__":
    collector = DataCollector("AVAL", "src/static/historical.csv")
    try:
        data = collector.fetch_data()
        collector.save_data(data)

        # Si todo sale bien, también lo registra como "Éxito"
        log_data_path = "log_data.csv"
        from datetime import datetime
        log_entry = {
            "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Símbolo": collector.symbol,
            "Registros_descargados": len(data),
            "Registros_agregados": "Ver arriba",  # Ya está en save_data()
            "Total_en_archivo": "Ver arriba",
            "Estado": "Éxito"
        }
        pd.DataFrame([log_entry]).to_csv(log_data_path, mode='a', header=not os.path.exists(log_data_path), index=False)

    except Exception as e:
        collector.logger.error(f"Error al procesar datos: {str(e)}")

        # Registrar error en log_data.csv también
        log_data_path = "log_data.csv"
        from datetime import datetime
        log_entry = {
            "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Símbolo": collector.symbol,
            "Registros_descargados": 0,
            "Registros_agregados": 0,
            "Total_en_archivo": "Error",
            "Estado": f"Error: {str(e)}"
        }
        pd.DataFrame([log_entry]).to_csv(log_data_path, mode='a', header=not os.path.exists(log_data_path), index=False)