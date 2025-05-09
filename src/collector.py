import yfinance as yf
import pandas as pd
import os
from logger import Logger  # Importar la clase Logger
from datetime import datetime
import csv_logger  # Este es el archivo para escribir el log en formato CSV

class DataCollector:
    def __init__(self, symbol, filepath):
        """Inicializa el recolector de datos con el símbolo y archivo especificado."""
        self.symbol = symbol
        self.filepath = filepath
        self.logger = Logger()  # Instanciar Logger

    def fetch_data(self):
        """Descarga los datos de un símbolo usando yfinance."""
        self.logger.info('DataCollector', 'fetch_data', f"Descargando datos para {self.symbol}")
        df = yf.download(self.symbol, progress=False, auto_adjust=False, actions=True)
        df.reset_index(inplace=True)

        # LIMPIAR columnas por si vienen jerárquicas
        df.columns = [col if isinstance(col, str) else ' '.join(col).strip() for col in df.columns]
        return df

    def save_data(self, df):
        """Guarda los datos descargados en un archivo CSV y registra detalles."""
        # Crear el directorio si no existe
        dir_path = os.path.dirname(self.filepath)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        downloaded_count = len(df)

        # Leer archivo histórico si existe
        if os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0:
            old_df = pd.read_csv(self.filepath, parse_dates=["Date"])

            # LIMPIAR columnas del archivo viejo
            old_df.columns = [col if isinstance(col, str) else ' '.join(col).strip() for col in old_df.columns]

            before_merge_count = len(old_df)
            # Combinar datos antiguos con los nuevos, eliminando duplicados por fecha
            merged_df = pd.concat([old_df, df]).drop_duplicates(subset="Date").sort_values("Date")
            after_merge_count = len(merged_df)
            new_rows_added = after_merge_count - before_merge_count
        else:
            merged_df = df
            new_rows_added = len(df)

        # LIMPIAR columnas antes de guardar para evitar multi-index accidental
        merged_df.columns = [col if isinstance(col, str) else ' '.join(col).strip() for col in merged_df.columns]

        # Guardar datos en el archivo CSV
        merged_df.to_csv(self.filepath, index=False)
        self.logger.info('DataCollector', 'save_data', f"Datos guardados en {self.filepath}")
        self.logger.info('DataCollector', 'save_data', f"Registros descargados: {downloaded_count}")
        self.logger.info('DataCollector', 'save_data', f"Nuevos registros agregados: {new_rows_added}")
        self.logger.info('DataCollector', 'save_data', f"Total de registros en el archivo: {len(merged_df)}")

        # Registrar en archivo CSV centralizado
        csv_logger.write_csv_log(self.symbol, downloaded_count, new_rows_added, len(merged_df), "Éxito")

    def handle_error(self, error_message):
        """Maneja errores y los registra en el log."""
        self.logger.error('DataCollector', 'handle_error', f"Error al procesar datos: {error_message}")

        # Registrar error en archivo CSV de log
        log_entry = {
            "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Símbolo": self.symbol,
            "Registros_descargados": 0,
            "Registros_agregados": 0,
            "Total_en_archivo": "Error",
            "Estado": f"Error: {error_message}"
        }
        csv_logger.write_csv_log(self.symbol, 0, 0, "Error", f"Error: {error_message}")


if __name__ == "__main__":
    collector = DataCollector("AVAL", "src/static/historical.csv")
    try:
        data = collector.fetch_data()
        collector.save_data(data)

    except Exception as e:
        collector.handle_error(str(e))