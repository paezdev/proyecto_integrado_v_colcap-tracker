import csv
import os
from datetime import datetime

def init_csv_log(file_path="log_data.csv"):
    """Inicializa el archivo CSV si no existe y agrega los encabezados."""
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["Fecha", "Símbolo", "Registros_descargados", "Registros_agregados", "Total_en_archivo", "Estado"])
            writer.writeheader()

def write_csv_log(symbol, downloaded_count, new_rows_added, total_count, status):
    """Escribe un nuevo registro en el archivo CSV de log."""
    log_entry = {
        "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Símbolo": symbol,
        "Registros_descargados": downloaded_count,
        "Registros_agregados": new_rows_added,
        "Total_en_archivo": total_count,
        "Estado": status
    }

    log_data_path = "log_data.csv"

    # Si el archivo ya existe, solo agregamos la entrada sin encabezado
    with open(log_data_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Fecha", "Símbolo", "Registros_descargados", "Registros_agregados", "Total_en_archivo", "Estado"])
        writer.writerow(log_entry)