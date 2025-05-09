import logging
import os

# Definir el directorio de logs
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Definir la ruta al archivo de log
LOG_FILE = os.path.join(LOG_DIR, "logfile.log")

# Configuración del logger
def setup_logger():
    """Configura el sistema de logging para registrar eventos tanto en consola como en archivo."""
    logger = logging.getLogger("LoggerAVAL")
    logger.setLevel(logging.INFO)  # Nivel de log (INFO para mensajes generales)

    # Formato del log
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Manejador de archivo (para guardar logs en un archivo)
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(log_format)

    # Manejador de consola (para imprimir los logs en la consola)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)

    # Añadir manejadores al logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger