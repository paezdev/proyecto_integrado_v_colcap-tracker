import logging
import os
import datetime

# Directorio de logs (dentro de src/logs/text_logs)
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
TEXT_LOG_DIR = os.path.join(LOG_DIR, "text_logs")  # Directorio específico para logs de texto
os.makedirs(TEXT_LOG_DIR, exist_ok=True)  # Crear el directorio si no existe

# Generar el nombre del archivo de log usando la fecha y hora actual
log_filename = f"aval_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE = os.path.join(TEXT_LOG_DIR, log_filename)

def setup_logger():
    """Configura el sistema de logging para registrar eventos en consola y archivo."""
    logger = logging.getLogger("LoggerAVAL")
    logger.setLevel(logging.INFO)

    # Evitar duplicar handlers si ya están configurados
    if logger.handlers:
        return logger

    # Formato uniforme
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Log a archivo
    file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')  # Modo 'a' para append
    file_handler.setFormatter(formatter)

    # Log a consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Agregar handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger