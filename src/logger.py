import logging
import os

# Directorio del archivo actual
BASE_DIR = os.path.dirname(__file__)

# Subcarpeta específica para logs de texto
TEXT_LOG_DIR = os.path.join(BASE_DIR, "logs", "text_logs")
os.makedirs(TEXT_LOG_DIR, exist_ok=True)

# Ruta completa al archivo .log
LOG_FILE = os.path.join(TEXT_LOG_DIR, "log_data.log")

def setup_logger():
    """Configura el sistema de logging para registrar eventos en consola y archivo."""
    logger = logging.getLogger("LoggerAVAL")
    logger.setLevel(logging.INFO)

    # Evitar duplicar handlers si ya están configurados
    if logger.handlers:
        return logger

    # Formato uniforme para los mensajes de log
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Log a archivo
    file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)

    # Log a consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Agregar handlers al logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger