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

# Formateador personalizado para evitar KeyError si faltan claves
class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Añade atributos personalizados si no existen
        if not hasattr(record, 'class_name'):
            record.class_name = 'N/A'
        if not hasattr(record, 'function_name'):
            record.function_name = 'N/A'
        return super().format(record)

class Logger:
    def __init__(self):
        if not os.path.exists(TEXT_LOG_DIR):
            os.makedirs(TEXT_LOG_DIR)

        # Crear logger y configurar manejador
        self.logger = logging.getLogger('LoggerAVAL')
        self.logger.setLevel(logging.DEBUG)

        # Crear manejador de archivo
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(logging.DEBUG)

        # Aplicar el formateador personalizado
        formatter = CustomFormatter(
            '[%(asctime)s | %(name)s | %(class_name)s | %(function_name)s | %(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        # Evitar duplicados
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

    def debug(self, class_name, function_name, description, exc_info=False):
        self.logger.debug(
            description,
            extra={'class_name': class_name, 'function_name': function_name},
            exc_info=exc_info
        )

    def info(self, class_name, function_name, description, exc_info=False):
        self.logger.info(
            description,
            extra={'class_name': class_name, 'function_name': function_name},
            exc_info=exc_info
        )

    def warning(self, class_name, function_name, description, exc_info=False):
        self.logger.warning(
            description,
            extra={'class_name': class_name, 'function_name': function_name},
            exc_info=exc_info
        )

    def error(self, class_name, function_name, description, exc_info=False):
        self.logger.error(
            description,
            extra={'class_name': class_name, 'function_name': function_name},
            exc_info=exc_info
        )

    def critical(self, class_name, function_name, description, exc_info=False):
        self.logger.critical(
            description,
            extra={'class_name': class_name, 'function_name': function_name},
            exc_info=exc_info
        )

# Uso del logger
logger = Logger()
logger.debug('MiClase', 'mi_funcion', 'Este es un mensaje de debug')
logger.info('MiClase', 'mi_funcion', 'Este es un mensaje de info')
logger.warning('MiClase', 'mi_funcion', 'Este es un mensaje de warning')

# Registra un error con detalles de la excepción
try:
    1 / 0
except ZeroDivisionError as e:
    logger.error('MiClase', 'mi_funcion', f"Error en la operación: {str(e)}", exc_info=True)

logger.critical('MiClase', 'mi_funcion', 'Este es un mensaje de critical')