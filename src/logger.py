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

class Logger:
    def __init__(self):
        if not os.path.exists(TEXT_LOG_DIR):
            os.makedirs(TEXT_LOG_DIR)

        # Configurar el archivo de log con el formato requerido
        logging.basicConfig(
            filename=LOG_FILE,
            level=logging.DEBUG,  # Para capturar todos los niveles (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format='[%(asctime)s | %(name)s | %(class_name)s | %(function_name)s | %(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        self.logger = logging.getLogger('LoggerAVAL')

    def debug(self, class_name, function_name, description):
        """Registra un mensaje de nivel DEBUG."""
        self.logger.debug(
            description,
            extra={'class_name': class_name, 'function_name': function_name}
        )

    def info(self, class_name, function_name, description):
        """Registra un mensaje de nivel INFO."""
        self.logger.info(
            description,
            extra={'class_name': class_name, 'function_name': function_name}
        )

    def warning(self, class_name, function_name, description):
        """Registra un mensaje de nivel WARNING."""
        self.logger.warning(
            description,
            extra={'class_name': class_name, 'function_name': function_name}
        )

    def error(self, class_name, function_name, description):
        """Registra un mensaje de nivel ERROR."""
        self.logger.error(
            description,
            extra={'class_name': class_name, 'function_name': function_name}
        )

    def critical(self, class_name, function_name, description):
        """Registra un mensaje de nivel CRITICAL."""
        self.logger.critical(
            description,
            extra={'class_name': class_name, 'function_name': function_name}
        )

logger = Logger()

# Registra logs con diferentes niveles
logger.debug('MiClase', 'mi_funcion', 'Este es un mensaje de debug')
logger.info('MiClase', 'mi_funcion', 'Este es un mensaje de info')
logger.warning('MiClase', 'mi_funcion', 'Este es un mensaje de warning')
logger.error('MiClase', 'mi_funcion', 'Este es un mensaje de error')
logger.critical('MiClase', 'mi_funcion', 'Este es un mensaje de critical')