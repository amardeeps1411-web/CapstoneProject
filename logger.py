import logging
import os
import sys

LOG_DIR = "logs"
LOG_FILE = "app.log"

def setup_logger(name=__name__):
    """
    Configures and returns a logger instance that writes strictly to a file.
    No output will appear in the console.
    """

    if not os.path.exists(LOG_DIR):
        try:
            os.makedirs(LOG_DIR)
        except OSError as e:
            print(f"Critical Error: Could not create log directory. {e}")
            sys.exit(1)

    log_path = os.path.join(LOG_DIR, LOG_FILE)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        
        logger.propagate = False

    return logger