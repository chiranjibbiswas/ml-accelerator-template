import logging
import sys

def get_logger(name:str):

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create console handler with stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # Create formatter and add it to the handler
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Avoid duplicate logs if handlers already exist
    if not logger.handlers:
        logger.addHandler(handler)

    return logger