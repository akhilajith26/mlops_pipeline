import logging


def setup_logger(name: str) -> logging.Logger:
    """Set up a logger for the given name."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler("api.log")
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add formatter to handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
