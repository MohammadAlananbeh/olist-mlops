
import logging
from pathlib import Path


def setup_logger(
    name: str = "olist_mlops",
    log_file: str = "logs/app.log",
    level: int = logging.INFO,
) -> logging.Logger:

    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(
        log_path,
        encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger



# -------------------------------------------------

# Then use it anywhere:
    # from utils.logger import setup_logger

    # logger = setup_logger()

    # logger.info("Application started")
# Instead of:
    # print("Application started")
