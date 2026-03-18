import logging
import os
from datetime import datetime


def setup_logging() -> None:
    """Configure the MLOps pipeline logging system.

    Sets up a logger named "mlops_pipeline" with both file and console handlers.
    Log files are stored in the "logs" directory with timestamp-based filenames.
    The log format includes timestamp, log level, logger name, and message.

    Returns:
        None
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".log"
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    logger.addHandler(logging.FileHandler(log_path))
    logger.addHandler(logging.StreamHandler())

    for h in logger.handlers:
        h.setFormatter(formatter)