import logging
import os
from datetime import datetime

def setup_logging():
  log_dir = "logs"
  os.makedirs(log_dir, exist_ok=True)

  log_file = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".log"
  log_path = os.path.join(log_dir, log_file)

  logger = logging.getLogger("mlops_pipeline")
  logger.setLevel(logging.INFO)

  if logger.handlers:
      return  

  formatter = logging.Formatter(
      "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
  )

  logger.addHandler(logging.FileHandler(log_path))
  logger.addHandler(logging.StreamHandler())

  for h in logger.handlers:
      h.setFormatter(formatter)