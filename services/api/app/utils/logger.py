import logging
import os

def setup_logging():
  logger = logging.getLogger("app")
  logger.setLevel(logging.INFO)

  if logger.handlers:
      return  

  formatter = logging.Formatter(
      "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
  )

  logger.addHandler(logging.StreamHandler())

  for h in logger.handlers:
      h.setFormatter(formatter)