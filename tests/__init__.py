import logging
import os

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", logging.getLevelName(logging.INFO)).upper(),
    format="[%(asctime)s][%(levelname)s][%(module)s][%(funcName)s][Ln %(lineno)s] - %(message)s",
)
