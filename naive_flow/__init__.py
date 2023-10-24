import logging
from .config import BaseConfig
from .config.utils import strfconfig
from .log import LoggingLevel
from . import tracker


if not (logger := logging.getLogger(__name__)).hasHandlers():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

def set_stream_logging_level(level: int):
    stream_handler.setLevel(level)
    return
