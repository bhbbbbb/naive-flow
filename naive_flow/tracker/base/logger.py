from typing_extensions import deprecated
import logging
import os

FORMAT = (
    "%(asctime)s %(levelname)s\n"
    "%(message)s"
    "\n"
)
# FORMAT = (
#     "%(asctime)s %(levelname)s\nFile \"%(pathname)s\", at line %(lineno)d, "
#     "in %(module)s, %(funcName)s \n"
#     "%(message)s" "\n"
# )

def set_handlers(log_file_root: str = None):
    """
    Args:
        log_file_root (str, optional): pass None or ignore to use console handler only (without
            file handler).
    """

    name = __name__.rsplit(".", 3)[0]
    logger = logging.getLogger(name)
    if logger.level < logging.DEBUG:
        logger.setLevel(logging.DEBUG)
    if log_file_root is not None:
        log_file_path = os.path.join(log_file_root, "log.log")
        file_handler = logging.FileHandler(log_file_path, "a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt=FORMAT))
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    
    # stream_handler = logging.StreamHandler()
    # stream_handler.setLevel(logging.INFO)
    # logger.addHandler(stream_handler)
    return

@deprecated("")
def get_logger(name: str = None, log_file_root: str = None):
    """
    Args:
        log_file_root (str, optional): pass None or ignore to use console handler only (without
            file handler).
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    if log_file_root is not None:
        log_file_path = os.path.join(log_file_root, "log.log")
        file_handler = logging.FileHandler(log_file_path, "a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt=FORMAT))
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    return logger
