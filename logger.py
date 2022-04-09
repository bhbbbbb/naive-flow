import logging
import os
from enum import Enum, auto

class Mode(Enum):
    CONSOLE = auto()
    FILE = auto()
class _Logger:

    FORMAT = (
        "%(asctime)s %(levelname)s\nFile \"%(pathname)s\", at line %(lineno)d, "
        "in %(module)s, %(funcName)s \n"
        "%(message)s" "\n"
    )

    mode: Mode
    logger: logging.Logger

    def __init__(self, log_file_root: str = None):
        """
        Args:
            log_file_root (str, optional): pass None or ignore to use console mode.
        """

        if log_file_root is not None:
            self.mode = Mode.FILE
            self.log_file_path = os.path.join(log_file_root, "log.log")
            self.logger = logging.getLogger("file")
            handler = logging.FileHandler(self.log_file_path, "a", encoding="utf-8")
            handler.setFormatter(logging.Formatter(fmt=_Logger.FORMAT))
        else:
            self.mode = Mode.CONSOLE
            self.logger = logging.getLogger("console")
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(fmt="%(message)s"))

        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log(self, msg: str):
        self.logger.info(msg)
        return
    
    def write(self, msg: str):
        self.logger.info(msg)
        return
    
    def warning(self, msg: str):
        self.logger.warning(msg)
        return

    


class Logger:

    def __init__(self, log_file_root: str):
        self._c_logger = _Logger()
        self._f_logger = _Logger(log_file_root)
        return

    @property
    def file(self):
        return self._f_logger

    @property
    def console(self):
        return self._c_logger

    def log(self, msg: str):
        self._c_logger.log(msg)
        self._f_logger.log(msg)
        return
    
    def write(self, msg: str):
        self.log(msg)
        return
    
    def warning(self, msg: str):
        self._c_logger.warning(msg)
        self._f_logger.warning(msg)
        return
