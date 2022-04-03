import logging
import os

class Logger:

    FORMAT = (
        "%(asctime)s %(levelname)s\nFile \"%(pathname)s\", at line %(lineno)d, "
        "in %(module)s, %(funcName)s \n"
        "\t%(message)s" "\n"
    )

    def __init__(self, log_file_root: str):
        self.log_file_path = os.path.join(log_file_root, "log.log")
        self.logger = logging.getLogger()
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(fmt="%(message)s"))

        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        file_handler = Logger._create_file_handler(self.log_file_path)
        self.logger.addHandler(file_handler)
        return

    @staticmethod
    def _create_file_handler(file_path: str):
        file_handler = logging.FileHandler(file_path, "a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt=Logger.FORMAT))
        return file_handler
    
    def reset_log_file_root(self, new_log_file_root: str):
        self.log_file_path = os.path.join(new_log_file_root, "log.log")
        self.logger.handlers.pop()
        new_handler = Logger._create_file_handler(self.log_file_path)
        self.logger.addHandler(new_handler)
        return

    def log(self, msg: str):
        self.logger.info(msg)
        return
    
    def write(self, msg: str):
        self.log(msg)
        return
    
    def warning(self, msg: str):
        self.logger.warning(msg)
        return
