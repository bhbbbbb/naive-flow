import logging
import os
import sys
from typing import Dict
import contextlib
from typing_extensions import deprecated
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile

@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err

class RangeTqdm(tqdm):
    def __init__(self, start: int, end: int, **kwargs):
        self.start = start
        self.end = end
        super().__init__(
            range(start, end),
            **kwargs,
            bar_format=(
                "Epoch: {cur}/{total_epochs}, {desc}"
                "|{bar}| "
                "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            ),
        )
        return
    @property
    def format_dict(self):
        d = super().format_dict
        d.update({"total_epochs": self.end, "cur": self.start + self.n})
        return d


class _ScalarTqdm(tqdm):

    def __init__(self, **kwargs):
        super().__init__(bar_format="{desc}", **kwargs)
        return

class ScalarTqdms:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tqdms: Dict[str, _ScalarTqdm] = {}
        return
    
    def update(self, tag: str, msg: str):
        if tag not in self.tqdms:
            self.tqdms[tag] = _ScalarTqdm(**self.kwargs)

        self.tqdms[tag].set_description_str(msg)
        return
    
    def close(self):
        for scalar_tqdm in self.tqdms.values():
            scalar_tqdm.clear()
            scalar_tqdm.close()
        return
    
    def __del__(self):
        self.close()
        return


class _NullFile:
    """A dummy blackhole file-like object which acts like /dev/null"""
    def write(self, s: str): # pylint: disable=unused-argument
        return

    def flush(self):
        return
    
    def close(self):
        return

class StdoutLogFile:

    def __init__(self, log_dir: str = None):
        """
        Args:
            log_file_root (str, optional): pass None or ignore to use stdout only.
        """
        if log_dir is not None:
            log_file_path = os.path.join(log_dir, "log.log")
            self.log = open(log_file_path, "a", encoding="utf-8")
        else:
            self.log = _NullFile()
        return

    def __del__(self):
        self.log.close()
        return

    def write(self, s: str):
        self.log.write(s)
        sys.stdout.write(s)
        return

    def flush(self):
        self.log.flush()
        sys.stdout.flush()
        return
    
    def close(self):
        self.log.close()
        return

FORMAT = (
    "%(asctime)s %(levelname)s\n"
    "%(message)s"
    "\n"
)

@deprecated("")
def del_file_handler(log_dir: str):
    name = __name__.rsplit(".", 3)[0]
    logger = logging.getLogger(name)
    for handler in logger.handlers:
        if handler.get_name() == log_dir:
            logger.removeHandler(handler)

@deprecated("")
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
        file_handler.set_name(log_file_root)
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
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
