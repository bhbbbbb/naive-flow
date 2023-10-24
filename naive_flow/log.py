from enum import IntEnum
import logging

class LoggingLevel(IntEnum):

    ON_CHECKPOINT_LOAD: int = logging.INFO + 5
    ON_CHECKPOINT_PURGE: int = logging.INFO + 3
    ON_CHECKPOINT_SAVE: int = logging.INFO + 3
    EARLY_STOPPING_PROGRESS: int = logging.INFO + 2
    """
    1. Early stopping counter {i}/{rounds}
    2. Current best {metrics}@epoch{i}: {v}
    """
    TRAINING_PROGRESS: int = logging.INFO + 1
    """
    1. Epoch {epoch}/{epochs}
    2. Early Stopping!
    2. Training is finish ...
    """
    ON_SCALAR_ADD: int = logging.INFO
