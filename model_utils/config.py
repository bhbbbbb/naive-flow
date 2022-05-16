from .base.config import UNIMPLEMENTED
from .base.early_stopping_handler import EarlyStoppingConfig
class ModelUtilsConfig(EarlyStoppingConfig):

    device = UNIMPLEMENTED
    """Device to use, cpu or gpu"""

    learning_rate: float = UNIMPLEMENTED

    epochs_per_checkpoint: int = UNIMPLEMENTED
    """num of epochs per checkpoints

        Example:
            1: stand for save model every epoch
            0: for not save until finish
    """

    log_dir: str = UNIMPLEMENTED
    """dir for saving checkpoints and log files"""

    logging: bool = UNIMPLEMENTED
    """whether log to file 'log.log'. It's useful to turn this off when inference"""

    epochs_per_eval: int = UNIMPLEMENTED
    """Number of epochs per evalution"""

    # ----------- Early Stoping Config -----------------------
    early_stopping: bool = UNIMPLEMENTED
    """whether enable early stopping"""

    early_stopping_threshold: int = UNIMPLEMENTED
    """Threshold for early stopping mode. Only matter when EARLY_STOPPING is set to True."""

    save_best: bool = UNIMPLEMENTED
    """set True to save every time when the model reach best valid score."""
