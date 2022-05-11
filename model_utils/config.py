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
    """whether log to log.log. It's useful to turn this off when inference"""

    epochs_per_eval: int = UNIMPLEMENTED
    """Number of epochs per evalution"""
