from ..config import BaseConfig, UNIMPLEMENTED
class ModelUtilsConfig(BaseConfig):

    device = UNIMPLEMENTED
    """Device to use, cpu or gpu"""

    learning_rate: float = UNIMPLEMENTED

    epochs_per_checkpoint: int = UNIMPLEMENTED
    """num of epochs per checkpoints

        Example:
            1: stand for save model every epoch
            0: for not save until finish
    """

    save_best: bool = UNIMPLEMENTED
    """set True to save every time when the model reach highest val_acc"""

    log_dir: str = UNIMPLEMENTED
    """dir for saving checkpoints and log files"""

    early_stopping: bool = UNIMPLEMENTED
    """whether enable early stopping"""

    early_stopping_threshold: int = UNIMPLEMENTED
    """Threshold for early stopping mode. Only matter when EARLY_STOPPING is set to True."""

    num_class: int = UNIMPLEMENTED
    """number of classes"""
