from .base.config import BaseConfig, UNIMPLEMENTED, register_checking_hook
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

    logging: bool = UNIMPLEMENTED
    """whether log to log.log. It's useful to turn this off when inference"""

    early_stopping: bool = UNIMPLEMENTED
    """whether enable early stopping"""

    early_stopping_threshold: int = UNIMPLEMENTED
    """Threshold for early stopping mode. Only matter when EARLY_STOPPING is set to True."""

    early_stopping_by_acc: bool = UNIMPLEMENTED
    """
    Early stopping with valid_acc as criterion, cannot be true if enable_accuracy is False.
    Turn off to use valid_loss as criterion.
    """

    enable_accuracy: bool = UNIMPLEMENTED
    """Whether enable logging accuracy in history. Turn off to use loss only."""

    @register_checking_hook
    def check_acc(self):
        if self.early_stopping_by_acc:
            assert self.enable_accuracy,\
                "have to set enable_accuracy to True to enable early_stopping_by_acc"
    