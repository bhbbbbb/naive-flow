from ..base.early_stopping_handler import EarlyStoppingConfig

class ModelUtilsConfig(EarlyStoppingConfig):

    device: str
    """Device to use, cpu or gpu"""

    epochs_per_checkpoint: int
    """num of epochs per checkpoints

        Example:
            1: stand for save model every epoch
            0: for not save until finish
    """

    log_dir: str
    """dir for saving checkpoints and log files"""

    logging: bool
    """whether log to file 'log.log'. It's useful to turn this off when inference on kaggle"""

    epochs_per_eval: int
    """Number of epochs per evalution"""

    # ----------- Early Stoping Config -----------------------
    early_stopping: bool
    """whether enable early stopping"""

    early_stopping_threshold: int
    """Threshold for early stopping mode. Only matter when EARLY_STOPPING is set to True."""

    save_best: bool
    """set True to save every time when the model reach best valid score."""
