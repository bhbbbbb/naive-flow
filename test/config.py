from model_utils import ModelUtilsConfig

class TestModelUtilsConfig(ModelUtilsConfig):

    device = "cpu"
    """Device to use, cpu or gpu"""

    learning_rate: float = 0.1

    epochs_per_checkpoint: int = 0
    """num of epochs per checkpoints

        Example:
            1: stand for save model every epoch
            0: for not save until finish
    """

    log_dir: str = "log"
    """dir for saving checkpoints and log files"""

    logging: bool = True
    """whether log to log.log. It's useful to turn this off when inference"""

    epochs_per_eval: int = 1
    """Number of epochs per evalution"""

    # ----------- Early Stoping Config -----------------------
    early_stopping: bool = False
    """whether enable early stopping"""

    early_stopping_threshold: int = 0
    """Threshold for early stopping mode. Only matter when EARLY_STOPPING is set to True."""

    save_best: bool = False
    """set True to save every time when the model reach best valid score."""
