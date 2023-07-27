from pydantic import field_validator, NonNegativeInt
from ..base.config import BaseConfig

class ModelUtilsConfig(BaseConfig):

    device: str
    """Device to use, cpu or gpu"""

    epochs_per_checkpoint: NonNegativeInt
    """num of epochs per checkpoints

        Example:
            1: stand for save model every epoch
            0: for not save until finish
    """

    log_dir: str
    """dir for saving checkpoints and log files"""

    logging: bool
    """whether log to file "log.log". It"s useful to turn this off when inference on kaggle"""

    epochs_per_eval: NonNegativeInt
    """Number of epochs per evalution"""

    early_stopping_rounds: NonNegativeInt
    """Early stopping threshold. If early_stopping_rounds == 0, then early stopping would
    not be enable. I.e. model would train until the specified epoch
    """

    save_n_best: NonNegativeInt
    """only save n latest models with best validation scorea.

    If set to 0, no checkpoint would be saved due to its validation score. (still some
    checkpoints might be saved due to `epochs_per_checkpoint > 0`)

    If set to 1, every better model would be saved, and the previously best-saved models would
    get deleted.
    """


    @field_validator(
        "early_stopping_rounds", "save_n_best", "epochs_per_eval", "epochs_per_checkpoint",
        mode="after"
    )
    @classmethod
    def check_non_negative_int(cls, v: int):
        if v < 0:
            raise ValueError(f"should be non negative integer but got {v}.")
        return v
