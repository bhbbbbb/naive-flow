from typing import Union
from pydantic import field_validator, NonNegativeInt
from ..base.config import BaseConfig

class TrackerConfig(BaseConfig):

    log_root_dir: str = "runs"
    """dir for saving checkpoints and log files"""

    epochs_per_checkpoint: NonNegativeInt = 0
    """num of epochs per checkpoints

        Example:
            1: stand for save model every epoch
            0: for not save until finish
    """

    enable_logging: bool = True
    """whether log to file "log.log". It"s useful to turn this off when inference on kaggle"""

    early_stopping_rounds: NonNegativeInt = 0
    """Early stopping threshold. If early_stopping_rounds == 0, then early stopping would
    not be enable. I.e. model would train until the specified epoch
    """

    save_n_best: NonNegativeInt = 1
    """only save n latest models with best validation scorea.

    If set to 0, no checkpoint would be saved due to its validation score. (still some
    checkpoints might be saved due to `epochs_per_checkpoint > 0`)

    If set to 1, every better model would be saved, and the previously best-saved models would
    get deleted.
    """

    comment: Union[str, None] = None
    """Same as the argument of SummaryWriter"""

    @field_validator(
        "early_stopping_rounds", "save_n_best", "epochs_per_checkpoint",
        mode="after"
    )
    @classmethod
    def check_non_negative_int(cls, v: int):
        if v < 0:
            raise ValueError(f"should be non negative integer but got {v}.")
        return v
