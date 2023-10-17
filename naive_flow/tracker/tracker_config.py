from typing import Union
from pydantic import field_validator, NonNegativeInt
from ..config import BaseConfig

class TrackerConfig(BaseConfig):

    log_root_dir: str = "runs"
    """
    `log_root_dir` for logs

    ```text
    LOG_ROOT_DIR
    ├── LOG_DIR_1
    └── LOG_DIR_2
    ```
    Here is an example of set `runs` as `log_root_dir`, Tracker will create `log_dir` following
    the naming rule (DATETIME_HOSTNAME_COMMENT) and those created `log_dir` will
    be put in `log_root_dir`. Note that here DATETIME and HOSTNAME will be
    generated in runtime automatically. While the COMMENT can be set by user in the
    `tracker_config__comment` field.

    ```text
    runs
    ├── Oct14_23-34-08_user_HGT_IMDB
    │   ├── events.out.tfevents.1697297648.user.1838590.0
    │   ├── log.log
    │   ├── Oct14_23-34-54_epoch_51_best
    │   └── Oct14_23-35-14_epoch_81_end
    ├── Oct15_00-17-18_user_HGT_acm
    │   ├── events.out.tfevents.1697300238.user.1861603.0
    │   ├── log.log
    │   ├── Oct15_00-19-14_epoch_74_best
    │   └── Oct15_00-20-01_epoch_104_end
    └── Oct15_00-21-21_user_HGT_dblp
        ├── events.out.tfevents.1697300481.user.1863785.0
        ├── log.log
        ├── Oct15_00-21-45_epoch_48_best
        └── Oct15_00-21-59_epoch_78_end
    ```
    """

    epochs_per_checkpoint: NonNegativeInt = 0
    """Num of epochs per checkpoints.
       
       By setting this field with positive integer, Tracker will save the checkpoint regularly
       
       E.g.:
            epochs_per_checkpoint == 1: Tracker will save a checkpoint for every epoch
            epochs_per_checkpoint == 0: Tracker will not save checkpoint regularly. Note that
                                        Tracker will still save checkpoint at the end of training
                                        and may save some checkpoints if `save_n_best` is set with 
                                        poistive integer.
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
