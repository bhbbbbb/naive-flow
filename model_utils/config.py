from typing import Callable
from torch.optim import Optimizer
from ..config import BaseConfig, UNIMPLEMENTED
class ModelUtilsConfig(BaseConfig):
    """
        for optimizer:
            Examples:
                >>> lambda params, config: Adam(params, lr=config.learing_rate)
    """

    # device
    device = UNIMPLEMENTED

    optimizer: Callable[[any, BaseConfig], Optimizer] = UNIMPLEMENTED

    # this is for display and log
    _optimizer_name: str = UNIMPLEMENTED

    learning_rate: float = UNIMPLEMENTED


    # num of epochs per checkpoints
    # e.g. 1 stand for save model every epoch
    #      0 for not save until finish
    epochs_per_checkpoint: int = UNIMPLEMENTED

    # dir for saving checkpoints and log files
    log_dir: str = UNIMPLEMENTED

    early_stopping: bool = UNIMPLEMENTED

    # only matter when EARLY_STOPPING is set to True
    early_stopping_threshold: int = UNIMPLEMENTED

    num_class: int = UNIMPLEMENTED
