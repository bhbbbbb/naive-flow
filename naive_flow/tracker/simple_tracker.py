from typing import Union, Callable, overload

from torch import nn, optim

from .tracker_config import TrackerConfig
from .base_tracker import BaseTracker



class SimpleTracker(BaseTracker):

    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler._LRScheduler

    @overload
    def __init__(
        self,
        model,
        optimizer,
        scheduler = None,
        config: TrackerConfig = None,
        from_checkpoint: Union[str, Callable] = None,
    ):...

    @overload
    def __init__(
        self,
        model,
        optimizer,
        scheduler = None,
        log_root_dir: str = None,
        epochs_per_checkpoint: int = None,
        early_stopping_rounds: int = None,
        save_n_best: int = None,
        comment: str = None,
        enable_logging: bool = True,
        from_checkpoint: Union[str, Callable] = None,
    ):...

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        model,
        optimizer,
        scheduler = None,
        log_root_dir: str = None,
        epochs_per_checkpoint: int = None,
        early_stopping_rounds: int = None,
        save_n_best: int = None,
        comment: str = None,
        enable_logging: bool = True,
        config: TrackerConfig = None,
        from_checkpoint: Union[str, Callable] = None,
    ):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        super().__init__(
            log_root_dir=log_root_dir,
            epochs_per_checkpoint=epochs_per_checkpoint,
            early_stopping_rounds=early_stopping_rounds,
            save_n_best=save_n_best,
            comment=comment,
            enable_logging=enable_logging,
            config=config,
            from_checkpoint=from_checkpoint,
        )
        return
    
    def load(self, checkpoint_dict: dict):
        
        self.model.load_state_dict(checkpoint_dict['model'])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint_dict['scheduler'])

        return

    def save(self) -> dict:

        scheduler_dict = self.scheduler.state_dict() if self.scheduler else {}
        
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': scheduler_dict,
        }
    