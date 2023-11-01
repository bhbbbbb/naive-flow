import logging
from .metrics import MetricsLike
from ...log import LoggingLevel

class EarlyStoppingHandler:

    best_criterion: MetricsLike
    counter: int

    def __init__(self, early_stopping_rounds: int):
        self.best_criterion = None
        self.counter = 0
        self.best_epoch = -1
        self.best_name = None
        self.early_stopping_rounds = early_stopping_rounds
        self.logger = logging.getLogger(__name__)
        return
    
    def update(self, name: str, new_criterion: MetricsLike, epoch: int):

        if self.best_criterion is not None and not new_criterion.better_than(self.best_criterion):
            self.counter += 1
            return

        self.best_criterion = new_criterion
        self.best_epoch = epoch
        self.best_name = name
        self.counter = 0
        return
    
    def is_best_epoch(self, epoch: int) -> bool:
        return self.best_epoch >= 0 and self.best_epoch == epoch

    def should_stop(self, epoch: int) -> bool:

        if epoch == self.best_epoch:
            self._print_best_criterion("New")
            return False

        threshold = self.early_stopping_rounds or "infinity"
        self.logger.log(
            LoggingLevel.EARLY_STOPPING_PROGRESS,
            "Early stopping counter:" f"{self.counter} / {threshold}"
        )
        
        self._print_best_criterion("Current")
        return (
            self.early_stopping_rounds > 0
            and self.counter == self.early_stopping_rounds
        )

    def _print_best_criterion(self, new_or_current: str):
        if self.best_criterion is not None:
            self.logger.log(
                LoggingLevel.EARLY_STOPPING_PROGRESS,
                f"{new_or_current} best {self.best_name}@epoch{self.best_epoch}: "
                f"{self.best_criterion}"
            )
        return
