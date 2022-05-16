from .criteria import Criteria
from .config import BaseConfig, UNIMPLEMENTED

class EarlyStoppingConfig(BaseConfig):

    early_stopping: bool = UNIMPLEMENTED
    """whether enable early stopping"""

    early_stopping_threshold: int = UNIMPLEMENTED
    """Threshold for early stopping mode. Only matter when EARLY_STOPPING is set to True."""

    save_best: bool = UNIMPLEMENTED
    """set True to save every time when the model reach best valid score."""

    
class EarlyStoppingHandler:

    best_criteria: Criteria
    counter: int
    _best_saved: bool

    def __init__(self, config: EarlyStoppingConfig):
        self.best_criteria = None
        self.counter = 0
        self.config = config
        self._best_saved = False
        return
    
    def should_stop(self, new_criteria: Criteria) -> bool:
        if new_criteria is None:
            return False
        if not new_criteria.better_than(self.best_criteria):
            self.counter += 1
            threshold = self.config.early_stopping_threshold\
                            if self.config.early_stopping else "infinity"
            print("Early stopping counter:" f"{self.counter} / {threshold}")
            
            self._print_best_criterion()
            return (
                self.config.early_stopping
                and self.counter == self.config.early_stopping_threshold
            )

        self.best_criteria = new_criteria
        self.counter = 0
        self._best_saved = False
        self._print_best_criterion()
        return False

    def _print_best_criterion(self):
        best_criterion = self.best_criteria.primary_criterion
        print(f"Current best {best_criterion.full_name}: {best_criterion}")
        return
    
    def should_save_best(self):
        if (
            self.config.save_best and self.counter == 0 and not self._best_saved
            and self.best_criteria is not None
        ):
            self._best_saved = True
            return True
        return False
