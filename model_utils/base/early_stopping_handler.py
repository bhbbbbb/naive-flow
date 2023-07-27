from .criteria import Criteria

class EarlyStoppingHandler:

    best_criteria: Criteria
    counter: int

    # def __init__(self, config: EarlyStoppingConfig):
    def __init__(self, early_stopping_rounds: int):
        self.best_criteria = None
        self.counter = 0
        self.early_stopping_rounds = early_stopping_rounds
        return
    
    def update(self, new_criteria: Criteria):
        if new_criteria is None:
            return

        if not new_criteria.better_than(self.best_criteria):
            self.counter += 1
            return

        self.best_criteria = new_criteria
        self.counter = 0
        return


    def should_stop(self, new_criteria: Criteria) -> bool:
        if new_criteria is None:
            return False

        if new_criteria == self.best_criteria: # ref. equal
            self._print_best_criterion()
            return False

        threshold = self.early_stopping_rounds or "infinity"
        print("Early stopping counter:" f"{self.counter} / {threshold}")
        
        self._print_best_criterion()
        return (
            self.early_stopping_rounds > 0
            and self.counter == self.early_stopping_rounds
        )

    def _print_best_criterion(self):
        best_criterion = self.best_criteria.primary_criterion
        print(f"Current best {best_criterion.config.full_name}: {best_criterion}")
        return
    
    def is_best(self, new_criteria: Criteria):
        return (
            self.counter == 0 and id(self.best_criteria) == id(new_criteria)
            and self.best_criteria is not None
        )
