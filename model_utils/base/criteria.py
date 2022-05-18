from __future__ import annotations
from typing import List, Union, TypeVar
from .config import NamespaceDict


class PlotConfig(NamespaceDict):
    short_name: str
    full_name: str

    plot: bool
    """Whether plot this criterion"""

    default_lower_limit_for_plot: float
    default_upper_limit_for_plot: float

class _Criterion:

    short_name: str
    full_name: str

    plot: bool
    """Whether plot this criterion"""

    default_lower_limit_for_plot: float
    default_upper_limit_for_plot: float

    value: float
    primary: bool
    """whether be used as primary criterion"""

    def __init__(self, value: float):
        self.value = value
        return
    
    def better_than(self, rhs: _Criterion) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self.value)

class Loss(_Criterion):

    short_name: str = "loss"
    full_name: str = "Loss"
    
    plot: bool = True
    """Whether plot this criterion"""

    default_lower_limit_for_plot: float = None
    default_upper_limit_for_plot: float = None

    primary: bool = True

    def __init__(self, value: float):
        super().__init__(value)
        return
    
    def better_than(self, rhs: Loss):
        return self.value < rhs.value

    def __str__(self) -> str:
        if self.value < 1e-4 or self.value > 1e6:
            return f"{self.value:.6e}"
        return f"{self.value:.6f}"
    

class Accuarcy(_Criterion):

    short_name: str = "acc"
    full_name: str = "Accuracy"
    plot: bool = True

    default_lower_limit_for_plot: float = 0.0
    default_upper_limit_for_plot: float = 1.0

    primary: bool = False
    
    def __init__(self, value: float):
        super().__init__(value)
        return
    
    def better_than(self, rhs: Accuarcy):
        return self.value > rhs.value

    def __str__(self):
        return f"{self.value * 100:.4f} %"

_CriterionT = TypeVar("_CriterionT", bound=_Criterion)

class Criteria:
    """Container of Criterion"""

    _data: List[_Criterion]
    primary_criterion: _Criterion
    __registered_criteria: List[_Criterion] = []

    def __init__(self, *criteria: Union[_Criterion, float]):
        if len(criteria) == 1 and isinstance(criteria[0], float):
            criteria = [Loss(criteria[0])]
        
        is_got_primary = False
        self._data = criteria
        for criterion in criteria:

            if criterion.primary:
                assert is_got_primary is False, (
                    f"Got both '{self.primary_criterion.short_name}' and "
                    f"'{criterion.short_name}' set primary to True,"
                    "but expected num of primary <= 1."
                )
                self.primary_criterion = criterion
                is_got_primary = True
        
        return

    def better_than(self, rhs: Union[Criteria, None]):
        if rhs is not None:
            return self.primary_criterion.better_than(rhs.primary_criterion)
        return True
    
    def __iter__(self):
        for criterion in self._data:
            yield criterion.short_name, criterion.value

    def asdict(self):
        return dict(self)

    def display(self):
        for criterion in self._data:
            print(f"{criterion.full_name}:  {criterion}", end="\t")
        print("")
        return
    
    def get_plot_configs(self) -> dict[str, PlotConfig]:
        configs = {}
        for criterion in self._data:
            configs[criterion.short_name] = PlotConfig(
                short_name=criterion.short_name,
                full_name=criterion.full_name,
                plot=criterion.plot,
                default_lower_limit_for_plot=criterion.default_lower_limit_for_plot,
                default_upper_limit_for_plot=criterion.default_upper_limit_for_plot,
            )
        return configs

    @staticmethod
    def register_criterion(
        *,
        short_name: str = None,
        full_name: str = None,
        plot: bool = None,
        default_lower_limit_for_plot: float = None,
        default_upper_limit_for_plot: float = None,
        primary: bool = None,
    ):

        # if criterion is not None:
        #     Criteria.__registered_criteria.append(criterion)
        #     return criterion
        
        def wrapper(_criterion: type[_CriterionT]) -> type[_CriterionT]:
            NewCriterion = type("NewCriterion", (_criterion, ), {})

            if short_name:
                NewCriterion.short_name = short_name
            if full_name:
                NewCriterion.full_name = full_name
            if plot:
                NewCriterion.plot = plot
            if default_lower_limit_for_plot:
                NewCriterion.default_lower_limit_for_plot = default_lower_limit_for_plot
            if default_upper_limit_for_plot:
                NewCriterion.default_upper_limit_for_plot = default_upper_limit_for_plot
            if primary:
                NewCriterion.primary = primary
            Criteria.__registered_criteria.append(NewCriterion)
            return NewCriterion
        
        return wrapper
    
    @staticmethod
    def get_plot_configs_from_registered_criterion() -> dict[str, PlotConfig]:
        configs = {}
        for criterion in Criteria.__registered_criteria:
            configs[criterion.short_name] = PlotConfig(
                short_name=criterion.short_name,
                full_name=criterion.full_name,
                plot=criterion.plot,
                default_lower_limit_for_plot=criterion.default_lower_limit_for_plot,
                default_upper_limit_for_plot=criterion.default_upper_limit_for_plot,
            )
        return configs
    