from __future__ import annotations
from typing import (
    Union,
    ClassVar,
    Optional,
    Dict,
    Type,
    overload,
    Callable,
    Protocol,
    Iterable,
    TypedDict,
)
from collections.abc import Sequence
from functools import cached_property
from typing_extensions import Unpack

from pydantic import RootModel, BaseModel, ConfigDict


class CriterionConfigFields(TypedDict, total=False):
    """for type hints using `Unpack`"""
    name: str
    primary: bool
    full_name: str
    plot: bool
    default_lower_limit_for_plot: Optional[float]
    default_upper_limit_for_plot: Optional[float]

class CriterionConfig(BaseModel):

    model_config = ConfigDict(validate_assignment=True)

    name: str
    """Name which has to be unique, serves like key of dict"""

    primary: bool
    """whether be used as primary criterion"""

    full_name: str
    """Full Name used to display"""

    plot: bool
    """Whether plot this criterion"""

    default_lower_limit_for_plot: Optional[float]
    default_upper_limit_for_plot: Optional[float]

    def update(self, /, **kwargs: Unpack[CriterionConfigFields]):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

# class MergeInfoMeta(type):
#     def __new__(cls, name, bases, attrs: dict):
#         info = {}

#         for base in bases:
#             info.update(getattr(base, "_info", {}))
        
#         info.update(attrs.get("_info", {}))


#         attrs["_info"] = info

#         return super().__new__(cls, name, bases, attrs)
        
            
class CriterionLike(Protocol):
    config: ClassVar[CriterionConfig]
    """static: describes information of this criterion"""

    root: float

    def better_than(self, rhs: _Criterion) -> bool:...

    def __repr__(self) -> str:...

    def __str__(self) -> str:...
    
    @classmethod
    def config_copy(cls):...

    @classmethod
    def is_primary(cls) -> bool:...
    
    @classmethod
    def name(cls) -> str:...

    @property
    def value(self) -> float:...

class _Criterion(RootModel[float]):

    config: ClassVar[CriterionConfig]
    """static: describes information of this criterion"""

    root: float

    def better_than(self, rhs: _Criterion) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.config.name}: {self.root}"

    def __str__(self) -> str:
        return f"{self.config.full_name}: {self.root}"
    
    @classmethod
    def config_copy(cls):
        return cls.config.model_copy()

    @classmethod
    def is_primary(cls) -> bool:
        return cls.config.primary
    
    @classmethod
    def name(cls) -> str:
        return cls.config.name

    @property
    def value(self) -> float:
        return self.root

class Loss(_Criterion):
    """Default Loss Critierion

    Info:
        primary (bool): True.
        name (str): loss
        full_name (str): Loss
        plot (bool): True
        default_lower_limit_for_plot (float): None
        default_upper_limit_for_plot (float): None

    """

    config: ClassVar[CriterionConfig] = CriterionConfig(
        name = "loss",
        full_name = "Loss",
        plot = True,
        default_lower_limit_for_plot = None,
        default_upper_limit_for_plot = None,
        primary = True,
    )

    def better_than(self, rhs: Loss):
        return self.root < rhs.root

    def __str__(self) -> str:
        if self.root < 1e-4 or self.root > 1e6:
            return f"{self.config.full_name}:  {self.root:.6e}"
        return f"{self.config.full_name}:  {self.root:.6f}"
    

class Accuarcy(_Criterion):
    """Default Accuracy Critierion

    Attributes:
        primary (bool): False
        name (str): acc
        full_name (str): Accuracy
        plot (bool): True
        default_lower_limit_for_plot (float): 0.0
        default_upper_limit_for_plot (float): 1.0

    """

    config: ClassVar[CriterionConfig] = CriterionConfig(
        name = "acc",
        full_name = "Accuracy",
        plot = True,
        default_lower_limit_for_plot = 0.0,
        default_upper_limit_for_plot = 1.0,
        primary = False,
    )
    
    def better_than(self, rhs: Accuarcy):
        return self.root > rhs.root

    def __str__(self):
        return f"{self.config.full_name}:  {self.root * 100:.4f} %"


class Criteria(RootModel[Dict[str, _Criterion]]):
    """Container of Criterion"""

    root: Dict[str, _Criterion]
    # _primary_criterion: _Criterion
    __registered_criteria: ClassVar[Dict[str, Type[_Criterion]]] = {}

    
    @overload
    def __init__(self, *criteria: _Criterion):...

    @overload
    def __init__(self, criterion_list: Iterable[_Criterion]):...

    @overload
    def __init__(self, **key_value: float):...

    def __init__(self, *criteria: _Criterion, **key_value: float):

        assert bool(criteria) != bool(key_value), (
            "Cannot use both arguments and keywords arguments when initialization."
        )

        if bool(criteria) and isinstance(criteria[0], Sequence): # Criteria([c1, c2, ...])
            assert len(criteria) == 1
            criteria = criteria[0]

        key_criterion = (
            {criterion.name(): criterion for criterion in criteria}
            or
            {k: self.__registered_criteria[k](v) for k, v in key_value.items()}
        )

        super().__init__(**key_criterion)
        return

    @cached_property
    def primary_criterion(self):
        _primary_criterion = None
        for criterion in self.root.values():
            if criterion.is_primary():
                assert _primary_criterion is None, (
                    f"Got both '{_primary_criterion.name()}' and "
                    f"'{criterion.name()}' set primary to True,"
                    "but expected num of primary <= 1."
                )
                _primary_criterion = criterion
        
        assert _primary_criterion is not None
        return _primary_criterion

    def better_than(self, rhs: Optional[Criteria]) -> bool:
        if rhs is not None:
            assert self.primary_criterion is not None
            return self.primary_criterion.better_than(rhs.primary_criterion)
        return True
    
    __gt__ = better_than
    
    def __iter__(self):
        yield from self.root.keys()
    
    def __getitem__(self, index: str):
        return self.root[index]
    
    def items(self):
        yield from self.root.items()

    # def asdict(self):
    #     return dict(self)

    def display(self):
        for criterion in self.root.values():
            print(str(criterion), end="\t")
        print("")
        return
    
    # def get_plot_configs(self) -> dict[str, CriterionInfo]:
        # configs = {}
        # for criterion in self._data:
        #     configs[criterion.name] = CriterionInfo(
        #         name=criterion.name,
        #         full_name=criterion.full_name,
        #         plot=criterion.plot,
        #         default_lower_limit_for_plot=criterion.default_lower_limit_for_plot,
        #         default_upper_limit_for_plot=criterion.default_upper_limit_for_plot,
        #     )
        # return configs
        # return

    @overload
    @staticmethod
    def register_criterion(
        new_criterion_name: str,
        **info: Unpack[CriterionConfigFields],
    ) -> Callable[[Type[_Criterion]], Type[_Criterion]]:
        """
        Example:
        ```
        NewCriterion = @Criteria.register_criterion(
            'NewCriterion',
            name='new_name',
            primary=False,
        )(Loss)
        ```
        """

    @overload
    @staticmethod
    def register_criterion(NewCriterion: Type[_Criterion]) -> Type[_Criterion]:
        # pylint: disable=invalid-name
        """
        Example:
        ```
        @Criteria.register_criterion
        class NewLoss(Loss):
            config = Loss.config_copy().update(**new_config)
        ```
        """

    @overload
    @staticmethod
    def register_criterion() -> Callable[[Type[_Criterion]], Type[_Criterion]]:
        """
        Example:
        ```
        @Criteria.register_criterion()
        class NewLoss(Loss):
            config = Loss.config_copy().update(**new_config)
        ```
        """
    
    @staticmethod
    def register_criterion(
        NewCriterion_or_new_criterion_name: Union[Type[_Criterion], str] = None,
        **info: Unpack[CriterionConfigFields],
    ):
        # pylint: disable=invalid-name
        """
        ```
        NewCriterion = @Criteria.register_criterion(
            'NewCriterion',
            name='new_name',
            primary=False,
        )(Loss)

        ```
        ```
        @Criteria.register_criterion
        class NewLoss(Loss):
            config = Loss.config_copy().update(**new_config)
        ```
        ```
        @Criteria.register_criterion()
        class NewLoss(Loss):
            config = Loss.config_copy().update(**new_config)
        ```
        """
        def __register_new_criteria(name: str, Criterion: Type[_Criterion]):
            assert name not in Criteria.__registered_criteria, (
                f"The name '{name}' is already registerd, "
                "note that all of the name of criteria have to be unique."
            )
            Criteria.__registered_criteria[name] = Criterion
            return

        if isinstance(NewCriterion_or_new_criterion_name, str):
            def wrapper_from_base(BaseCriterion: Type[_Criterion]) -> Type[_Criterion]:

                NewCriterion = type(
                    NewCriterion_or_new_criterion_name,
                    (BaseCriterion, ),
                    {"config": BaseCriterion.config_copy().update(**info)}
                )

                __register_new_criteria(NewCriterion.name(), NewCriterion)
                return NewCriterion
            return wrapper_from_base
        
        def wrapper_add_new(NewCriterion: Type[_Criterion]) -> Type[_Criterion]:
            assert issubclass(NewCriterion, _Criterion)
            __register_new_criteria(
                NewCriterion.name(),
                NewCriterion
            )
            return NewCriterion
        
        if NewCriterion_or_new_criterion_name is None:
            return wrapper_add_new

        return wrapper_add_new(NewCriterion_or_new_criterion_name)

    # @staticmethod
    # def get_plot_configs_from_registered_criterion() -> dict[str, CriterionInfo]:
    #     configs = {}
    #     for criterion in Criteria.__registered_criteria:
    #         configs[criterion.name] = CriterionInfo(
    #             name=criterion.name,
    #             full_name=criterion.full_name,
    #             plot=criterion.plot,
    #             default_lower_limit_for_plot=criterion.default_lower_limit_for_plot,
    #             default_upper_limit_for_plot=criterion.default_upper_limit_for_plot,
    #         )
    #     return configs
