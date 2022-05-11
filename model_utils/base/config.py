from io import StringIO
from typing import Callable, List, TypeVar, Any, Union
from .writable import Writable
from .namespace_dict import NamespaceDict

class _Unimplemented:
    def __init__(self, msg: str = ""):
        self.msg = msg
        return
    

UNIMPLEMENTED = _Unimplemented()
"""`NotImplementedError` would raise when someone try to reach this constant"""

PropertyGetterT = TypeVar("PropertyGetterT", bound=Callable[[Any], Any])
PropertySetterT = TypeVar("PropertySetterT", bound=Callable[[Any, Any], None])
MemberSetterT = TypeVar("MemberSetterT", bound=Callable[[Any, Any], Any])
FunctionSetterT = TypeVar("FunctionSetterT", bound=Callable[[Any], Any])


class Mutable(property):
    
    fget: Callable[[Any], Any]
    fset: Callable[[Any, Any], None]
    fdel: Callable[[Any], None]
    _data: list = []

    _idx: int

    def __init__(
        self,
        getter_or_default_val: Union[PropertyGetterT, Any],
        setter: Union[PropertySetterT, MemberSetterT, FunctionSetterT] = None,
        fdel = None,
        doc = None,
        _idx: int = None,
    ):
        self._idx = _idx
        getter_ = getter_or_default_val
        if _idx is None:
            _idx = len(self._data)
            self._idx = _idx
            if not callable(getter_or_default_val):
                # call as default value, rather than property getter
                self._data.append(getter_or_default_val)
                def default_getter(_):
                    return self._data[_idx]
                getter_ = default_getter
            else:
                self._data.append(_Unimplemented(getter_.__name__))
        
        setter_ = None
        if not isinstance(self._data[_idx], _Unimplemented):
            if setter is None:
                def default_setter(_, val):
                    self._data[_idx] = val
                    return
            elif setter.__code__.co_argcount == 1:
                setter: FunctionSetterT
                def default_setter(_, val):
                    self._data[_idx] = setter(val)
                    return
            elif setter.__code__.co_argcount == 2:
                setter: MemberSetterT
                def default_setter(obj, val):
                    self._data[_idx] = setter(obj, val)
                    return
            else:
                raise Exception(
                    "count of arguments for setter is supposed to be exactly 1 or 2.",
                )
            setter_ = default_setter
        elif setter is not None:
            self._data[_idx] = True
            setter_ = setter

        super().__init__(getter_, setter_, fdel, doc)
        return
    
    def setter(self, setter: Union[PropertySetterT, MemberSetterT, FunctionSetterT]):
        return Mutable(
            self.fget,
            setter,
            self.fdel,
            self.__doc__,
            self._idx,
        )

    @staticmethod
    def check_mutable_implementation():
        for data in Mutable._data:
            if isinstance(data, _Unimplemented):
                raise NotImplementedError(
                    f"property '{data.msg}' is mark as Mutable, but without setter implemented."
                )
class BaseConfig(NamespaceDict):
    """simple wrap of NamespaceDict with utils for config"""

    __checking_hooks__: List[Callable] = []
    __checked__: bool = False
    __immutable__: bool = False
    
    def __getattribute__(self, __name: str):
        tem = super().__getattribute__(__name)

        if isinstance(tem, _Unimplemented):
            raise NotImplementedError(f"attribute: '{__name}' should be implemented")
        return tem

    def __setattr__(self, __name: str, __value: Any) -> None:
        if not self.__immutable__ or self._is_builtin_name(__name):
            super().__setattr__(__name, __value)
            return
        
        if not hasattr(self, __name):
            raise AttributeError(
                f"attribute '{__name}' is not an attribute of object '{self.__class__.__name__}'."
            )

        obj = getattr(type(self), __name)
        if isinstance(obj, Mutable):
            obj.__set__(self, __value)
            return

        raise AttributeError(f"attribute '{__name}' is immutable.")

    def _check(self):
        for func in self.__checking_hooks__:
            if isinstance(func, staticmethod):
                func.__func__()
            elif func.__code__.co_argcount == 1:
                func(self)
            elif func.__code__.co_argcount == 0:
                func()
            else:
                raise Exception(
                    "except 0 or 1 arguments of checking function, "
                    f"but {func.__code__.co_argcount} was given"
                )
        return

    def check_and_freeze(self, freeze: bool = True):
        """Run checking hooks, check implementation, and freeze.

        Args:
            freeze (bool, optional): whether freeze. Defaults to True.
        """
        self._check()
        for _, _ in dict(self).items():
            pass
        if freeze:
            self.__immutable__ = True
        self.__checked__ = True
        return
    
    def display(self, file: Writable = None, skip_check: bool = False):
        """display all the configurations

        Args:
            file (Writable, optional): file-like object. It'll be passed to the bulit-in function
                `print(file=file)` 
            forced_check (bool, optional): Run checking hook
                if (`BaseConfig.__checked` is False or `forced_check` is True). Defaults to False.
        """
        if not skip_check:
            assert self.__checked__, "run 'check_and_freeze' before 'display'."

        sio = StringIO()
        sio.write("Configuration:\n")
        for attr, value in dict(self).items():
            sio.write(f"{attr:30} {value}\n")
        sio.write("\n")
        print(sio.getvalue(), file=file, end="")
        sio.close()
        return
    
def register_checking_hook(func):
    BaseConfig.__checking_hooks__.append(func)
    return func

register_checking_hook(Mutable.check_mutable_implementation)
