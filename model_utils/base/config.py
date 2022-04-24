from io import StringIO
from typing import Callable, List
from .writable import Writable
from .namespace_dict import NamespaceDict

class _Unimplemented:
    def __init__(self):
        return

UNIMPLEMENTED = _Unimplemented()
"""`NotImplementedError` would raise when someone try to reach this constant"""

class BaseConfig(NamespaceDict):
    """simple wrap of NamespaceDict with utils for config"""

    __checking_hooks__: List[Callable] = []
    
    def __getattribute__(self, __name: str):
        tem = super().__getattribute__(__name)

        if isinstance(tem, _Unimplemented):
            raise NotImplementedError(f"attribute: '{__name}' should be implemented")
        return tem

    def _check(self):
        for func in self.__checking_hooks__:
            func(self)
        return

    def check(self):
        self._check()
        for _, _ in dict(self).items():
            pass
        return
    
    def display(self, file: Writable = None):
        self._check()
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
