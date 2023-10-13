from argparse import Namespace
from typing_extensions import deprecated

@deprecated
class NamespaceDict(Namespace):
    
    __length__: int

    def __getitem__(self, key):
        value = getattr(self, key)
        if isinstance(value, NamespaceDict):
            return dict(value)
        return value
    
    def __len__(self):
        if not hasattr(self, "__length__"):
            self.__length__ = len(dict(self))
        return self.__length__

    def __iter__(self):
        for a_dir in dir(self):
            if not self._is_builtin_name(a_dir) and not callable(getattr(self, a_dir, None)):
                yield a_dir, self.__getitem__(a_dir)

    def asdict(self) -> dict:
        """return dict, ok with nested `NamespaceDict`

        E.g.
        ```
        class Foo(NamespaceDict):
            a: int
            b: int

        class Bar(NamespaceDict):
            foo: Foo
            c: int

        bar = Bar(c=1, foo=Foo(a=1, b=2))
        print(bar.asdict())
        # {'c': 1, 'foo': {'a': 1, 'b': 2}}
        ```
        """
        return dict(self)
    
    @staticmethod
    def _is_builtin_name(__name: str):
        return __name.endswith("__")
    