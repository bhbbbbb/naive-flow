from argparse import Namespace

class NamespaceDict(Namespace):
    
    __length__: int

    def __getitem__(self, key):
        value = getattr(self, key)
        if isinstance(value, NamespaceDict):
            return dict(value)
        return value
    
    def __len__(self):
        if not hasattr(self, "__len"):
            self.__length__ = len(dict(self))
        return self.__length__

    def __iter__(self):
        for a_dir in dir(self):
            if not a_dir.endswith("__") and not callable(getattr(self, a_dir)):
                yield a_dir, self.__getitem__(a_dir)

    def asdict(self) -> dict:
        return dict(self)
    