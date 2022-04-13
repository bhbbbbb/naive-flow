from argparse import Namespace

class NamespaceDict(Namespace):
    
    __len: int

    def __getitem__(self, key):
        value = getattr(self, key)
        if isinstance(value, NamespaceDict):
            return dict(value)
        return value
    
    def __len__(self):
        if not hasattr(self, "__len"):
            self.__len = len(dict(self))
        return self.__len

    def __iter__(self):
        for a_dir in dir(self):
            if not a_dir.startswith("__") and not callable(getattr(self, a_dir)):
                yield a_dir, self.__getitem__(a_dir)
