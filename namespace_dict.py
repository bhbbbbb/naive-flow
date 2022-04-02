from argparse import Namespace

class NamespaceDict(Namespace):
    def __iter__(self):
        for a_dir in dir(self):
            if not a_dir.startswith("__") and not callable(getattr(self, a_dir)):
                yield a_dir, getattr(self, a_dir)
