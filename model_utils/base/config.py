from io import StringIO
from .writable import Writable
from .namespace_dict import NamespaceDict

class Unimplemented:
    def __init__(self):
        return

UNIMPLEMENTED = Unimplemented()
"""`NotImplementedError` would raise when someone try to reach this constant"""

class BaseConfig(NamespaceDict):
    """simple wrap of NamespaceDict with utils for config"""


    def __getattribute__(self, __name: str):
        tem = super().__getattribute__(__name)

        if isinstance(tem, Unimplemented):
            raise NotImplementedError(f"attribute: '{__name}' should be implemented")
        return tem

    def _check_implementation(self, name: str):
        assert hasattr(self, name), f"attribute: {name} must be specified or overrided"
        return
    
    def display(self, file: Writable = None):
        sio = StringIO()
        sio.write("Configuration:\n")
        for attr, value in dict(self).items():
            sio.write(f"{attr:30} {value}\n")
        sio.write("\n")
        print(sio.getvalue(), file=file, end="")
        sio.close()
        return
    