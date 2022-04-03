from .namespace_dict import NamespaceDict

class Unimplemented:
    def __init__(self):
        return

UNIMPLEMENTED = Unimplemented()

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
    
    def display(self):
        print("Configuration:")
        for attr, value in dict(self).items():
            print("{:30} {}".format(attr, value))
        print("\n")
    