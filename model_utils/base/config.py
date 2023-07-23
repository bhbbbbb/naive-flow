from io import StringIO

from typing import TypeVar, Generic
from pydantic import RootModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

class Writable:
    """for type hint only"""
    def write(self, msg: str):
        pass

T = TypeVar("T")

class BaseConfig(BaseSettings):

    class MutableField(RootModel[T], Generic[T]):
        """Only works inside `BaseConfig`

        Example:
        ```
        class Config(BaseConfig):
            immutable_field: int
            mutable_field: BaseConfig.MutableField[int]
        ```
        """

        model_config = ConfigDict(frozen=False, validate_assignment=True, extra="forbid")

        root: T

    model_config = SettingsConfigDict(
        frozen=True,
        extra="allow",
        validate_assignment=True,
    )

    def __str__(self):
        sio = StringIO()
        sio.write("Configurations:\n")
        for field, value in self.model_dump().items():
            sio.write(f"{field:30} {value}\n")
        sio.write("\n")
        string = sio.getvalue()
        sio.close()
        return string

    def display(self, file: Writable = None):
        """display all the configurations

        Args:
            file (Writable, optional): file-like object. It'll be passed to the bulit-in function
                `print(file=file)` 
        """

        print(str(self), file=file, end="")
        return
    
    def __getattribute__(self, __name: str):
        tem = super().__getattribute__(__name)
        if isinstance(tem, BaseConfig.MutableField):
            return tem.root
        return tem
    
    def __setattr__(self, name: str, value) -> None:
        tem = super().__getattribute__(name)
        if isinstance(tem, BaseConfig.MutableField):
            tem.root = value
            return

        super().__setattr__(name, value)
        return
