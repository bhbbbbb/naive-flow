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
        env_nested_delimiter="__",
        display_field_padding_len=4,
        display_field_min_len=16,
    )

    def __str__(self):
        
        sio = StringIO()
        sio.write("Configurations:\n")

        def walk_config(prefix: str, config: BaseConfig):
            for field, value in dict(config).items():
                if isinstance(value, BaseConfig):
                    new_prefix = f"{prefix}{field}{self.model_config.get('env_nested_delimiter')}"
                    yield from walk_config(new_prefix, value)
                    continue

                yield f"{prefix}{field}", str(value)

        field_value_pairs = list(walk_config("", self))
        longest_field, _ = max(field_value_pairs, key=lambda p: len(p[0]))
        padding_len = self.model_config.get("display_field_padding_len", 4)
        min_len = self.model_config.get("display_field_min_len", 16)

        indent = max(padding_len + len(longest_field), min_len)

        for field, value in field_value_pairs:
            sio.write(f"{field:{indent}}{value}\n")


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
