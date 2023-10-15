__all__ = ["BaseConfig"]

from io import StringIO
from functools import lru_cache

from typing import TypeVar
from pydantic_settings import BaseSettings, SettingsConfigDict

class Writable:
    """for type hint only"""
    def write(self, msg: str):
        pass

T = TypeVar("T")


class BaseConfig(BaseSettings):

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

            dumpped_config = config.model_dump()
            for field, value in dict(config).items():
                if isinstance(value, BaseConfig):
                    new_prefix = f"{prefix}{field}{self.model_config.get('env_nested_delimiter')}"
                    yield from walk_config(new_prefix, value)
                    continue

                yield f"{prefix}{field}", dumpped_config[field]

        field_value_pairs = list(walk_config("", self))
        longest_field, _ = max(field_value_pairs, key=lambda p: len(p[0]))
        padding_len = self.model_config.get("display_field_padding_len", 4)
        min_len = self.model_config.get("display_field_min_len", 16)

        indent = max(padding_len + len(longest_field), min_len)

        for field, value in field_value_pairs:
            sio.write(f"{field:{indent}}= {value}\n")


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
    
    @lru_cache(maxsize=1)
    def __find_config_members(self):
        config_member_fields = []
        for field in self.__fields__:
            if isinstance(getattr(self, field), BaseConfig):
                config_member_fields.append(field)
        return config_member_fields


    def __getattribute__(self, __name: str):
        """Allow get attributes in member configs
        """
        
        try:
            value = super().__getattribute__(__name)
        except AttributeError as e:
            
            found = False
            for field in self.__find_config_members():
                member_config = getattr(self, field)
                try:
                    value = getattr(member_config, __name)
                    found = True
                except AttributeError:
                    pass

            if found is False:
                raise e

        return value
    
    # def __setattr__(self, name: str, value) -> None:
    #     tem = super().__getattribute__(name)

    #     super().__setattr__(name, value)
    #     return
