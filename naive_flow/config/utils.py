from typing import Literal
import json
from io import StringIO
from pydantic_settings import BaseSettings


def strfconfig(
    config: BaseSettings,
    strformat: Literal["env", "markdown"] = "env",
    exclude_none: bool = True,
    padding_len: int = 4,
    min_len: int = 16,
):

    sio = StringIO()
    print(f"## {config.__class__.__name__}", file=sio)
    if strformat == "markdown":
        print("```env", file=sio)

    def walk_config(prefix: str, config: BaseSettings):

        for field, data in config.model_dump(exclude_none=exclude_none
                                             ).items():
            delimiter = config.model_config.get("env_nested_delimiter")
            if isinstance(
                setting := getattr(config, field), BaseSettings
            ) and delimiter:
                new_prefix = f"{prefix}{field}{delimiter}"
                yield from walk_config(new_prefix, setting)

            else:
                yield f"{prefix}{field}", data

    field_value_pairs = list(walk_config("", config))
    longest_field, _ = max(field_value_pairs, key=lambda p: len(p[0]))

    indent = max(padding_len + len(longest_field), min_len)

    for field, value in field_value_pairs:
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        sio.write(f"{field:{indent}}= {value}\n")

    if strformat == "markdown":
        print("```", file=sio)
    else:
        sio.write("\n")

    string = sio.getvalue()
    sio.close()
    return string


def dump_config(
    config: BaseSettings,
    file_path: str,
    dump_format: Literal["env", "json"] = "env",
    exclude_none: bool = True,
    padding_len: int = 4,
    min_len: int = 16,
):

    with open(file_path, "w", encoding="utf8") as fout:
        if dump_format == "env":
            print(
                strfconfig(
                    config,
                    "env",
                    exclude_none=exclude_none,
                    padding_len=padding_len,
                    min_len=min_len,
                ),
                file=fout,
            )

        else:
            print(
                config.model_dump_json(exclude_none=exclude_none),
                file=fout,
            )
    return
