import os
from typing import Literal
import json
from io import StringIO
from pydantic import Field
from pydantic_settings import BaseSettings, DotEnvSettingsSource


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


def load_env_file(
    env_path: str,
    env_nested_delimiter: str = "__",
    explode_env_files: bool = True,
) -> dict[str, str]:

    assert os.path.isfile(env_path), f"No env file found at {env_path}"

    def load(env_file: str):
        src = DotEnvSettingsSource(
            BaseSettings,
            env_file,
            env_nested_delimiter=env_nested_delimiter,
        )
        env_vars = src._load_env_vars()  # pylint: disable=protected-access
        for key in env_vars:
            tem = key.split(env_nested_delimiter, maxsplit=1)
            if len(tem) >= 2 and tem[0].strip():
                yield tem[0], src.explode_env_vars(tem[0], Field(), env_vars)
            else:
                yield tem[0], src.get_field_value(Field(), tem[0])[0]

    def resolve_env_path(d: dict[str, str], env_path):

        for k, v in d.items():
            if isinstance(v, dict):
                yield k, dict(resolve_env_path(v, env_path))
            elif k == "_env_file":
                v = v.replace("__file__", env_path)
                v = os.path.abspath(v)
                assert os.path.isfile(v), f"{v} does not exist"
                if explode_env_files:
                    yield "_env_obj", dict(resolve_env_path(dict(load(v)), v))
                else:
                    yield k, v
            else:
                yield k, v

    def handle_env_obj(data: dict):

        def merge(d: dict, env_data: dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    env_data[k] = merge(d[k], env_data.get(k, {}))
                else:
                    env_data[k] = v
            return env_data

        env_data = {}
        if "_env_obj" in data:
            env_data = handle_env_obj(data["_env_obj"])
            del data["_env_obj"]

        for k, v in data.items():
            if not isinstance(v, dict):
                env_data[k] = v
            else:
                env_data[k] = merge(v, env_data.get(k, {}))
                env_data[k] = handle_env_obj(env_data[k])

        return env_data

    data = dict(load(env_path))
    data = dict(resolve_env_path(data, env_path))
    if explode_env_files:
        data = handle_env_obj(data)
    return data
