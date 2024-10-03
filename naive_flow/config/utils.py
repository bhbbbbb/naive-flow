import warnings
from contextlib import contextmanager
import re
import os
from typing import Literal
import json
from io import StringIO

from dotenv import dotenv_values
from pydantic import Field, version
from pydantic_settings import BaseSettings, DotEnvSettingsSource, sources
from pydantic_settings.sources import parse_env_vars


def strfconfig(
    config: BaseSettings,
    strformat: Literal["env", "markdown"] = "env",
    exclude_none: bool = True,
    padding_len: int = 4,
    min_len: int = 16,
    description: Literal["inline", "full"] | None = None,
    **kwargs,
):
    """format the config as string

    Args:
        config (BaseSettings): the config to format
        strformat (Literal[&quot;env&quot;, &quot;markdown&quot;], optional): "env" or "markdown".
            Defaults to "env".
        exclude_none (bool, optional): arg. passed to setting.model_dump(exclude_none=exclude_none).
            Defaults to True.
        padding_len (int, optional): Number of spaces padded after the longest field. Defaults to 4.
        min_len (int, optional): minimum length of fields. Defaults to 16.
        description (Literal[&#39;inline&#39;, &#39;full&#39;] | None, optional): whether includes
            description of fields as comments.
            If description=inline, the description will be added after value.
            If description=full, the description will be inserted in the next line following
                the key value pair.
            Defaults to None.
        kwargs: kwargs pass to config.model_dump(**kwargs)

    Returns:
        str: formatted config.
    """
    if description is not None and float(version.version_short()) < 2.7:  #pylint: disable=no-member
        warnings.warn(
            "Your version of pydantic is before 2.7, "
            "which does not take docstrings of fields as description."
        )

    sio = StringIO()
    print(f"## {config.__class__.__name__}", file=sio)
    if config.__class__.__doc__ is not None:
        header_doc = "\n".join(
            f"# {line}" for line in config.__class__.__doc__.split("\n")
        )
        print(header_doc, file=sio)
    if strformat == "markdown":
        print("```env", file=sio)

    def walk_config(prefix: str, config: BaseSettings):

        for field, data in config.model_dump(
            exclude_none=exclude_none, **kwargs
        ).items():
            delimiter = config.model_config.get("env_nested_delimiter")
            if isinstance(
                setting := getattr(config, field), BaseSettings
            ) and delimiter:
                new_prefix = f"{prefix}{field}{delimiter}"
                yield from walk_config(new_prefix, setting)

            else:
                # extra fields would not have info
                field_info = config.model_fields.get(field, None)
                yield f"{prefix}{field}", (
                    data, getattr(field_info, "description", "")
                )

    field_value_pairs = list(walk_config("", config))
    longest_field, _ = max(field_value_pairs, key=lambda p: len(p[0]))

    indent = max(padding_len + len(longest_field), min_len)

    for field, (value, desp) in field_value_pairs:
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        sio.write(f"{field:{indent}}= {value}")
        if description == "inline":
            if desp is not None:
                sio.write(" # " + re.sub(r"\s+", " ", desp))
        elif description == "full":
            if desp is not None:
                desp = "\n".join(f"# {line}" for line in desp.split("\n"))
                sio.write(f"\n{desp}\n")
        sio.write("\n")

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
    description: Literal["inline", "full"] | None = None,
    **kwargs,
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
                    description=description,
                    **kwargs,
                ),
                file=fout,
            )

        else:
            print(
                config.model_dump_json(exclude_none=exclude_none, **kwargs),
                file=fout,
            )
    return


@contextmanager
def add_system_vars(system_vars: dict[str, str] | None):

    def read_env_file(
        file_path,
        *,
        encoding: str | None = None,
        case_sensitive: bool = False,
        ignore_empty: bool = False,
        parse_none_str: str | None = None,
    ):
        with StringIO() as stream:
            for k, v in system_vars.items():
                print(f"{k}={v}", file=stream)
            with open(file_path, encoding=encoding) as fin:
                stream.write(fin.read())
            stream.seek(0)
            file_vars = dotenv_values(
                stream=stream, encoding=encoding or "utf8"
            )
        for k in system_vars:
            file_vars.pop(k)
        return parse_env_vars(
            file_vars, case_sensitive, ignore_empty, parse_none_str
        )

    original_fn = sources.read_env_file
    if system_vars:
        sources.read_env_file = read_env_file
    try:
        yield
    finally:
        sources.read_env_file = original_fn
    return


def load_env_file(
    env_path: str,
    env_nested_delimiter: str = "__",
    explode_env_files: bool = True,
    preset_env_vars: dict[str, str] = None,
) -> dict[str, str]:
    """Load an extended-version dot-env file. This allows env-file inheritance by
        taking `_env_file=path/to/env/file/`

    Args:
        env_path (str): path to the dot-env file
        env_nested_delimiter (str, optional): Delimiter. Defaults to "__".
        explode_env_files (bool, optional): whether explode all nested _env_file fields.
            Defaults to True.
        preset_env_vars (dict[str, str], optional): Preset the env key-value pairs for 
            dotenv parsing. For example, one can set preset_env_vars={'__file__': env_path}
            to use relative path in the dot-env. 

    Returns:
        dict[str, str]: data in strings. Can then be used as
        >>> env_data = nf.load_env_file(path, preset_env_vars={'__file__': path})
        >>> config = Config.model_validate_strings(env_data)

    """

    assert os.path.isfile(env_path), f"No env file found at {env_path}"

    def load(env_file: str):
        with add_system_vars(preset_env_vars):
            src = DotEnvSettingsSource(
                BaseSettings,
                env_file,
                env_nested_delimiter=env_nested_delimiter,
            )
            for key in src.env_vars:
                tem = key.split(env_nested_delimiter, maxsplit=1)
                if len(tem) >= 2 and tem[0].strip():
                    yield tem[0], src.explode_env_vars(
                        tem[0], Field(), src.env_vars
                    )
                else:
                    yield tem[0], src.get_field_value(Field(), tem[0])[0]

    def resolve_env_path(d: dict[str, str], env_path):

        for k, v in d.items():
            if isinstance(v, dict):
                yield k, dict(resolve_env_path(v, env_path))
            elif k == "_env_file":
                if "__file__" in v:
                    warnings.warn(
                        "Using __file__ has deprecated. "
                        "Consider __file__ a variable and use ${__file__} instead"
                    )
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
