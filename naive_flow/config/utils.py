import json
from io import StringIO
from pydantic_settings import BaseSettings

def strfconfig(config: BaseSettings, padding_len: int = 4, min_len: int = 16):
    
    sio = StringIO()
    # sio.write("Configurations:\n")

    def walk_config(prefix: str, config: BaseSettings):

        dumpped_config = config.model_dump()
        for field, value in dict(config).items():
            delimiter = config.model_config.get("env_nested_delimiter")
            if isinstance(value, BaseSettings) and delimiter:
                new_prefix = f"{prefix}{field}{delimiter}"
                yield from walk_config(new_prefix, value)

            else:
                yield f"{prefix}{field}", dumpped_config[field]

    field_value_pairs = list(walk_config("", config))
    longest_field, _ = max(field_value_pairs, key=lambda p: len(p[0]))

    indent = max(padding_len + len(longest_field), min_len)

    for field, value in field_value_pairs:
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        sio.write(f"{field:{indent}}= {value}\n")


    sio.write("\n")
    string = sio.getvalue()
    sio.close()
    return string
