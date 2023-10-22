from functools import lru_cache
from typing import Union, Literal
from argparse import ArgumentParser, Namespace

class Args(Namespace):
    command: Literal["load", "best", "load_best", "latest", "load_latest"]
    checkpoint: Union[str, None]
    # delete_ok: Union[bool, None]
    log_dir: Union[str, None]

@lru_cache(maxsize=1)
def get_default_arg_parser() -> ArgumentParser:

    arg_parser = ArgumentParser(
        "tracker",
        description=(
            "Command line interface to decide whether start a new training or "
            "load from existing checkpoint"
        ),
    )

    # arg_parser.add_argument(
    #     "-e", "--to-epoch",
    #     type=int,
    #     help=(
    #         "Specify to_epoch passed to tracker.range(to_epoch). "
    #         "Note that this will OVERRIDE the argument written in your code."
    #     ),
    # )
    sub_parsers = arg_parser.add_subparsers(dest="command")

    load_parser = sub_parsers.add_parser(
        "load",
        help="Load a specific checkpoint",
    )
    load_parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to the checkpoint",
    )
    # load_parser.add_argument(
    #     "--delete-ok",
    #     type=bool,
    # )

    load_best_parser = sub_parsers.add_parser(
        "load_best",
        aliases=["best"],
        help="Load the best checkpoint in the latest log_dir",
    )
    # load_best_parser.add_argument(
    #     "--delete-ok",
    #     type=bool,
    # )
    load_best_parser.add_argument(
        "--log-dir",
        type=str,
        help=(
            "Load the best checkpoint from the given log-dir. If not specified, "
            "Tracker will detect and use the latest log-dir in the parameter log_root_dir"
        )
    )

    load_latest_parser = sub_parsers.add_parser(
        "load_latest",
        aliases=["latest"],
        help=(
            "Load the latest checkpoint in the given log-dir. "
            "If log-dir is not specified, load the latest checkpoint in the latest log-dir"
        ),
    )
    # load_latest_parser.add_argument(
    #     "--delete-ok",
    #     type=bool,
    # )
    load_latest_parser.add_argument(
        "--log-dir",
        type=str,
        help=(
            "Load the latest checkpoint from the given log-dir. If not specified, "
            "Tracker will detect and use the latest log-dir in the parameter log_root_dir"
        )
    )
    return arg_parser

def use_default_arg_parser():
    get_default_arg_parser().parse_args()
    return

def get_args() -> Args:
    args = get_default_arg_parser().parse_args()
    return args
