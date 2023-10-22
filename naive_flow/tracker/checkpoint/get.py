import re
import os
from .utils import list_checkpoints, get_latest_log_dir
from ..base.arg_parser import get_args

def parse_args():

    args = get_args()
    if args.command in ["best", "load_best"]:
        return best(log_dir=args.log_dir)
    if args.command in ["latest", "load_latest"]:
        return latest(log_dir=args.log_dir)
    if args.command == "load":
        return args.checkpoint

    return None

def best(log_dir: str = None):

    def find_best(log_dir: str):
        checkpoint_list = list_checkpoints(log_dir)

        if len(checkpoint_list) == 0:
            raise RuntimeError(
                f"Cannot find any checkpoint in the log_dir: {log_dir}"
            )

        best_checkpoint = checkpoint_list[-1]

        for name in reversed(checkpoint_list):
            if re.match("_best", name):
                best_checkpoint = name
                break

        return os.path.join(log_dir, best_checkpoint)

    if log_dir is None:
        def hook(log_root_dir: str):
            log_dir = get_latest_log_dir(log_root_dir)
            return find_best(log_dir)
        return hook
    
    return find_best(log_dir)

def latest(log_dir: str = None):
    """Load latest saved checkpoint automatically

    Args:
        log_dir (str, optional): If not specfied, the latest time-formatted directory
            in `log_root_dir` will be used as log_dir.
    """

    def find_latest_checkpoint(log_dir: str):
        checkpoint_list = list_checkpoints(log_dir)

        assert len(checkpoint_list) > 0, f"cannot find any checkpoint in dir: '{log_dir}'"

        return os.path.join(log_dir, checkpoint_list[-1])

    if log_dir is None:
        def hook(log_root_dir: str):
            log_dir = get_latest_log_dir(log_root_dir)
            return find_latest_checkpoint(log_dir)
        return hook

    return find_latest_checkpoint(log_dir)
