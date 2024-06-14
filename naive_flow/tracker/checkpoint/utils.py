import re
import os
from datetime import datetime


def get_latest_time_formatted_dir(log_root_dir: str):
    """
    Structure of log_root_dir:
        └──log_root_dir
           ├─221012_16-08-48_DESKTOP-4A3P6E6
           |      20221017T01-44-38_epoch_10
           |      20221017T01-44-48_epoch_20
           │
           └─221013_17-09-52_DESKTOP-4A3P6E6
                   20220517T01-44-31_epoch_10
                   20220517T01-44-41_epoch_20
    """
    TIME_FORMAT_PATTERN = r"\d{6}_\d{2}-\d{2}-\d{2}"
    TIME_FORMAT = r"%y%m%d_%H-%M-%S"

    def get_dir_datetime(name: str) -> datetime:
        """check whether a name of dir is contains formatted time
        """
        match = re.search(TIME_FORMAT_PATTERN, name)
        if not match:
            return None

        path = os.path.join(log_root_dir, name)
        if not os.path.isdir(path):
            return None

        return datetime.strptime(match.group(0), TIME_FORMAT)

    arr = [
        (dir_name, get_dir_datetime(dir_name))
        for dir_name in os.listdir(log_root_dir)
    ]
    arr = list(filter(lambda tup: tup[1] is not None, arr))

    if len(arr) == 0:
        return None

    latest_dir, _ = max(arr, key=lambda tup: tup[1])
    latest_dir = os.path.join(log_root_dir, latest_dir)
    return latest_dir


def list_checkpoints(log_dir: str):
    # TODO: filiter using comment

    PATTERN = r"_epoch_(\d+)"

    def unorder_list():
        for name in os.listdir(log_dir):

            if not os.path.isfile(os.path.join(log_dir, name)):
                continue

            match = re.search(PATTERN, name)
            if match is None:
                continue

            yield (int(match.group(1)), name)

    checkpoints_list = list(unorder_list())
    checkpoints_list.sort(key=lambda t: t[0])

    return [name for _, name in checkpoints_list]


def list_checkpoints_later_than(checkpoint_path: str):

    checkpoint_name = os.path.basename(checkpoint_path)
    log_dir = os.path.dirname(checkpoint_path)

    checkpoint_list = list_checkpoints(log_dir)
    idx = checkpoint_list.index(checkpoint_name)

    return checkpoint_list[idx + 1:]


def get_latest_log_dir(log_root_dir: str):
    log_dir = get_latest_time_formatted_dir(log_root_dir)
    if log_dir is None:
        raise ValueError(
            f"Cannot find any log_dir in the log_root_dir: {log_root_dir}"
        )
    return log_dir
