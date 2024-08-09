from .config.utils import strfconfig, dump_config, load_env_file
from .tracker.base.log import set_global, get_global
from . import tracker


def load_checkpoint(checkpoint_path: str):
    # pylint: disable=import-outside-toplevel
    """Load the specific checkpoint save by Tracker"""
    import os
    import glob
    import torch
    from typing import get_type_hints
    from .tracker.base_tracker import _CheckpointDict

    if os.path.isfile(checkpoint_path):
        # backward compatibility
        checkpoint_dict: _CheckpointDict = torch.load(checkpoint_path)

        if set(get_type_hints(_CheckpointDict).keys()
               ) <= set(checkpoint_dict.keys()):
            return checkpoint_dict["user_data"]
        checkpoint_name = os.path.basename(checkpoint_path)
        raise ValueError(
            f"Checkpoint: {checkpoint_name} is not a checkpoint stored by Tracker."
        )
    else:
        user_data = {}
        for path in glob.glob(os.path.join(checkpoint_path, "*.pth")):
            name, _ = os.path.splitext(os.path.basename(path))
            user_data[name] = torch.load(path)
        return user_data
