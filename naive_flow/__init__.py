from .config.utils import strfconfig
from . import tracker


def load_checkpoint(checkpoint_path: str):
    # pylint: disable=import-outside-toplevel
    """Load the specific checkpoint save by Tracker"""
    import os
    import torch
    from typing import get_type_hints
    from .tracker.base_tracker import _CheckpointDict

    checkpoint_dict: _CheckpointDict = torch.load(checkpoint_path)

    if set(get_type_hints(_CheckpointDict).keys()) <= set(checkpoint_dict.keys()):
        return checkpoint_dict["user_data"]
    checkpoint_name = os.path.basename(checkpoint_path)
    raise ValueError(
        f"Checkpoint: {checkpoint_name} is not a checkpoint stored by Tracker."
    )
