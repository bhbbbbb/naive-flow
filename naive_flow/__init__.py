import logging
from .config import BaseConfig
from .config.utils import strfconfig
from .log import LoggingLevel
from . import tracker


if not (logger := logging.getLogger(__name__)).hasHandlers():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    logger.propagate = False

def set_stream_logging_level(level: int):
    stream_handler.setLevel(level)
    return

def _load_checkpoint(checkpoint_path: str):
    # pylint: disable=import-outside-toplevel
    """Load the specific checkpoint save by Tracker"""
    import os
    import torch
    from typing import get_type_hints
    from .tracker.base_tracker import _CheckpointDict

    checkpoint_dict: _CheckpointDict = torch.load(checkpoint_path)
    if set(checkpoint_dict.keys()) != set(get_type_hints(_CheckpointDict).keys()):
        checkpoint_name = os.path.basename(checkpoint_path)
        raise ValueError(
            f"Checkpoint: {checkpoint_name} is not a checkpoint stored by Tracker."
        )
    return checkpoint_dict["user_data"], checkpoint_dict["config"]

def load_config_dict(checkpoint_path: str) -> dict:
    """Load the config used in the checkpoint"""
    _, config_dict = _load_checkpoint(checkpoint_path)
    return config_dict

def load_checkpoint(checkpoint_path: str):
    """Load the specific checkpoint save by Tracker"""

    user_data, _ = _load_checkpoint(checkpoint_path)
    return user_data
