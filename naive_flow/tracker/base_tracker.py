import os
import re
import logging
import socket
from datetime import datetime
from fnmatch import fnmatch
from typing import (
    Union, Tuple, TypedDict, Type, List, get_type_hints, Callable, overload, NamedTuple, Dict, Any
)
from functools import wraps

import torch
from torch.utils.tensorboard import SummaryWriter

from pydantic import BaseModel
import termcolor

from .tracker_config import TrackerConfig
from .checkpoint.utils import list_checkpoints_later_than
from .base.logger import set_handlers, del_file_handler
from .base.metrics import MetricsLike, BUILTIN_METRICS, BUILTIN_TYPES
from .base.early_stopping_handler import EarlyStoppingHandler
from ..log import LoggingLevel
from .. import strfconfig

__all__ = ["BaseTracker"]

class SaveReason(BaseModel):

    early_stopping: bool = False
    end: bool = False
    regular: int = 0 # epochs_per_checkpoint
    best: bool = False


class _CheckpointDict(TypedDict):

    user_data: dict
    config: dict
    epoch: int

class _Scalar(NamedTuple):
    
    tag: str
    metrics: MetricsLike

    @classmethod
    def from_arg(cls, tag: str, scalar_type: Union[BUILTIN_TYPES, MetricsLike]):
        # ('a_loss/valid', 'loss')
        # ('b_loss/valid', 'loss')
        # ('a_metrics/valid', Metrics)
        # ('b_metrics/valid', Metrics)

        if isinstance(scalar_type, str):

            if BUILTIN_METRICS.get(scalar_type) is None:
                raise KeyError(
                    f"'{scalar_type}' is not a bulitin metrics. \n"
                    f"Currently available metrics are: {list(BUILTIN_METRICS.keys())}"
                )

            scalar_type = BUILTIN_METRICS.get(scalar_type)
        
        return cls(tag, scalar_type)

class _ScalarCache:
    def __init__(self):
        self.scalar_cache = [{"epoch": None}, {"epoch": None}]
        return

    def cache_scalar(self, tag: str, scalar, epoch: int):
        if self.scalar_cache[epoch & 1].get("epoch", None) != epoch:
            self.scalar_cache[epoch & 1] = {"epoch": epoch}

        self.scalar_cache[epoch & 1][tag] = scalar
        return
    
    def has_cache(self, tag: str, epoch: int):
        scalars = self.scalar_cache[epoch & 1]
        if scalars["epoch"] != epoch:
            return False
        return scalars.get(tag, False)

    def get_cache_scalars(self, epoch: int):
        scalars = self.scalar_cache[epoch & 1]
        assert scalars["epoch"] == epoch
        return scalars


# pylint: disable=invalid-name
MetricsArgT = Union[str, Tuple[str, Union[str, Type[MetricsLike]]]]

class BaseTracker:
    log_root_dir: str
    log_dir: str
    """_summary_
    \\RUNS
        ├─Oct12_19-00-17_DESKTOP-4A3P6E6nminst
        └─Oct12_19-01-16_DESKTOP-4A3P6E6nminst
    
        In the above case, \\RUNS is the log_root_dir, `Oct....` is the log_dir
    """


    @overload
    def __init__(
        self,
        *,
        config: TrackerConfig = None,
        from_checkpoint: Union[str, Callable] = None,
    ):...

    @overload
    def __init__(
        self,
        *,
        log_root_dir: str = None,
        epochs_per_checkpoint: int = None,
        early_stopping_rounds: int = None,
        save_n_best: int = None,
        save_end: bool = None,
        comment: str = None,
        enable_logging: bool = True,
        from_checkpoint: Union[str, Callable] = None,
    ):...

    def __init__(
        self,
        *,
        log_root_dir: str = None,
        epochs_per_checkpoint: int = None,
        early_stopping_rounds: int = None,
        save_n_best: int = None,
        save_end: bool = None,
        comment: str = None,
        enable_logging: bool = True,
        config: TrackerConfig = None,
        from_checkpoint: Union[str, Callable] = None,
    ):

        
        def default_arg(arg, value):
            return arg if arg is not None else value

        if config is None:
            config = TrackerConfig(
                log_root_dir=default_arg(log_root_dir, "runs"),
                epochs_per_checkpoint=default_arg(epochs_per_checkpoint, 0),
                early_stopping_rounds=default_arg(early_stopping_rounds, 0),
                save_n_best=default_arg(save_n_best, 1),
                save_end=default_arg(save_end, True),
                enable_logging=enable_logging,
                comment=comment,
            )

        self.config = config

        self._es_metrics = None
        self._metrics = []
        self._writer = None
        self._cur_epoch: int = None
        self._best_scalars: Dict[str, Any] = None
        self._scalar_cache = _ScalarCache()

        if from_checkpoint is None:
            self.log_dir, self.start_epoch = BaseTracker._start_new_training(config)
        else:
            checkpoint_path = from_checkpoint if isinstance(from_checkpoint, str)\
                                                        else from_checkpoint(config.log_root_dir)
            self.log_dir, self.start_epoch =\
                    BaseTracker._load_checkpoint(checkpoint_path, config, user_load_hook=self.load)
        
        self.logger = logging.getLogger(__name__)
        self._evaluated = False
        return

    

    @staticmethod
    def _start_new_training(config: TrackerConfig):
        log_dir = new_time_formatted_log_dir(config.comment, config.log_root_dir)

        set_handlers(log_dir if config.enable_logging else None)
        logging.getLogger(__name__).debug(strfconfig(config))
        start_epoch = 0
        return (log_dir, start_epoch)


    @staticmethod
    def _load_checkpoint(
        checkpoint_path: str,
        config: TrackerConfig,
        user_load_hook: Callable,
    ):
        """Load the specfiic checkpoint

        Args:
            checkpoint_path (str): path of saved checkpoint
                Notice that if the directory of the checkpoint does not in the
                `self.log_root_dir`, then it would be consideranother directory will be
                created in `self.log_root_dir`

        """

        assert os.path.isfile(checkpoint_path), (
            f"expect checkpoint_path: '{checkpoint_path}' is file."
        )
        checkpoint_dict: _CheckpointDict = torch.load(checkpoint_path)

        if set(checkpoint_dict.keys()) == set(get_type_hints(_CheckpointDict).keys()):
            # Checkpoint was created by tracker
            user_load_hook(checkpoint_dict["user_data"])
            start_epoch = checkpoint_dict["epoch"] + 1
            # TODO: log the change of config checkpoint_dict["config"]
        
        else:
            user_load_hook(checkpoint_dict)
            start_epoch = 0


        checkpoint_name = os.path.basename(checkpoint_path)
        checkpoint_log_dir = os.path.dirname(checkpoint_path)
        checkpoint_root_dir = os.path.dirname(checkpoint_log_dir)

        def create_new_dir(delete_ok: bool) -> bool:
            if os.path.abspath(checkpoint_root_dir) != os.path.abspath(config.log_root_dir):
                return True
                
            to_delete = list_checkpoints_later_than(checkpoint_path)

            if len(to_delete) == 0:
                return False

            delete_ok = delete_ok if delete_ok is not None\
                            else prompt_delete_later(checkpoint_name, to_delete)
            
            if delete_ok is False:
                return True

            for name in to_delete:
                path = os.path.join(checkpoint_log_dir, name)
                os.remove(path)
                print(f"Checkpoint: {name} was purged.")
            
            return False

            
        if create_new_dir(config.delete_ok):
            # create a new log_dir in the root_dir
            log_dir = new_time_formatted_log_dir(config.comment, config.log_root_dir)
        else:
            # use the existing log_dir
            log_dir = checkpoint_log_dir

        set_handlers(log_dir if config.enable_logging else None)
        logger = logging.getLogger(__name__)
        logger.log(
            LoggingLevel.ON_CHECKPOINT_LOAD,
            f"Checkpoint {os.path.basename(checkpoint_path)} was loaded."
        )
        logger.debug(strfconfig(config))

        return (log_dir, start_epoch)

    def _register_add_scalar_hook(self, writer: SummaryWriter):
        
        if getattr(writer.add_scalar, "__wrapper__", None) is self:
            # has been wrapped
            return writer

        self._writer = writer
        original_fn = writer.add_scalar
        @wraps(original_fn)
        def add_scalar_wrapper(
            tag,
            scalar_value,
            global_step=None,
            walltime=None,
            new_style=False,
            double_precision=False,
        ):
            original_fn(
                tag,
                scalar_value,
                global_step,
                walltime,
                new_style,
                double_precision,
            )
            return self._add_scalar_hook(tag, scalar_value, global_step)
        
        add_scalar_wrapper.__wrapper__ = self
        writer.add_scalar = add_scalar_wrapper
        return writer

    def create_summary_writer(
        self,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = ""
    ):
        """Create a registed summary writer. This is a shorthand of
        ```python
        writer = SummaryWriter(log_dir=tracker.log_dir, purge_step=tracker.start_epoch)
        writer = tracker.register_summary_writer(writer)
        ```

        Returns:
            SummaryWriter: return a summary writer which has been registed
        """
        assert self._writer is None, (
            "Try to create a new summary writer while there is one writer that has been registered"
        )
        writer = SummaryWriter(
            log_dir=self.log_dir,
            purge_step=self.start_epoch,
            max_queue=max_queue,
            flush_secs=flush_secs,
            filename_suffix=filename_suffix,
        )
        return self._register_add_scalar_hook(writer)

    def register_summary_writer(self, writer: SummaryWriter):
        assert self._writer is None, (
            "Try to register a summary writer while another summary writer has been registerd."
        )
        return self._register_add_scalar_hook(writer)

    def register_scalar(
        self,
        tag: str,
        scalar_type: Union[BUILTIN_TYPES, MetricsLike],
        for_early_stopping: bool = False,
    ):

        if for_early_stopping:
            assert self._es_metrics is None, (
                "try to register a second scalar for early stopping, which is not allowed."
            )
            self._es_metrics = _Scalar.from_arg(tag, scalar_type)

        else:
            self._metrics.append(_Scalar.from_arg(tag, scalar_type))
        return


    def range(self, to_epoch: int):

        if to_epoch <= self.start_epoch:
            raise ValueError(f"expect to_epoch > {self.start_epoch}, got: to_epoch={to_epoch}")


        # pylint: disable=attribute-defined-outside-init
        self._es_handler = EarlyStoppingHandler(
            self.config.early_stopping_rounds,
        )

        for epoch in range(self.start_epoch, to_epoch):

            self._cur_epoch = epoch
            self._evaluated = False
            self.logger.log(
                LoggingLevel.TRAINING_PROGRESS,
                f"Epoch {epoch} / {to_epoch}:"
            )

            yield epoch

            if self._evaluated is False:
                if self.config.early_stopping_rounds > 0:
                    self.logger.warning(
                        "EarlyStoppingWarning:\n"
                        "Early stopping is enable while no criterion scalar has been added.\n"
                        "TODO: provide ways to suppress this warning\n" #TODO
                    )

            save_reason = SaveReason()

            if epoch == to_epoch - 1:
                save_reason.end = True
            
            if (
                self.config.epochs_per_checkpoint
                and (epoch + 1 - self.start_epoch) % self.config.epochs_per_checkpoint == 0
            ):
                save_reason.regular = self.config.epochs_per_checkpoint

            if self._es_handler.should_stop(epoch):
                self.logger.log(LoggingLevel.TRAINING_PROGRESS, "Early stopping!")
                save_reason.early_stopping = True
                if self.config.save_end is True:
                    self._save(epoch, save_reason)
                break

            if self._es_handler.is_best_epoch(epoch):
                save_reason.best = True
                if self._es_metrics is not None:
                    self.get_best_scalars(True)

            if (
                save_reason.end and self.config.save_end or
                save_reason.regular or
                save_reason.best and self.config.save_n_best
            ):
                self._save(epoch, save_reason)


        self.logger.log(
            LoggingLevel.TRAINING_PROGRESS,
            f"Training is finish for epochs: {to_epoch}"
        )
        
        self.start_epoch = self._cur_epoch + 1
        return
    
    def get_best_scalars(self, no_within_loop_warning: bool = False):
        assert self._es_metrics is not None , (
            "You have to have registered summarywriter in tracker using "
            "register_scalar(..., for_early_stopping=True) so that tracker can know what are the "
            "best metrics for an epoch"
        )

        if self.start_epoch < self._cur_epoch + 1:
            # called within training loop
            if not self._evaluated:
                # called before  for-earlystopping scalar added
                return self._best_scalars

            if no_within_loop_warning is False:
                self.logger.warning(
                    "get_best_scalars called within loop. If this epoch is evaluated as the best"
                    ", and some of the scalar have not been added, they will not be included.\n"
                    "You can make sure the get_best_scalars is called after all of the scalars for "
                    "this epoched added, and pass no_within_loop_warning=True to suppress this "
                    "warning."
                )
            # update best_scalar
            if self._es_handler.is_best_epoch(self._cur_epoch):
                scalars = self._scalar_cache.get_cache_scalars(self._cur_epoch)
                self._best_scalars = scalars
            return self._best_scalars

        assert self.start_epoch == self._cur_epoch + 1
        # call after training loop
        return self._best_scalars

    def is_regular_saving_epoch(self, epoch: int):
        """Return whether the epoch is a regular saving epoch determined by
            `epochs_per_checkpoint`

        """
        return (
            bool(self.config.epochs_per_checkpoint)
            and (epoch + 1 - self.start_epoch) % self.config.epochs_per_checkpoint == 0
        )

    def is_best_epoch(self, epoch: int):
        return self._es_handler.is_best_epoch(epoch)

    def add_scalar(
        self,
        tag: str,
        scalar_value,
        global_step=None,
    ):
        self._add_scalar_hook(tag, scalar_value, global_step)
        return
    
    def _add_scalar_hook(
        self,
        tag: str,
        scalar_value,
        global_step=None,
    ):

        if global_step is None:
            global_step = self._cur_epoch or 0

        cached_scalar = self._scalar_cache.has_cache(tag, global_step)
        if cached_scalar is not False:
            assert cached_scalar == scalar_value, (
                f"more than one unique scalar value was added for tag: {tag}, epoch: {global_step}"
            )
            return 
        self._scalar_cache.cache_scalar(tag, scalar_value, global_step)
        name = os.path.dirname(tag)
        group = os.path.basename(tag)
        if group == tag:
            group = None
            name = tag
        if self._es_metrics is not None and tag == self._es_metrics.tag:
            if self._evaluated is True:
                raise RuntimeError("More than one evaluation scaler were added during an epoch.")

            scalar_value = self._es_metrics.metrics(scalar_value)
            self._es_handler.update(tag, scalar_value, global_step)
            self._evaluated = True

        else:
            found = False
            for pattern, met in self._metrics:
                if fnmatch(tag, pattern):
                    scalar_value = met(scalar_value)
                    found = True
                    break

            if not found and name in BUILTIN_METRICS:
                scalar_value = BUILTIN_METRICS[name](scalar_value)

        group_str = f"/{group}" if group is not None else ""
        
        msg = f"{name:15}{group_str:6}: {scalar_value}"
        if self._es_metrics is not None and tag == self._es_metrics.tag:
            attrs = ["underline"] if self._es_handler.is_best_epoch(global_step) else []
            msg = termcolor.colored(msg, "cyan", attrs=attrs)
        self.logger.log(LoggingLevel.ON_SCALAR_ADD, msg)
        return

    def save_checkpoint(self, suffix: str = ""):
        """Save the current state as a checkpoint manually.

        Args:
            suffix (str, optional): Suffix append to the filename of checkpoint.
            Defaults to "".
        """
        
        return self.__save(self._cur_epoch, suffix=suffix)

    
    def _save(self, cur_epoch: int, save_reason: SaveReason) -> str:

        def save_reason_suffix(save_reason: SaveReason):
            if save_reason.best:
                return "_best"
            
            if save_reason.early_stopping or save_reason.end:
                return "_end"
            
            return ""
        
        return self.__save(cur_epoch, suffix=save_reason_suffix(save_reason))
        
    def __save(self, cur_epoch: int, suffix: str = "") -> str:
        checkpoint_dict = _CheckpointDict(
            user_data=self.save(),
            config=self.config.model_dump(),
            epoch=cur_epoch,
        )
        now = formatted_now()
        
        name = f"{now}_epoch_{cur_epoch}{suffix}"

        path = os.path.join(self.log_dir, name)
        torch.save(checkpoint_dict, path)
        self.logger.log(
            LoggingLevel.ON_CHECKPOINT_SAVE,
            f"Checkpoint: {name} was saved."
        )

        if self.config.save_n_best:
            # purge redundant checkpoints
            pat = r"_epoch_(\d+)_best"
            best_checkpoints = [
                (int(match.group(1)), filename) for filename in os.listdir(self.log_dir)
                    if (match := re.search(pat, filename)) is not None
            ]
            best_checkpoints.sort(key=lambda t: t[0])
            not_n_best = best_checkpoints[:-self.config.save_n_best]

            for _epoch, filename in not_n_best:
                os.remove(os.path.join(self.log_dir, filename))
                self.logger.log(
                    LoggingLevel.ON_CHECKPOINT_PURGE,
                    f"Checkpoint: {filename} was purged "
                    f"due to save_n_best={self.config.save_n_best}."
                )
                
        return name
    
    def __del__(self):
        if hasattr(self, "log_dir"):
            del_file_handler(self.log_dir)
        return

    def load(self, checkpoint_dict: dict):
        raise NotImplementedError

    def save(self) -> dict:
        raise NotImplementedError
    
def formatted_now():
    return datetime.now().strftime(r"%b%d_%H-%M-%S")

def new_time_formatted_log_dir(
    comment: str,
    log_root_dir: str,
):
    current_time = formatted_now()
    comment = comment or ""
    log_dir = os.path.join(
        log_root_dir, current_time + "_" + socket.gethostname() + comment
    )

    os.makedirs(log_dir)

    return log_dir

def prompt_delete_later(checkpoint_name: str, to_delete: List[str]):

    to_delete_str = "\n".join(f"\t{i + 1}. {name}" for i, name in enumerate(to_delete))

    print(
        f"You are trying to load checkpoint: {checkpoint_name}\n" +
        to_delete_str + "\n"
        f"While there are {len(to_delete)} checkpoints with larger epochs shown above, "
        "you can choose to delete[D] them or keep[K] them and create a new log_dir.\n"
        "Note that you can set the `tracker_config.delete_ok` to avoid this prompt."
    )
    
    while True:
        option = input("Delete [D], Keep [K]: ")
        if option.lower() == "d":
            return True

        if option.lower() == "k":
            return False

# def once_assign_property(attribute: str):

#     _attribute = f"_{attribute}"

#     def getter(self):
#         attr = getattr(self, _attribute, None)
#         if attr is None:
#             raise ValueError(f"'{attribute}' is referenced before assigned.")
#         return attr
    
#     def setter(self, value):
#         attr = getattr(self, _attribute, None)
#         if attr is not None:
#             raise ValueError(f"'{attribute}' is not mutable")
        
#         setattr(self, _attribute, value)
#         return
    
#     return property(getter, setter)
