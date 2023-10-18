import os
import re
import socket
from datetime import datetime
from fnmatch import fnmatch
from typing import Union, Tuple, overload, TypedDict, Type, List, get_type_hints
from functools import wraps
import logging

import termcolor
import torch
from torch.utils.tensorboard import SummaryWriter

from pydantic import BaseModel

from .tracker_config import TrackerConfig
from .base.arg_parser import get_args
from .base.logger import get_logger
from .base.metrics import MetricsLike, BUILTIN_METRICS
from .base.early_stopping_handler import EarlyStoppingHandler

__all__ = ["BaseTracker", "load_checkpoint", "load_config_dict"]

class SaveReason(BaseModel):

    early_stopping: bool = False
    end: bool = False
    regular: int = 0 # epochs_per_checkpoint
    best: bool = False


class _CheckpointDict(TypedDict):

    user_data: dict
    config: dict
    epoch: int

def once_assign_property(attribute: str):

    _attribute = f"_{attribute}"

    def getter(self):
        attr = getattr(self, _attribute, None)
        if attr is None:
            raise ValueError(f"'{attribute}' is referenced before assigned.")
        return attr
    
    def setter(self, value):
        attr = getattr(self, _attribute, None)
        if attr is not None:
            raise ValueError(f"'{attribute}' is not mutable")
        
        setattr(self, _attribute, value)
        return
    
    return property(getter, setter)

def _load_checkpoint(checkpoint_path: str):
    """Load the specific checkpoint save by Tracker"""

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


    def __init__(
        self,
        criterion: MetricsArgT = None,
        log_root_dir: str = None,
        epochs_per_checkpoint: int = None,
        early_stopping_rounds: int = None,
        save_n_best: int = None,
        comment: str = None,
        enable_logging: bool = True,
        config: TrackerConfig = None,
        metrics: List[MetricsArgT] = None,
    ):

        def parse_criterion_arg(arg: MetricsArgT):
            # 'loss/valid'
            # ('a_loss/valid', 'loss')
            # ('b_loss/valid', 'loss')
            # ('a_metrics/valid', Metrics)
            # ('b_metrics/valid', Metrics)

            if isinstance(arg, str):
                arg = (arg, os.path.dirname(arg))

            criterion_tag, criterion_type = arg
            
            if isinstance(criterion_type, str):

                if BUILTIN_METRICS.get(criterion_type) is None:
                    raise KeyError(
                        f"'{criterion_type}' is not a bulitin metrics. \n"
                        f"Currently available metrics are: {list(BUILTIN_METRICS.keys())}"
                    )

                criterion_type = BUILTIN_METRICS.get(criterion_type)
            
            return criterion_tag, criterion_type
        
        def default_arg(arg, value):
            return arg if arg is not None else value

        if config is None:
            config = TrackerConfig(
                log_root_dir=default_arg(log_root_dir, "runs"),
                epochs_per_checkpoint=default_arg(epochs_per_checkpoint, 0),
                early_stopping_rounds=default_arg(early_stopping_rounds, 0),
                save_n_best=default_arg(save_n_best, 1),
                enable_logging=enable_logging,
                comment=comment,
            )

        self.config = config

        if self.config.early_stopping_rounds > 0:
            if criterion is None:
                raise ValueError("The early stopping is enable while criterion is not provided.")
        
        self.metrics = dict(map(parse_criterion_arg, default_arg(metrics, [])))
        self.criterion_tag, self.criterion_type = parse_criterion_arg(criterion)\
                                                        if criterion is not None else (None, None)

        self._loaded = False
        self._evaluated = False
        return
    
    logger: logging.Logger = once_assign_property("logger")
    log_dir: str = once_assign_property("log_dir")
    writer: SummaryWriter = once_assign_property("writer")

    def _second_init(
        self,
        start_epoch: int,
        writer: SummaryWriter,
        logger: logging.Logger,
    ):
        self.start_epoch = start_epoch #pylint: disable=attribute-defined-outside-init
        self.log_dir = writer.log_dir
        self.writer = writer
        self.logger = logger
        self._loaded = True

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
            self._add_scalar_hook(tag, scalar_value, global_step)
            return original_fn(
                tag,
                scalar_value,
                global_step,
                walltime,
                new_style,
                double_precision,
            )
        
        writer.add_scalar = add_scalar_wrapper

        writer.add_text("config", self.config.model_dump_json(indent=4), start_epoch)
        return

    @overload
    @staticmethod
    def _init_writer(*, log_dir: str = None) -> SummaryWriter:...

    @overload
    @staticmethod
    def _init_writer(*, comment: str = None, log_root_dir: str = None) -> SummaryWriter:...

    @staticmethod
    def _init_writer(
            *,
            log_dir: str = None,
            comment: str = None,
            log_root_dir: str = None,
        ):
        

        if log_dir is not None:
            return SummaryWriter(log_dir)

        current_time = formatted_now()
        comment = comment or ""
        log_dir = os.path.join(
            log_root_dir, current_time + "_" + socket.gethostname() + comment
        )

        return SummaryWriter(log_dir)




    def load_checkpoint(self, checkpoint_path: str, delete_ok: bool = None):
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
            self.load(checkpoint_dict["user_data"])
            start_epoch = checkpoint_dict["epoch"] + 1
            # TODO: log the change of config checkpoint_dict["config"]
        
        else:
            self.load(checkpoint_dict)
            start_epoch = 0


        checkpoint_name = os.path.basename(checkpoint_path)
        checkpoint_log_dir = os.path.dirname(checkpoint_path)
        checkpoint_root_dir = os.path.dirname(checkpoint_log_dir)

        def create_new_dir(delete_ok: bool) -> bool:
            if os.path.abspath(checkpoint_root_dir) != os.path.abspath(self.config.log_root_dir):
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

            
        if create_new_dir(delete_ok):
            # create a new log_dir in the root_dir
            writer = BaseTracker._init_writer(
                log_root_dir=self.config.log_root_dir, comment=self.config.comment)
        else:
            # use the existing log_dir
            writer = BaseTracker._init_writer(log_dir=checkpoint_log_dir)

        logger = (
            get_logger(__name__, writer.log_dir)
                if self.config.enable_logging else get_logger(__name__)
        )
        # log information
        logger.info(f"Checkpoint {os.path.basename(checkpoint_path)} was loaded.")
        logger.debug(self.config)

        self._second_init(
            start_epoch=start_epoch,
            writer=writer,
            logger=logger,
        )
        return self


    def load_latest_checkpoint(self, log_dir: str = None, delete_ok: bool = None):
        """Load latest saved checkpoint automatically

        Args:
            log_dir (str, optional): If not specfied, the latest time-formatted directory
                in `log_root_dir` will be used as log_dir.
        """

        if log_dir is None:
            log_dir = get_latest_time_formatted_dir(self.config.log_root_dir)

            if log_dir is None:
                raise ValueError(
                    f"Cannot find any log_dir in the log_root_dir: {self.config.log_root_dir}"
                )

        checkpoint_list = list_checkpoints(log_dir)

        assert len(checkpoint_list) > 0, f"cannot find any checkpoint in dir: '{log_dir}'"

        return self.load_checkpoint(
            checkpoint_path=os.path.join(log_dir, checkpoint_list[-1]),
            delete_ok=delete_ok,
        )

    def load_best_checkpoint(
        self,
        log_dir: str = None,
        delete_ok: bool = None,
    ):
        """load best checkpoint from given log_dir. If the log_dir is not specified,
            the latest dir in `log_root_dir` will be used.

        Args:
            log_dir (str, optional): If not specfied, the latest time-formatted directory
                in `log_root_dir` will be used as log_dir.
        """

        if log_dir is None:
            log_dir = get_latest_time_formatted_dir(self.config.log_root_dir)
            if log_dir is None:
                raise RuntimeError(
                    f"Cannot find any log_dir in the log_root_dir: {self.config.log_root_dir}"
                )


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

        return self.load_checkpoint(
            checkpoint_path=os.path.join(log_dir, best_checkpoint),
            delete_ok=delete_ok,
        )

    
    def _start_new_training(self):
        
        writer = BaseTracker._init_writer(
            log_root_dir=self.config.log_root_dir, comment=self.config.comment)
        logger = (
            get_logger(__name__, writer.log_dir)
                if self.config.enable_logging else get_logger(__name__)
        )
        # log information
        logger.debug(self.config)

        self._second_init(
            start_epoch=0,
            writer=writer,
            logger=logger,
        )
        return


    def range(self, to_epoch: int):

        if not self._loaded:
            args = get_args()
            if args.command in ["best", "load_best"]:
                self.load_best_checkpoint(log_dir=args.log_dir, delete_ok=args.delete_ok)
            elif args.command in ["latest", "load_latest"]:
                self.load_latest_checkpoint(log_dir=args.log_dir, delete_ok=args.delete_ok)
            elif args.command == "load":
                self.load_checkpoint(args.checkpoint, delete_ok=args.delete_ok)
            else:
                self._start_new_training()

        if to_epoch <= self.start_epoch:
            raise ValueError(f"expect to_epoch > {self.start_epoch}, got: to_epoch={to_epoch}")


        # pylint: disable=attribute-defined-outside-init
        self._es_handler = EarlyStoppingHandler(
            self.config.early_stopping_rounds,
            self.logger,
        )

        for epoch in range(self.start_epoch, to_epoch):

            self._evaluated = False
            self.logger.info(f"Epoch {epoch + 1} / {to_epoch}:")

            yield epoch, self.writer

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
                self.logger.info("Early stopping!")
                save_reason.early_stopping = True
                self._save(epoch, save_reason)
                break

            if self._es_handler.is_best_epoch(epoch):
                save_reason.best = True

            if (
                save_reason.end and self.config.save_end or
                save_reason.regular or
                save_reason.best and self.config.save_n_best
            ):
                self._save(epoch, save_reason)


        self.logger.info(f"Training is finish for epochs: {to_epoch}")
        
        self.start_epoch = to_epoch
        return
            

    def _add_scalar_hook(
        self,
        tag: str,
        scalar_value,
        global_step=None,
    ):

        name = os.path.dirname(tag)
        group = os.path.basename(tag)
        if group == tag:
            group = None
        if tag == self.criterion_tag:
            if self._evaluated is True:
                raise RuntimeError("More than one evaluation scaler were added during an epoch.")

            scalar_value = self.criterion_type(scalar_value)
            self._es_handler.update(name, scalar_value, global_step)
            self._evaluated = True

        else:
            for pattern, met in self.metrics.items():
                if fnmatch(tag, pattern):
                    scalar_value = met(scalar_value)

            if name in BUILTIN_METRICS:
                scalar_value = BUILTIN_METRICS[name](scalar_value)

        group_str = f"/{group:5}" if group is not None else f"{group:6}"
        
        msg = f"{name:15}{group_str}: {scalar_value}"
        if tag == self.criterion_tag:
            attrs = ["underline"] if self._es_handler.is_best_epoch(global_step) else []
            msg = termcolor.colored(msg, "cyan", attrs=attrs)
        self.logger.info(msg)

        return


    def _save(self, cur_epoch: int, save_reason: SaveReason) -> str:

        def save_reason_suffix(save_reason: SaveReason):
            if save_reason.best:
                return "_best"
            
            if save_reason.early_stopping or save_reason.end:
                return "_end"
            
            return ""

        
        checkpoint_dict = _CheckpointDict(
            user_data=self.save(),
            config=self.config.model_dump(),
            epoch=cur_epoch,
        )
        now = formatted_now()
        
        name = f"{now}_epoch_{cur_epoch + 1}{save_reason_suffix(save_reason)}"

        path = os.path.join(self.log_dir, name)
        torch.save(checkpoint_dict, path)
        self.logger.info(f"Checkpoint: {name} was saved.")

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
                self.logger.info(
                    f"Checkpoint: {filename} was purged "
                    f"due to save_n_best={self.config.save_n_best}."
                )
                
        return name

    def load(self, checkpoint_dict: dict):
        raise NotImplementedError

    def save(self) -> dict:
        raise NotImplementedError
    
def formatted_now():
    return datetime.now().strftime(r"%b%d_%H-%M-%S")

def get_latest_time_formatted_dir(log_root_dir: str):
    """
    Structure of log_root_dir:
        └──log_root_dir
           ├─Oct12_16-08-48_DESKTOP-4A3P6E6
           |      20221017T01-44-38_epoch_10
           |      20221017T01-44-48_epoch_20
           │
           └─Oct13_17-09-52_DESKTOP-4A3P6E6
                   20220517T01-44-31_epoch_10
                   20220517T01-44-41_epoch_20
    """
    TIME_FORMAT_PATTERN = r"\w{3}\d{2}_\d{2}-\d{2}-\d{2}"
    TIME_FORMAT = r"%b%d_%H-%M-%S"
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

    arr = [(dir_name, get_dir_datetime(dir_name)) for dir_name in os.listdir(log_root_dir)]
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

def prompt_delete_later(checkpoint_name: str, to_delete: List[str]):

    to_delete_str = "\n".join(f"\t{name}" for name in to_delete)

    print(
        f"You are trying to load checkpoint: {checkpoint_name}\n" +
        to_delete_str + "\n"
        "While there are several checkpoints with larger epochs, "
        "you can choose to delete[D] them or keep[K] them and create a new log_dir.\n"
        "Note that you can set the argument `delete_ok` to avoid this prompt."
    )
    
    while True:
        option = input("Delete [D], Keep [K]: ")
        if option.lower() == "d":
            return True

        if option.lower() == "k":
            return False
