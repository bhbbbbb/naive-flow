import logging
from typing import List, Tuple, Union, Dict
import os
import re
import json
import matplotlib.pyplot as plt
import pandas as pd
from .namespace_dict import NamespaceDict
from .criteria import Criteria, PlotConfig


class Stat:
    train_criteria: Criteria
    valid_criteria: Criteria
    test_criteria: Criteria
    epoch: int

    def __init__(
        self,
        epoch: int,
        train_criteria: Criteria,
        valid_criteria: Criteria = None,
        test_criteria: Criteria = None,
    ):
        self.epoch = epoch
        self.train_criteria = train_criteria
        self.valid_criteria = valid_criteria
        self.test_criteria = test_criteria
        return
    
    def display(self):
        
        if self.test_criteria is not None:
            print("Test Criteria:")
            self.test_criteria.display()
            
        else:
            print("Train Criteria:")
            self.train_criteria.display()

            if self.valid_criteria is not None:
                print("Valid Criteria:")
                self.valid_criteria.display()
        return
    
    def __iter__(self):
        for key, value in vars(self).items():
            if key == "epoch":
                yield key, value
            elif value is not None:
                assert isinstance(value, Criteria), f"got {type(value)}"
                yield key, value.asdict()

    def asdict(self):
        return dict(self)
    
    def get_plot_configs(self):
        return self.train_criteria.get_plot_configs()


class SaveReason(NamespaceDict):

    def __init__(
        self,
        *,
        early_stopping: bool = False,
        end: bool = False,
        regular: int = 0,
        best: bool = False,
    ):
        super().__init__(
            early_stopping = early_stopping,
            end = end,
            regular = regular,
            best = best,
        )
        return
    

    early_stopping: bool
    end: bool
    regular: int # epochs_per_checkpoint
    best: bool


class CheckpointInfo(NamespaceDict):

    def __init__(
        self,
        *,
        name: str,
        epoch: int,
        save_reason: SaveReason,
    ):
        super().__init__(
            name = name,
            epoch = epoch,
            save_reason = save_reason,
        )
        return

    name: str
    epoch: int
    save_reason: SaveReason

class History(NamespaceDict):

    history: List[dict] # List of Stat in dict format

    checkpoints: List[dict] # list of CheckpointInfo in dict format

    def __init__(self, history: List[dict], checkpoints: List[dict], **_):
        super().__init__()
        self.history = history
        self.checkpoints = checkpoints
        return


class HistoryUtils:
    """Utils for handling the operation relate to history.json"""

    history: History
    root: str
    path: str

    HISTORY_JSON_PATTERN = r"^\d{8}T\d{2}-\d{2}-\d{2}_history.json"

    def __init__(self, root: str, path: str = None, history: History = None):
        self.root = root
        root_name = os.path.basename(root)
        
        self.path = path or os.path.join(root, f"{root_name}_history.json")

        self.history = history or History(history=[], checkpoints=[])
        return
    
    @classmethod
    def load_history(
        cls,
        input_root: str,
        root: str,
        start_epoch: int,
        logger: logging.Logger,
    ):
        """

        Args:
            input_root (str): directory where the history.json locating
            root (str): directory where the history.json going to save at
            start_epoch (int): history after `start_epoch` would be trimmed
            logger (logging.Logger): _description_

        """
        tem = [name for name in os.listdir(input_root) if re.match(cls.HISTORY_JSON_PATTERN, name)]

        assert len(tem) <= 1, f"Suppose <= 1 history.json in the folder, but got {len(tem)}"

        if len(tem) == 0:
            logger.warning(f"Warning: No history.json in {input_root}")
            history = None
            history_log_path = None
        else:
            history_log_name = tem[0]

            input_history_log_path = os.path.join(input_root, history_log_name)
            with open(input_history_log_path, "r", encoding="utf-8") as fin:
                tem_dict = json.load(fin)
                history = History(**tem_dict)

                if len(history.history) > start_epoch:
                    history.history = history.history[:start_epoch]
            history_log_path = os.path.join(root, history_log_name)
        return cls(root=root, path=history_log_path, history=history)

    @staticmethod
    def get_history(root: str) -> Union[History, None]:
        """get history from `root`, return None if no history file found.

        Args:
            root (str): root direcotry which should contain history json file

        Returns:
            Union[History, None]: return None if there is no history file found.
        """
        tem = [
            name for name in os.listdir(root)\
            if re.match(HistoryUtils.HISTORY_JSON_PATTERN, name)
        ]

        if len(tem) == 0:
            return None
        
        history_log_path = os.path.join(root, tem[0])
        with open(history_log_path, "w", encoding="utf8") as fin:
            tem_dict = json.load(fin)
            history = History(**tem_dict)

        return history

    def new_saved_checkpoint(self, name: str, epoch: int, save_reason: SaveReason):
        self.history.checkpoints.append(
            CheckpointInfo(name=name, epoch=epoch, save_reason=save_reason).asdict()
        )
        return

    def log_history(self, stat: Stat) -> History:
        """log history for the statistics coming from new epoch

        Args:
            stat (Stat): statistics data

        Returns:
            str: path to the log file history.json
        """
        self.history.history.append(stat.asdict())

        os.makedirs(self.root, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fout:
            json.dump(self.history.asdict(), fout, indent=4)
        
        return self.history
    
    @staticmethod
    def _plot(
        title: str,
        train_data: Union[list, Tuple[list, list]],
        valid_data: Union[list, Tuple[list, list]],
        output_dir: str,
        y_lim: Tuple[float, float] = None,
        show: bool = False,
        save: bool = True
    ) -> str:
        """plot history graph

        Args:
            title (str): title of image
            train_data (list): train_data. `y` or `(x, y)`
            valid_data (list): valid_data
            output_dir (str): dir to store the output image
            y_lim (Tuple[float, float]): use specified scale of y instead of auto-scale.
            show (bool): whether show the image. Defaults to False.
            save (bool): whether save the image. Defaults to True.
                Note that (show or save) must be True.

        Returns:
            str: path of saved image, return None when save is False.
        """
        assert show or save, "show or save must not be both False."
        plt.figure(figsize=(10,5))

        def _plot_(data, label: str):
            if data is not None:
                if isinstance(data[0], list):
                    plt.plot(*data, label=label)
                else:
                    plt.plot(data, label=label)
            return

        _plot_(train_data, "train")
        _plot_(valid_data, "valid")

        if y_lim is not None:
            bottom, top = y_lim
            if bottom is not None:
                plt.ylim(bottom=bottom)
            if top is not None:
                plt.ylim(top=top)
            
        plt.title(title)
        plt.xlabel("epochs")
        plt.legend()
        path = os.path.join(output_dir, title.lower())
        if save:
            plt.savefig(path)
        if show:
            plt.show(block=False)
        return path


    @staticmethod
    def plot_history(
        history_or_path: Union[str, History],
        output_dir: str,
        show: bool = False,
        save: bool = True,
        plot_configs: Dict[str, PlotConfig] = None,
    ):
        """plot the loss-epoch figure

        Args:
            history_path (str): file of history (in json)
            output_dir (str): dir to export the result figure
            show (bool): whether show the image. Defaults to False.
            save (bool): whether save the image. Defaults to True.
                Note that (show or save) must be True.
            plot_configs (dict[key, PlotConfig]): ...
        """
        if isinstance(history_or_path, str):
            with open(history_or_path, "r", encoding="utf-8") as fin:
                tem_dict = json.load(fin)
                history = History(**tem_dict)
        else:
            history = history_or_path

        train_criteria: List[dict] = [data["train_criteria"] for data in history.history]
        valid_tem = [
            (data["epoch"], data["valid_criteria"]) for data in history.history\
                if "valid_criteria" in data
        ]
        valid_epochs, valid_criteria = zip(*valid_tem)
        valid_epochs = list(valid_epochs)
        valid_criteria: List[dict]

        train_df = pd.DataFrame(train_criteria)
        valid_df = pd.DataFrame(valid_criteria)

        plot_configs = plot_configs or Criteria.get_plot_configs_from_registered_criterion()
        for key, config in plot_configs.items():
            if not config.plot:
                continue

            y_lim = (config.default_lower_limit_for_plot, config.default_upper_limit_for_plot)

            train_data = train_df[key].tolist() if key in train_df.columns else None
            valid_data = (valid_epochs, valid_df[key].tolist()) if key in valid_df.columns else None
            if train_data or valid_data:
                HistoryUtils._plot(
                    config.full_name,
                    train_data,
                    valid_data,
                    output_dir=output_dir,
                    y_lim=y_lim,
                    show=show,
                    save=save,
                )
        return

    def plot(
        self,
        show: bool = False,
        save: bool = True,
        plot_configs: Dict[str, PlotConfig] = None,
    ):
        """plot the loss-epoch figure

        Args:
            show (bool): whether show the image. Defaults to False.
            save (bool): whether save the image. Defaults to True.
                Note that (show or save) must be True.
        """
        HistoryUtils.plot_history(
            self.history,
            self.root,
            show=show,
            save=save,
            plot_configs=plot_configs,
        )
        return
