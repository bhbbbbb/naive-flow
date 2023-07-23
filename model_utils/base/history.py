import logging
from typing import List, Tuple, Union, Dict, Optional
import os
import re
import json
import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel

from .criteria import Criteria, CriterionConfig


class Stat(BaseModel):
    epoch: int
    train_criteria: Criteria
    valid_criteria: Optional[Criteria] = None
    test_criteria: Optional[Criteria] = None

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
    
    # def __iter__(self):
    #     for key, value in vars(self).items():
    #         if key == "epoch":
    #             yield key, value
    #         elif value is not None:
    #             assert isinstance(value, Criteria), f"got {type(value)}"
    #             yield key, value.asdict()

    # def asdict(self):
    #     return dict(self)
    
    # def get_plot_configs(self):
    #     return self.train_criteria.get_plot_configs()


class SaveReason(BaseModel):

    early_stopping: bool = False
    end: bool = False
    regular: int = 0 # epochs_per_checkpoint
    best: bool = False


class CheckpointInfo(BaseModel):

    name: str
    epoch: int
    save_reason: SaveReason

class History(BaseModel):

    stats: List[Stat]

    checkpoints: List[CheckpointInfo]

    # def __init__(self, history: List[Stat], checkpoints: List[CheckpointInfo], **_):
    #     super().__init__()
    #     self.stats = history
    #     self.checkpoints = checkpoints
    #     return
    
    def get_best_stat(self) -> Stat:
        return max(self.stats, key=lambda s: s.valid_criteria)

    def get_best_criterion(self):
        return self.get_best_stat().valid_criteria.primary_criterion

    def get_best_checkpoint_info(self):
        bests = [checkpoint for checkpoint in self.checkpoints if checkpoint.save_reason.best]
        return bests[-1]

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

        self.history = history or History(stats=[], checkpoints=[])
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
                # tem_dict = json.load(fin)
                history = History.model_validate_json(fin.read())

                if len(history.stats) > start_epoch:
                    history.stats = history.stats[:start_epoch]
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
        with open(history_log_path, "r", encoding="utf8") as fin:
            # tem_dict = json.load(fin)
            # history = History(**tem_dict)
            history = History.model_validate_json(fin.read())

        return history

    def new_saved_checkpoint(self, name: str, epoch: int, save_reason: SaveReason):
        self.history.checkpoints.append(
            CheckpointInfo(name=name, epoch=epoch, save_reason=save_reason)
        )
        return

    def log_history(self, stat: Stat) -> History:
        """log history for the statistics coming from new epoch

        Args:
            stat (Stat): statistics data

        Returns:
            History: history object
        """
        self.history.stats.append(stat)

        os.makedirs(self.root, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fout:
            json.dump(self.history.model_dump(exclude_none=True), fout, indent=4)
        
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
        criterion_configs: Dict[str, CriterionConfig] = None,
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
                # tem_dict = json.load(fin)
                history = History.model_validate_json(fin.read())
        else:
            history = history_or_path

        train_criteria_list = [stat.train_criteria for stat in history.stats]
        valid_tem = [
            (stat.epoch, stat.valid_criteria) for stat in history.stats\
                if hasattr(stat, "valid_criteria") and stat.valid_criteria is not None
        ]
        valid_epochs, valid_criteria_list = list(zip(*valid_tem))
        valid_epochs: List[int]
        valid_criteria_list: List[Criteria]

        def model_dump(x: Criteria):
            return x.model_dump()
        train_df = pd.DataFrame(map(train_criteria_list, model_dump))
        valid_df = pd.DataFrame(map(valid_criteria_list, model_dump))

        def get_config_set(criteria: Criteria):
            return {name: c.config for name, c in criteria.items()}
        # train_criterion_config_set = {}
        # criterion_set = set(train_criteria[0]).union(valid_criteria[0])
        criterion_configs = {
            **get_config_set(train_criteria_list[0]),
            **get_config_set(valid_criteria_list[0]),
            **(criterion_configs or {})
        }
        # plot_configs = plot_configs or Criteria.get_plot_configs_from_registered_criterion()
        for key, config in criterion_configs.items():
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
        plot_configs: Dict[str, CriterionConfig] = None,
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
            criterion_configs=plot_configs,
        )
        return
