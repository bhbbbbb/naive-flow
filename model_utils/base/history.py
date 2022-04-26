from typing import List, Tuple, Union, Dict
import os
import re
import json
import matplotlib.pyplot as plt
import pandas as pd
from .writable import Writable
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

class History(NamespaceDict):

    # root of log dir
    root: str

    history: List[dict] # List of Stat in dict format

    checkpoints: dict

    def __init__(self, root: str, history: List[dict], checkpoints: dict):
        super().__init__()
        self.root = root
        self.history = history
        self.checkpoints = checkpoints


class HistoryUtils:
    """Utils for handling the operation relate to history.json"""

    history: History
    root: str
    path: str
    PLOT_CONFIGS: PlotConfig = None

    HISTORY_JSON_PATTERN = r"^\d{8}T\d{2}-\d{2}-\d{2}_history.json"

    def __init__(self, root: str, path: str = None, history: History = None):
        self.root = root
        root_name = os.path.basename(root)
        
        self.path = path or os.path.join(root, f"{root_name}_history.json")

        self.history = history or History(root=root, history=[], checkpoints={})
        return
    
    @classmethod
    def load_history(cls, root: str, start_epoch: int, logger: Writable):

        tem = [name for name in os.listdir(root) if re.match(cls.HISTORY_JSON_PATTERN, name)]

        assert len(tem) <= 1, f"Suppose <= 1 history.json in the folder, but got {len(tem)}"

        if len(tem) == 0:
            logger.write(f"Warning: No history.json in {root}")
            history = None
            history_log_path = None
        else:
            history_log_name = tem[0]

            history_log_path = os.path.join(root, history_log_name)
            with open(history_log_path, "r", encoding="utf-8") as fin:
                tem_dict = json.load(fin)
                history = History(**tem_dict)

                history.root = root
                if len(history.history) > start_epoch:
                    history.history = history.history[:start_epoch]
            
        return cls(root=root, path=history_log_path, history=history)

    def log_history(self, stat: Stat) -> str:
        """log history for the statistics coming from new epoch

        Args:
            stat (Stat): statistics data

        Returns:
            str: path to the log file history.json
        """
        self.history.history.append(stat.asdict())
        self.history.root = self.root
        if self.PLOT_CONFIGS is None:
            self.PLOT_CONFIGS = stat.get_plot_configs()

        os.makedirs(self.root, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fout:
            json.dump(self.history.asdict(), fout, indent=4)
        
        return self.path
    
    @staticmethod
    def _plot(title: str, train_data: list, valid_data: list, output_dir: str,
                y_lim: Tuple[float, float] = None,
                show: bool = False, save: bool = True) -> str:
        """plot history graph

        Args:
            title (str): title of image
            train_data (list): train_data
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
        plt.plot(train_data, label="train")
        plt.plot(valid_data, label="valid")
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
        valid_criteria: List[dict] = [data["valid_criteria"] for data in history.history]

        train_df = pd.DataFrame(train_criteria)
        valid_df = pd.DataFrame(valid_criteria)

        plot_configs = plot_configs or Criteria.get_plot_configs_from_registered_criterion()
        for key in train_criteria[0].keys():
            config = plot_configs.get(key, None)
            y_lim = None
            title = key
            if config is not None:
                if not config.plot:
                    continue
                title = config.full_name
                bottom = config.default_lower_limit_for_plot
                top = config.default_upper_limit_for_plot
                y_lim = (bottom, top)
            
            HistoryUtils._plot(
                title,
                train_df[key].tolist(),
                valid_df[key].tolist(),
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
        plot_configs = plot_configs or self.PLOT_CONFIGS
        HistoryUtils.plot_history(
            self.history,
            self.root,
            show=show,
            save=save,
            plot_configs=plot_configs,
        )
        return
