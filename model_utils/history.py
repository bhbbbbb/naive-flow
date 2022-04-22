from typing import List, Tuple
import os
import re
import json
import matplotlib.pyplot as plt
import pandas as pd
from .base.writable import Writable
from .base.namespace_dict import NamespaceDict

class Stat:
    train_loss: float
    valid_loss: float
    train_acc: float
    valid_acc: float
    test_loss: float
    test_acc: float
    epoch: int

    def __init__(
        self,
        epoch: int,
        train_loss: float,
        valid_loss: float,
        train_acc: float = None,
        valid_acc: float = None,
        test_loss: float = None,
        test_acc: float = None,
    ):
        self.epoch = epoch
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.train_acc = train_acc
        self.valid_acc = valid_acc
        self.test_loss = test_loss
        self.test_acc = test_acc
        return
    
    def display(self):
        
        if self.test_loss is not None:
            print(f"test_loss: {self.test_loss: .6f}")
            if self.test_acc is not None:
                print(f"test_acc: {self.test_acc * 100: .2f} %")
        else:
            print(f"train_loss: {self.train_loss: .6f}")
            print(f"valid_loss: {self.valid_loss: .6f}")
            if self.train_acc is not None:
                print(f"train_acc: {self.train_acc * 100: .2f} %")
            if self.valid_acc is not None:
                print(f"valid_acc: {self.valid_acc * 100: .2f} %")
        return
    
    def __iter__(self):
        for key, value in vars(self).items():
            if value is not None:
                yield key, value

    def asdict(self):
        return dict(self)
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
            plt.ylim(y_lim)
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
        history_path: str,
        output_dir: str,
        loss_uplimit: float = None,
        plot_accuracy: bool = False,
        acc_autoscale: bool = False,
        show: bool = False,
        save: bool = True
    ):
        """plot the loss-epoch figure

        Args:
            history_path (str): file of history (in json)
            output_dir (str): dir to export the result figure
            loss_uplimit (float): scale the upper limit to the specfied value.
                Defaluts to None (autoscale).
            plot_accuracy (bool): whether plot the accuarcy graph. Defaults to False.
            acc_autoscale (bool): whether autoscale the accuracy. Defaults to False. When
                plot_accuracy is set to False, this is not taking effect.
            show (bool): whether show the image. Defaults to False.
            save (bool): whether save the image. Defaults to True.
                Note that (show or save) must be True.
        """
        with open(history_path, "r", encoding="utf-8") as fin:
            tem_dict = json.load(fin)
            history = History(**tem_dict)
        
        df = pd.DataFrame(history.history)
        loss_y_lim = None if loss_uplimit is None else (0, loss_uplimit)
        acc_y_lim =  None if acc_autoscale else (0.0, 1.0)

        HistoryUtils._plot(
            "Loss",
            df["train_loss"].tolist(),
            df["valid_loss"].tolist(),
            output_dir,
            y_lim=loss_y_lim,
            show=show,
            save=save
        )
        if plot_accuracy:
            HistoryUtils._plot(
                "Accuracy",
                df["train_acc"].tolist(),
                df["valid_acc"].tolist(),
                output_dir,
                y_lim=acc_y_lim,
                show=show,
                save=save
            )
        return

    def plot(
        self,
        loss_uplimit: float = None,
        plot_accuracy: bool = False,
        acc_autoscale: bool = False,
        show: bool = False,
        save: bool = True,
    ):
        """plot the loss-epoch figure

        Args:
            history_path (str): file of history (in json)
            output_dir (str): dir to export the result figure
            loss_uplimit (float): scale the upper limit to the specfied value.
                Defaluts to None (autoscale).
            plot_accuracy (bool): whether plot the accuarcy graph. Defaults to False.
            acc_autoscale (bool): whether autoscale the accuracy. Defaults to False. When
                plot_accuracy is set to False, this is not taking effect.
            show (bool): whether show the image. Defaults to False.
            save (bool): whether save the image. Defaults to True.
                Note that (show or save) must be True.
        """
        HistoryUtils.plot_history(
            self.path,
            self.root,
            loss_uplimit=loss_uplimit,
            plot_accuracy=plot_accuracy,
            acc_autoscale=acc_autoscale,
            show=show,
            save=save,
        )
        return
