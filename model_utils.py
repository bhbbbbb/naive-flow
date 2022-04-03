import os
try:
    from typing import TypedDict
except ImportError: # for python < 3.8
    from typing_extensions import TypedDict
from typing import List, Tuple
import re
import json
from datetime import datetime
from argparse import Namespace
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import Dataset
import pandas as pd

from .config import BaseConfig
from ..logger import Logger
from .writable import Writable

class Stat:
    train_loss: float
    valid_loss: float
    train_acc: float
    valid_acc: float
    test_loss: float
    test_acc: float
    epoch: int

    def __init__(self, epoch: int, train_loss: float, valid_loss: float,
                train_acc: float, valid_acc: float):
        self.epoch = epoch
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.train_acc = train_acc
        self.valid_acc = valid_acc
        return
    
    def display(self):
        if hasattr(self, "test_loss") and hasattr(self, "test_acc"):
            print(f"test_loss: {self.test_loss: .6f}")
            print(f"test_acc: {self.test_acc * 100: .2f} %")
        else:
            print(f"train_loss: {self.train_loss: .6f}")
            print(f"train_acc: {self.train_acc * 100: .2f} %")
            print(f"valid_loss: {self.valid_loss: .6f}")
            print(f"valid_acc: {self.valid_acc * 100: .2f} %")
        return


class ModelStates(Namespace):
    # epoch to start from (0 is the first)
    start_epoch: int

    # config: BaseConfig

    ########## torch built-in model states ############
    model_state_dict: dict
    optimizer_state_dict: dict

    # statistic of the last epoch
    stat: Stat

class History(TypedDict):

    # root of log dir
    root: str

    history: List[dict] # List of Stat in dict format

    checkpoints: dict


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
        else:
            history_log_name = tem[0]

            history_log_path = os.path.join(root, history_log_name)
            with open(history_log_path, "r") as fin:
                history = json.load(fin)
                history["root"] = root
                if len(history["history"]) > start_epoch:
                    history["history"] = history["history"][:start_epoch]
            
        return cls(root=root, path=history_log_path, history=history)

    def log_history(self, stat: Stat) -> str:
        """log history for the statistics coming from new epoch

        Args:
            stat (Stat): statistics data

        Returns:
            str: path to the log file history.json
        """
        self.history["history"].append(vars(stat))
        self.history["root"] = self.root

        os.makedirs(self.root, exist_ok=True)
        with open(self.path, "w") as fout:
            json.dump(self.history, fout, indent=4)
        
        return self.path
    
    def plot(self):
        HistoryUtils.plot_history(self.path, self.root)
        return

    @staticmethod
    def _plot(title: str, train_data: list, valid_data: list, output_dir: str) -> str:
        plt.figure(figsize=(10,5))
        plt.plot(train_data, label="train")
        plt.plot(valid_data, label="valid")
        plt.title(title)
        plt.xlabel("epochs")
        plt.legend()
        path = os.path.join(output_dir, title.lower())
        plt.savefig(path)
        plt.show()
        return path


    @staticmethod
    def plot_history(history_path: str, output_dir: str):
        """plot the loss-epoch figure

        Args:
            history_path (str): file of history (in json)
            output_dir (str): dir to export the result figure
        """
        with open(history_path, "r") as fin:
            history: History = json.load(fin)
        
        df = pd.DataFrame(history["history"])
        HistoryUtils._plot("Loss", df["train_loss"].tolist(), df["valid_loss"].tolist(), output_dir)
        HistoryUtils._plot("Accuracy", df["train_acc"].tolist(),
                            df["valid_acc"].tolist(), output_dir)
        return
    

class BaseModelUtils:

    history: History
    logger: Logger # TODO make some msg print to log

    def __init__(self, model: nn.Module, config: BaseConfig, optimizer: Optimizer = None):

        self.model = model
        self.model.to(config.DEVICE)
        self.config = config

        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.start_epoch = 0

        # init for history and log
        time_str = formatted_now()
        self.root = os.path.join(self.config.LOG_DIR, time_str)
        self.history_utils = HistoryUtils(root=self.root)
        # self.logger = ? TODO
        return

    @classmethod
    def load_checkpoint(cls, model: nn.Module, checkpoint_path: str,
                        config: BaseConfig, optimizer: Optimizer = None):
        """init ModelUtils class with the saved model (or checkpoint)

        Args:
            model (nn.Module): model architecture
            checkpoint_path (str): path of saved model (or checkpoint)
            config (BaseConfig): config

        """

        assert os.path.isfile(checkpoint_path)

        tem = torch.load(checkpoint_path)
        checkpoint = ModelStates(**tem)

        new = BaseModelUtils(model, config, optimizer)

        new.model.load_state_dict(checkpoint.model_state_dict)
        new.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        new.start_epoch = checkpoint.start_epoch
        
        new.root = os.path.dirname(checkpoint_path)
        new.logger = Logger(new.root)
        new.logger.reset_log_file_root(new.root)
        new.history_utils = HistoryUtils.load_history(new.root, new.start_epoch, new.logger)
        new.logger.log(f"Checkpoint {os.path.basename(checkpoint_path)} is loaded.")
        return new

    @classmethod
    def load_last_checkpoint_from_dir(cls, model: nn.Module, config: BaseConfig,
                                    dir_path: str, optimizer: Optimizer = None):

        PATTERN = r".+?_epoch_(\d+)"
        max_epoch = 0
        max_idx = 0
        save_list = os.listdir(dir_path)
        for idx, save in enumerate(save_list):
            match = re.match(PATTERN, save)
            if match:
                epoch = int(match.group(1))
                if epoch > max_epoch:
                    max_epoch = epoch
                    max_idx = idx
        

        last_save = save_list[max_idx]

        last_save_path = os.path.join(dir_path, last_save)
        print(f"Try loading: {last_save_path}")

        return cls.load_checkpoint(
            model,
            checkpoint_path=last_save_path,
            config=config,
            optimizer=optimizer,
        )

    @classmethod
    def load_last_checkpoint(cls, model: nn.Module, config: BaseConfig, optimizer: Optimizer = None):

        TIME_FORMAT_PATTERN = r"^\d{8}T\d{2}-\d{2}-\d{2}"
        def is_timeformatted_not_empty(name: str) -> bool:
            """check whether a name of dir is start with formatted time and not empty

            E.g:
                - [v] 20220330T16-31-29_some_addtion 
                - [x] ResNet_1 
            """
            match = re.match(TIME_FORMAT_PATTERN, name)
            if not match:
                return False
            
            path = os.path.join(config.LOG_DIR, name)
            if len(os.listdir(path)) == 0: # if empty
                os.removedirs(path)
                return False
            return True

        arr = [dir_name for dir_name in os.listdir(config.LOG_DIR)
                                            if is_timeformatted_not_empty(dir_name)]

        last_train_root = max(arr)
        last_train_root = os.path.join(config.LOG_DIR, last_train_root)
        return cls.load_last_checkpoint_from_dir(
            model,
            dir_path=last_train_root,
            config=config,
            optimizer=optimizer,
        )
            



    def _save(self, cur_epoch: int, stat: Stat) -> str:
        tem = vars(ModelStates(
            start_epoch = cur_epoch + 1,
            model_state_dict = self.model.state_dict(),
            optimizer_state_dict = self.optimizer.state_dict(),
            stat = vars(stat),
        ))
        now = formatted_now()
        
        name = f"{now}_epoch_{cur_epoch + 1}"
        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, name)
        torch.save(tem, path)
        print(f"Checkpoint: {name} is saved.")
        self.history_utils.history["checkpoints"][cur_epoch + 1] = name
        return name
    

    def _train_epoch(self, train_dataset: Dataset) -> Tuple[float, float]:
        """train a single epoch

        Returns:
            Tuple[float, float]: train_loss, train_acc
        """
        raise NotImplementedError
    
    def _eval_epoch(self, eval_dataset: Dataset) -> Tuple[float, float]:
        """evaluate single epoch

        Returns:
            Tuple[float, float]: eval_loss, eval_acc
        """
        raise NotImplementedError
    
    def train(self, epochs: int, trainset: Dataset, validset: Dataset, testset: Dataset) -> str:
        """start training

        Args:
            epochs (int): defalut to None, if None. train to the epochs store in checkpoint.
            Specify to change the target epochs

        Returns:
            str: json path as the history
        """

        assert epochs > self.start_epoch,\
            f"expect epochs > {self.start_epoch}, got: epochs={epochs}"
        
        # counting for early stopping
        min_valid_loss = 999999
        counter = 0

        for epoch in range(self.start_epoch, epochs):
            print(f"Epoch: {epoch + 1} / {epochs}")
            train_loss, train_acc = self._train_epoch(trainset)
            valid_loss, valid_acc = self._eval_epoch(validset)

            stat = Stat(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_acc=train_acc,
                valid_loss=valid_loss,
                valid_acc=valid_acc,
            )
            stat.display()
            if valid_loss >= min_valid_loss:
                counter += 1
                if self.config.EARLY_STOPPING:
                    print(f"Early stopping counter:\
                            {counter} / {self.config.EARLY_STOPPING_THRESHOLD}")
                    
                    if counter == self.config.EARLY_STOPPING_THRESHOLD:
                        print("Early stopping!")
                        self._save(epoch, stat)
                        break
                else:
                    print(f"Early stopping counter: {counter} / infinity")
            else:
                min_valid_loss = valid_loss
                counter = 0
            
            if epoch == epochs - 1:
                self._save(epoch, stat)
            
            elif (
                self.config.EPOCHS_PER_CHECKPOINT
                and (epoch + 1) % self.config.EPOCHS_PER_CHECKPOINT == 0
            ):
                self._save(epoch, stat)

            if epoch != epochs - 1:
                self.history_utils.log_history(stat)

        print("Training is finish")
        test_loss, test_acc = self._eval_epoch(testset)
        stat.test_loss = test_loss
        stat.test_acc = test_acc
        stat.display()
        return self.history_utils.log_history(stat)
    
    
    def inference(self, *args, **kwargs):
        raise NotImplementedError
        
    
    def plot_history(self):
        self.history_utils.plot()


def formatted_now():
    return datetime.now().strftime("%Y%m%dT%H-%M-%S")
