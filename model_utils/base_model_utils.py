import os
import re
from typing import Tuple
from datetime import datetime
from argparse import Namespace
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import Dataset
from .config import ModelUtilsConfig
from .history import HistoryUtils, Stat
from ..logger import Logger


class ModelStates(Namespace):
    # epoch to start from (0 is the first)
    start_epoch: int

    config: ModelUtilsConfig

    ########## torch built-in model states ############
    model_state_dict: dict
    optimizer_state_dict: dict

    # statistic of the last epoch
    stat: Stat

    

class BaseModelUtils:
    """Base ModelUtils"""

    model: nn.Module
    config: ModelUtilsConfig
    optimizer: Optimizer
    criterion: nn.Module
    start_epoch: int
    root: str
    history_utils: HistoryUtils
    logger: Logger

    def __init__(
            self,
            model: nn.Module,
            config: ModelUtilsConfig,
            optimizer: Optimizer,
            start_epoch: int,
            root: str,
            history_utils: HistoryUtils,
            logger: Logger,
        ):

        self.model = model
        self.model.to(config.device)
        self.config = config
        self.optimizer = optimizer
        self.start_epoch = start_epoch
        self.root = root
        self.history_utils = history_utils
        self.logger = logger
        self.criterion = self._get_criterion(config)

        # log information
        print(model, file=logger)
        print(optimizer, file=logger)
        config.display(logger)
        return

    @staticmethod
    def _get_optimizer(model: nn.Module, config: ModelUtilsConfig) -> Optimizer:
        raise NotImplementedError

    @staticmethod
    def _get_criterion(config: ModelUtilsConfig) -> nn.Module:
        raise NotImplementedError

    @classmethod
    def start_new_training(cls, model: nn.Module, config: ModelUtilsConfig):
        
        optimizer = cls._get_optimizer(model, config)
        # init for history and log
        time_str = formatted_now()
        root = os.path.join(config.log_dir, time_str)
        os.makedirs(root, exist_ok=True)
        history_utils = HistoryUtils(root=root)
        logger = Logger(root)
        
        return cls(
            model = model,
            config = config,
            optimizer = optimizer,
            start_epoch = 0,
            root = root,
            history_utils = history_utils,
            logger = logger,
        )

    @classmethod
    def load_checkpoint(cls, model: nn.Module, checkpoint_path: str,
                        config: ModelUtilsConfig = None):
        """init ModelUtils class with the saved model (or checkpoint)

        Args:
            model (nn.Module): model architecture
            checkpoint_path (str): path of saved model (or checkpoint)
            config (ModelUtilsConfig): config

        """

        assert os.path.isfile(checkpoint_path)

        tem = torch.load(checkpoint_path)
        checkpoint = ModelStates(**tem)
        config = config or ModelUtilsConfig(**checkpoint.config)

        model.load_state_dict(checkpoint.model_state_dict)
        model.to(config.device)
        optimizer = cls._get_optimizer(model, config)
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        
        root = os.path.dirname(checkpoint_path)
        logger = Logger(root)
        start_epoch = checkpoint.start_epoch
        history_utils = HistoryUtils.load_history(root, start_epoch, logger)
        logger.log(f"Checkpoint {os.path.basename(checkpoint_path)} is loaded.")
        return cls(
            model = model,
            config = config,
            optimizer = optimizer,
            start_epoch = start_epoch,
            root = root,
            history_utils = history_utils,
            logger = logger,
        )

    @classmethod
    def load_last_checkpoint_from_dir(cls, model: nn.Module, dir_path: str,
                config: ModelUtilsConfig = None):

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
        )

    @classmethod
    def load_last_checkpoint(cls, model: nn.Module, config: ModelUtilsConfig):

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
            
            path = os.path.join(config.log_dir, name)
            if len(os.listdir(path)) == 0: # if empty
                os.removedirs(path)
                return False
            return True

        arr = [dir_name for dir_name in os.listdir(config.log_dir)
                                            if is_timeformatted_not_empty(dir_name)]

        last_train_root = max(arr)
        last_train_root = os.path.join(config.log_dir, last_train_root)
        return cls.load_last_checkpoint_from_dir(
            model,
            dir_path=last_train_root,
            config=config,
        )

    def _save(self, cur_epoch: int, stat: Stat) -> str:
        tem = vars(ModelStates(
            start_epoch = cur_epoch + 1,
            model_state_dict = self.model.state_dict(),
            optimizer_state_dict = self.optimizer.state_dict(),
            config = dict(self.config),
            stat = vars(stat),
        ))
        now = formatted_now()
        
        name = f"{now}_epoch_{cur_epoch + 1}"
        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, name)
        torch.save(tem, path)
        self.logger.log(f"Checkpoint: {name} is saved.")
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
        highest_valid_acc = 0.0
        counter = 0
        saved = False

        for epoch in range(self.start_epoch, epochs):
            self.logger.log(f"Epoch: {epoch + 1} / {epochs}")
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

            saved = False

            if valid_acc <= highest_valid_acc:
                counter += 1
                if self.config.early_stopping:
                    print("Early stopping counter:"
                            f"{counter} / {self.config.early_stopping_threshold}")
                    
                    if counter == self.config.early_stopping_threshold:
                        self.logger.log("Early stopping!")
                        self._save(epoch, stat)
                        break
                else:
                    print(f"Early stopping counter: {counter} / infinity")
            else:
                highest_valid_acc = valid_acc
                counter = 0
                if self.config.save_best:
                    self._save(epoch, stat)
                    saved = True
            
            print(f"Current best valid_acc: {highest_valid_acc * 100 :.2f} %")
            if not saved:
                if epoch == epochs - 1:
                    self._save(epoch, stat)
                
                elif (
                    self.config.epochs_per_checkpoint
                    and (epoch + 1 - self.start_epoch) % self.config.epochs_per_checkpoint == 0
                    and not saved
                ):
                    self._save(epoch, stat)

            if epoch != epochs - 1:
                self.history_utils.log_history(stat)

        self.logger.log(f"Training is finish for epochs: {epochs}")
        test_loss, test_acc = self._eval_epoch(testset)
        stat.test_loss = test_loss
        stat.test_acc = test_acc
        stat.display()
        return self.history_utils.log_history(stat)
    
    
    def plot_history(self, loss_uplimit: float = None, acc_autoscale: bool = False,
                        show: bool = False, save: bool = True):
        self.history_utils.plot(
            loss_uplimit=loss_uplimit,
            acc_autoscale=acc_autoscale,
            show=show,
            save=save,
        )
        return


def formatted_now():
    return datetime.now().strftime("%Y%m%dT%H-%M-%S")
