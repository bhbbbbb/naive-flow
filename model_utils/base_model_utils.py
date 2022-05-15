import os
import re
from datetime import datetime
from argparse import Namespace
from typing import Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset

from .config import ModelUtilsConfig
from .base.history import HistoryUtils, Stat
from .base.logger import Logger
from .base.criteria import Criteria
from .base.early_stopping_handler import EarlyStoppingHandler


class ModelStates(Namespace):
    # epoch to start from (0 is the first)
    start_epoch: int

    config: ModelUtilsConfig

    ########## torch built-in model states ############
    model_state_dict: dict
    optimizer_state_dict: dict
    scheduler_state_dict: dict

    # statistic of the last epoch
    stat: Stat

    

class BaseModelUtils:
    """Base ModelUtils"""

    model: nn.Module
    config: ModelUtilsConfig
    optimizer: Optimizer
    scheduler: _LRScheduler
    start_epoch: int
    root: str
    history_utils: HistoryUtils
    logger: Logger

    def __init__(
            self,
            model: nn.Module,
            config: ModelUtilsConfig,
            optimizer: Optimizer,
            scheduler: _LRScheduler,
            start_epoch: int,
            root: str,
            history_utils: HistoryUtils,
            logger: Logger,
        ):

        self.model = model
        self.model.to(config.device)
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.start_epoch = start_epoch
        self.root = root
        self.history_utils = history_utils
        self.logger = logger

        # log information
        print(model, file=logger.file)
        print(optimizer, file=logger.file)
        config.display(logger.file)
        return

    @staticmethod
    def _get_optimizer(model: nn.Module, config: ModelUtilsConfig) -> Optimizer:
        raise NotImplementedError

    @staticmethod
    def _get_scheduler(
        optimizer: Optimizer,
        config: ModelUtilsConfig,
        state_dict: Union[dict, None],
    ) -> _LRScheduler:
        """Define how to get scheduler, default returning None, which is equivalent to use constant
            learning rate.

        Args:
            optimizer (Optimizer): optimizer that return by `_get_optimizer`
            config (ModelUtilsConfig): config
            state_dict (dict, None): if resume by checkpoint, the state_dict is the value return
                by `scheduler.get_state_dict()`. Otherwise, state_dict is None.

        Returns:
            _LRScheduler: scheduler to use. return None to use constant learning rate.
        """
        # pylint: disable=unused-argument
        return None

    @classmethod
    def start_new_training(cls, model: nn.Module, config: ModelUtilsConfig):
        
        optimizer = cls._get_optimizer(model, config)
        scheduler = cls._get_scheduler(optimizer, config, None)
        # init for history and log
        time_str = formatted_now()
        root = os.path.join(config.log_dir, time_str)
        os.makedirs(root, exist_ok=True)
        history_utils = HistoryUtils(root=root)
        logger = Logger(root) if config.logging else Logger()
        
        return cls(
            model = model,
            config = config,
            optimizer = optimizer,
            scheduler = scheduler,
            start_epoch = 0,
            root = root,
            history_utils = history_utils,
            logger = logger,
        )

    @staticmethod
    def load_config(checkpoint_path: str, as_dict: bool = False) -> Union[ModelUtilsConfig, dict]:
        """load config from the given checkpoint

        Args:
            checkpoint_path (str): path to the given checkpoint

        Returns:
            Config: config
        """
        
        assert os.path.isfile(checkpoint_path)

        tem = torch.load(checkpoint_path)
        checkpoint = ModelStates(**tem)
        if as_dict:
            return checkpoint.config

        return ModelUtilsConfig(**checkpoint.config)

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
        scheduler = cls._get_scheduler(optimizer, config, checkpoint.scheduler_state_dict)
        
        root = os.path.dirname(checkpoint_path)
        logger = Logger(root) if config.logging else Logger()
        start_epoch = checkpoint.start_epoch
        history_utils = HistoryUtils.load_history(root, start_epoch, logger)
        logger.log(f"Checkpoint {os.path.basename(checkpoint_path)} is loaded.")
        return cls(
            model = model,
            config = config,
            optimizer = optimizer,
            scheduler = scheduler,
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

        assert config.log_dir is not None, (
            "when log_dir is set to None, load_last_checkpoint is not available"
        )

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
            scheduler_state_dict = self.scheduler.state_dict(),
            config = self.config.asdict(),
            stat = stat.asdict(),
        ))
        now = formatted_now()
        
        name = f"{now}_epoch_{cur_epoch + 1}"
        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, name)
        torch.save(tem, path)
        self.logger.log(f"Checkpoint: {name} is saved.")
        self.history_utils.history.checkpoints[cur_epoch + 1] = name
        return name
    

    def _train_epoch(self, train_dataset: Dataset) -> Criteria:
        """train a single epoch

        Returns:
            Criteria: train_criteria
        """
        raise NotImplementedError
    
    def _eval_epoch(self, eval_dataset: Dataset) -> Criteria:
        """evaluate single epoch

        Returns:
            Criteria: eval_criteria
        """
        raise NotImplementedError
    
    def train(self, epochs: int, trainset: Dataset, validset: Dataset = None,
                testset: Dataset = None) -> str:
        """start training

        Args:
            epochs (int): defalut to None, if None. train to the epochs store in checkpoint.
            Specify to change the target epochs
            validset (Dataset): Optional but unlike testset it is not supposed to be omit,
                unless you are testing your model by overfit it or something else.
            testset (Dataset): Optional.

        Returns:
            str: json path as the history
        """

        assert epochs > self.start_epoch,\
            f"expect epochs > {self.start_epoch}, got: epochs={epochs}"
        
        if validset is None:
            self.logger.warning(
                "Warning: You are Not passing the validset\n"
                "make sure you know what yor are doing."
            )

        es_handler = EarlyStoppingHandler(self.config)

        for epoch in range(self.start_epoch, epochs):

            self.logger.log(f"Epoch: {epoch + 1} / {epochs}")
            train_criteria = self._train_epoch(trainset)
            
            valid_criteria = None
            if (
                validset is not None
                and (epoch + 1 - self.start_epoch) % self.config.epochs_per_eval == 0
            ):
                valid_criteria = self._eval_epoch(validset)

            stat = Stat(
                epoch=epoch + 1,
                train_criteria=train_criteria,
                valid_criteria=valid_criteria,
            )
            stat.display()

            if es_handler.should_stop(valid_criteria):
                self.logger.log("Early stopping!")
                self._save(epoch, stat)
                break

            if epoch == epochs - 1:
                self._save(epoch, stat)
            
            elif (
                self.config.epochs_per_checkpoint
                and (epoch + 1 - self.start_epoch) % self.config.epochs_per_checkpoint == 0
                or es_handler.should_save_best()
            ):
                self._save(epoch, stat)

            if epoch != epochs - 1:
                self.history_utils.log_history(stat)

        self.logger.log(f"Training is finish for epochs: {epochs}")
        if testset is not None:
            stat.test_criteria = self._eval_epoch(testset)
            stat.display()
        
        return self.history_utils.log_history(stat)
    
    
    def plot_history(
        self,
        show: bool = False,
        save: bool = True,
        plot_configs = None,
    ):
        self.history_utils.plot(
            show=show,
            save=save,
            plot_configs=plot_configs,
        )
        return
    
    @staticmethod
    def get_default_plot_configs():
        return Criteria.get_plot_configs_from_registered_criterion()


def formatted_now():
    return datetime.now().strftime("%Y%m%dT%H-%M-%S")
