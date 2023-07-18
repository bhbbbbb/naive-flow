import os
import re
from datetime import datetime
from argparse import Namespace
from typing import Union, Optional
import logging

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset

from .config import ModelUtilsConfig
from .base.history import HistoryUtils, Stat
from .base.logger import get_logger
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
    """Base Model Utilities for training a given model

    - Predefined utils for training given model,
        including features (should be properly configured):
        1. Auto logging
        1. Auto saving checkpoints
        1. Auto Plotting
        1. Early Stopping
        1. K-fold Cross Validatoin

    ## Abstract Methods

    There are four member methods that are abstract and required overreided, which are

    1. `_get_optimizer` &mdash; define how and what the optimizer to use.

    ```
    @staticmethod
    def _get_optimizer(model: nn.Module, config: ModelUtilsConfig) ->\
        torch.optim.Optimizer: ...
    ```

    2. `_get_scheduler` &mdash; define how and what the scheduler to use,
    can be omitted to use no scheduler.

    ```
    @staticmethod
    def _get_scheduler(
        optimizer: Optimizer,
        config: ModelUtilsConfig,
    ) -> Optional[_LRScheduler]:
    ```

    3. `_train_epoch` &mdash; implement method for training a **single** epoch.
    Notice that the actual training process `train` has been predefined to enable
    the features going to be mentioned. Thus `train` should not be implemented manually,
    and `_train_epoch` method would be called in the method `train`.

    ```
    def _train_epoch(self, train_dataset) -> Criteria: ...
    ```

    4. `_eval_epoch` &mdash; implement method for evaluating a **single** epoch.

    ```
    def _eval_epoch(self, eval_dataset) -> Criteria: ...
    ```

        
    ---

    ## Predefined Attributes

    - `train` &mdash; start training
    - `plot_history` &mdash; plot the history


    ---

    ## Predefined Classmethods

    - `load_checkpoint` &mdash; load from specific checkpoint by its path.
    - `load_last_checkpoint` &mdash; load latest saved checkpoint automatically.
    - `load_last_checkpoint_from_dir` &mdash; load latest checkpoint saved in given directory.

    ---

    ## Advanced

    TODO

    """

    model: nn.Module
    config: ModelUtilsConfig
    optimizer: Optimizer
    scheduler: _LRScheduler
    start_epoch: int
    root: str
    history_utils: HistoryUtils
    logger: logging.Logger

    def __init__(
            self,
            model: nn.Module,
            config: ModelUtilsConfig,
            optimizer: Optimizer,
            scheduler: _LRScheduler,
            start_epoch: int,
            root: str,
            history_utils: HistoryUtils,
            logger: logging.Logger,
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
        self.logger.debug(model)
        self.logger.debug(optimizer)
        self.logger.debug(config)
        return

    @staticmethod
    def _get_optimizer(model: nn.Module, config: ModelUtilsConfig) -> Optimizer:
        raise NotImplementedError

    @staticmethod
    def _get_scheduler(
        optimizer: Optimizer,
        config: ModelUtilsConfig,
    ) -> Optional[_LRScheduler]:
        """Define how to get scheduler.
        Args:
            optimizer (Optimizer): optimizer that return by `_get_optimizer`
            config (ModelUtilsConfig): config

        Returns:
            _LRScheduler: scheduler to use. return None to train without scheduler.
        """
        # pylint: disable=unused-argument
        return None

    @classmethod
    def start_new_training(cls, model: nn.Module, config: ModelUtilsConfig):
        
        optimizer = cls._get_optimizer(model, config)
        scheduler = cls._get_scheduler(optimizer, config)
        # init for history and log
        rootname = formatted_now()
        root = os.path.join(config.log_dir, rootname)
        os.makedirs(root, exist_ok=True)
        history_utils = HistoryUtils(root=root)
        logger = get_logger(__name__, root) if config.logging else get_logger(__name__)
        
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
    def load_checkpoint(
        cls, model: nn.Module, checkpoint_path: str, config: ModelUtilsConfig = None
    ):
        """init ModelUtils class with the saved model (or checkpoint)

        Args:
            model (nn.Module): model architecture
            checkpoint_path (str): path of saved model (or checkpoint)
            config (ModelUtilsConfig): config

        """

        assert os.path.isfile(checkpoint_path), (
            f"expect checkpoint_path: '{checkpoint_path}' is file."
        )

        tem = torch.load(checkpoint_path)
        checkpoint = ModelStates(**tem)
        config = config or ModelUtilsConfig(**checkpoint.config)

        model.load_state_dict(checkpoint.model_state_dict)
        model.to(config.device)
        optimizer = cls._get_optimizer(model, config)
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)

        scheduler = cls._get_scheduler(optimizer, config)
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint.scheduler_state_dict)
        
        input_root = os.path.dirname(checkpoint_path)
        rootname = os.path.basename(input_root)
        root = os.path.join(config.log_dir, rootname)
        os.makedirs(root, exist_ok=True)
        logger = get_logger(__name__, root) if config.logging else get_logger(__name__)
        start_epoch = checkpoint.start_epoch
        history_utils = HistoryUtils.load_history(input_root, root, start_epoch, logger)
        logger.info(f"Checkpoint {os.path.basename(checkpoint_path)} is loaded.")
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
        """Load latest checkpoint saved in given directory

        Args:
            model (nn.Module): _description_
            dir_path (str): path to the directory.
            config (ModelUtilsConfig, optional): _description_. Defaults to None.
            If omitted, automatically use the configurations saved in checkpoint.

        Returns:
            ModelUtils
        """

        PATTERN = r".+?_epoch_(\d+)"
        max_epoch = 0
        max_idx = -1
        save_list = os.listdir(dir_path)
        for idx, save in enumerate(save_list):
            match = re.match(PATTERN, save)
            if match:
                epoch = int(match.group(1))
                if epoch > max_epoch:
                    max_epoch = epoch
                    max_idx = idx
        
        assert max_idx >= 0, f"cannot find any checkpoint in dir: '{dir_path}'"

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
        """Load latest saved checkpoint automatically

        Args:
            model (nn.Module): _description_
            config (ModelUtilsConfig): _description_

        Returns:
            ModelUtils
        """

        TIME_FORMAT_PATTERN = r"^\d{8}T\d{2}-\d{2}-\d{2}"
        def is_timeformatted_dir(name: str) -> bool:
            """check whether a name of dir is start with formatted time and not empty

            E.g:
                - [v] 20220330T16-31-29_some_addtion 
                - [x] ResNet_1 
            """
            match = re.match(TIME_FORMAT_PATTERN, name)
            if not match:
                return False
            
            path = os.path.join(config.log_dir, name)
            return os.path.isdir(path)

        arr = [dir_name for dir_name in os.listdir(config.log_dir)
                                            if is_timeformatted_dir(dir_name)]

        last_train_root = max(arr)
        last_train_root = os.path.join(config.log_dir, last_train_root)
        return cls.load_last_checkpoint_from_dir(
            model,
            dir_path=last_train_root,
            config=config,
        )

    def _save(self, cur_epoch: int, stat: Stat) -> str:
        scheduler_dict = self.scheduler.state_dict() if self.scheduler else {}
        tem = vars(ModelStates(
            start_epoch = cur_epoch + 1,
            model_state_dict = self.model.state_dict(),
            optimizer_state_dict = self.optimizer.state_dict(),
            scheduler_state_dict = scheduler_dict,
            config = self.config.asdict(),
            stat = stat.asdict(),
        ))
        now = formatted_now()
        
        name = f"{now}_epoch_{cur_epoch + 1}"
        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, name)
        torch.save(tem, path)
        self.logger.info(f"Checkpoint: {name} is saved.")
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
    
    def train(self, epochs: int, train_set: Dataset, valid_set: Dataset = None,
                test_set: Dataset = None) -> str:
        """start training

        Args:
            epochs (int): defalut to None, if None. train to the epochs store in checkpoint.
            Specify to change the target epochs
            valid_set (Dataset): Optional but unlike testset it is not supposed to be omit,
                unless you are testing your model by overfit it or something else.
            test_set (Dataset): Optional.

        Returns:
            str: json path as the history
        """

        assert epochs > self.start_epoch,\
            f"expect epochs > {self.start_epoch}, got: epochs={epochs}"
        
        if valid_set is None:
            self.logger.warning(
                "Warning: You are Not passing the valid_set\n"
                "make sure you know what yor are doing."
            )

        es_handler = EarlyStoppingHandler(self.config)

        for epoch in range(self.start_epoch, epochs):

            self.logger.info(f"Epoch: {epoch + 1} / {epochs}")
            train_criteria = self._train_epoch(train_set)
            
            valid_criteria = None
            if (
                valid_set is not None
                and (epoch + 1 - self.start_epoch) % self.config.epochs_per_eval == 0
            ):
                valid_criteria = self._eval_epoch(valid_set)

            stat = Stat(
                epoch=epoch + 1,
                train_criteria=train_criteria,
                valid_criteria=valid_criteria,
            )
            stat.display()

            if es_handler.should_stop(valid_criteria):
                self.logger.info("Early stopping!")
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

        self.logger.info(f"Training is finish for epochs: {epochs}")
        if test_set is not None:
            stat.test_criteria = self._eval_epoch(test_set)
            stat.display()
        
        self.start_epoch = epochs
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
