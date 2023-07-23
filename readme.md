Model Utils for training model with Pytorch.

## Install

```sh
pip install git+https://github.com/bhbbbbb/pytorch-model-utils
```


## Guides

TODO

<!--

### Basic Usage

- example

```python
from torch import nn
from torch.utils.data import Dataset
from my_config import MyConfig
from my_model_utils import MyModelUtils

def main():

    # see section Config for more example and explanation
    config = MyConfig()

    # config is still mutable at this time

    config.check_and_freeze()

    # run all registered checking hooks, check the implemtation of the configurations
    # markes as UNIMPLEMENT, and make config no longer mutable

    config.display()
    # print all your configurations

    model: nn.Module = get_model(...)

    # -------------choose one to initialize utils--------------

    # start new training process
    utils = MyModelUtils.start_new_training(model, config)

    # resume from last training
    utils = MyModelUtils.load_last_checkpoint(model, config)

    # load from particular checkpoint
    path = '/path/to/checkpoint/'
    utils = MyModelUtils.load_checkpoint(model, path, config)

    # see Section ModelUtils for more information

    # ---------------------------------------------------------

    train_set: Dataset = get_train_dataset(...)
    valid_set: Dataset = get_valid_dataset(...)
    test_set: Dataset = get_test_data_set(...)

    epochs = 100

    utils.train(epochs, train_set, valid_set, test_set)
    # notice that both valid_set and test_set are optional

    utils.plot_history()
    # for visualization
```



### Config

- example

```python
import os
from typing import Dict

from model_utils.base import UNIMPLEMENTED, NOT_NECESSARY

class DatasetConfig(BaseConfig):


    TRAIN_SET_PATH: str = UNIMPLEMENTED
    VALID_SET_PATH: str = UNIMPLEMENTED
    TEST_SET_PATH: str = UNIMPLEMENTED

    use_mixed_region: bool = UNIMPLEMENTED


    # ---------- DataLoader -------------------
    batch_size: Dict[Mode, int] = UNIMPLEMENTED

    num_workers: int = 4 if os.name == 'nt' else 2

    persistent_workers: bool = (os.name == 'nt')

    pin_memory: bool = True

    drop_last: bool = True

@DatasetConfig.register_checking_hook
def check_worker_setup(config: DatasetConfig):
    print(
        'check for presistent_workers, num_workers'
        f' = {config.persistent_workers}, {config.num_workers}'
    )
    if config.persistent_workers:
        assert config.num_workers > 0
    
```

### ModelUtils

- basic example

```python
from model_utils import BaseModelUtils, Criteria, Loss

class MyModelUtils(BaseModelUtils):


    @staticmethod
    def _get_optimizer(model: nn.Module, config: Config):
        """Must be overrided to define the way to get optimizer"""
        return get_optimizer(
            model,
            config.optimizer_name,
            config.learning_rate,
            config.weight_decay,
        )
    
    @staticmethod
    def _get_scheduler(optimizer, config: Config, state_dict: dict):
        """Define how to get scheduler.
        Args:
            optimizer (Optimizer): optimizer that return by `_get_optimizer`
            config (ModelUtilsConfig): config

        Returns:
            _LRScheduler: scheduler to use. return None to train without scheduler.
        """
        scheduler = get_scheduler(
            config.scheduler_name,
            optimizer,
            config.max_iters,
            config.power,
            config.warmup_iters,
            config.warmup_ratio,
        )
        if state_dict is not None:
            scheduler.load_state_dict(state_dict)
        return scheduler

    def _train_epoch(self, train_dataset: MyDataset) -> Criteria:
        """Must be overrided"""
        
        self.model.train()

        train_loss = 0.0
        idx = 0
        pbar = tqdm(train_dataset.dataloader, disable=(not self.config.show_progress_bar))

        for img, lbl in pbar:
            img: Tensor
            lbl: Tensor
            self.optimizer.zero_grad(set_to_none=True)

            img = img.to(self.config.device)
            lbl = lbl.to(self.config.device)
            
            with autocast(enabled=self.config.AMP):
                logits: Tensor = self.model(img)
                loss: Tensor = self.loss_fn(logits, lbl)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            torch.cuda.synchronize()

            lr = self.scheduler.get_lr()
            lr = sum(lr) / len(lr)
            running_loss = loss.item()
            train_loss += running_loss

            pbar.set_description(f'LR: {lr:.4e} Running Loss: {running_loss:.6f}')
            idx += 1
        
        self.loss_fn.step()
        train_loss /= idx
        torch.cuda.empty_cache()
        return Criteria(Loss(train_loss))

    
    @torch.no_grad()
    def _eval_epoch(self, eval_dataset: MyDataset) -> Criteria:
        """Must be overrided"""
        self.model.eval()

        for ... in eval_dataset.dataloader:
            ...
        
        return Criteria(...)
```

### Criteria

Criteria is an interface that allow you to use multi-criterion easily.

```python
from model_utils import Criteria, Loss, Accuarcy

class MyModelUtils(...):

    ...

    def _train_epoch(self, train_dataset: MyDataset) -> Criteria:
        """Must be overrided"""
        
        self.model.train()

        train_loss = ...
        train_accuarcy = ...

        for ... in train_dataset.dataloader:
            ...
            train_loss = ...
            train_accuarcy = ...
        
        return Criteria(
            Loss(train_loss),
            Accuarcy(train_accuracy),
        )

```

- example for customized Criterion

```python
from model_utils import Criteria, Loss as BaseLoss, Accuarcy as BaseAcc

Loss = Criteria.register_criterion(
    short_name='loss',
    full_name='Loss',
    primary=False,
    plot=True,
    default_lower_limit_for_plot=0.0,
    default_upper_limit_for_plot=1.0,
)(BaseLoss)
```

this is equivalent to following

```python
@Criteria.register_criterion()
class Loss(BaseLoss):
    short_name: str = 'loss' # short name that have to be unique
    full_name: str = 'Loss' # name for display

    plot: bool = False # whether be plot when ModelUtils.plot(...) is called

    default_lower_limit_for_plot: float = 0.0
    default_upper_limit_for_plot: float = 1.0

    primary: bool = False
    """
    Whether this criterion considered as primary.
    Since the early stopping feature have to compare between different Criteria
    and each Criteria may have multiple criterion.

    thus you have to set exact one Criterion to primary if enabling early stopping feature.
    """
```

- advance

```python
from model_utils.base.criteria import _Criterion
class MyCriterion(_Criterion):

    short_name: str
    full_name: str

    plot: bool
    """Whether plot this criterion"""

    default_lower_limit_for_plot: float
    default_upper_limit_for_plot: float

    value: float
    primary: bool
    """whether be used as primary criterion"""

    def __init__(self, value: float):
        self.value = value
        return
    
    def better_than(self, rhs: _Criterion) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self.value)
```


-->