Model Utils for training model with Pytorch.

**Working** on doc.


## Guides

### Config

#### basic

#### checking hook

#### mutable

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
