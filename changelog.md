# Change Log

### 0.3.2
    - Now only member configs set with `Field(direct_access=True)` will get direct access
        - Set int value to set the priority. `Field(direct_access=<integer>)`
        - Lower number has higher priority.


### 0.3.1
    - remove mutable field  
    - allow recursive get config value
        - E.g. config = Config(sub_config=SubConfig(a=1), b=2)
        - assert config.a == 1
        - recursive only works in the members those are instance of BaseConfig
### 0.3.0


### 0.2.1

- support nested display
- support save_n_best
- remove config `early_stopping` and `early_stopping_threshold`, use `early_stopping_rounds`
instead.

### 0.2.0
- adapt pydantic
- deprecate NamespaceDict
- change return type of `_train_epoch` and `_eval_epoch` from `Criteria` to
`Union[_Criterion, Tuple[_Criterion, ...]]`
- revise factories of BaseModelUtils
    - load_last_checkpoint_from_dir --> _load_last_checkpoint_from_dir
    - now use load_last_checkpoint(dir_path =...) instead


### 0.1.6

- support load_best_checkpoint
- add checkpoint_info to History
- fix bug of get_logger
- both prefix and suffix can work with time-formatted name

### 0.1.5

- update logger relatives
- make checking hooks only work on subclasses

- _get_scheduler no longer need to load_state_dict manually


### 0.1.4

- support scheduler

- refine register_criteria usage

- now it's not necessary to eval every epoch (refer to `epoch_per_eval` config)

- add mutable feature (unstable)

- fix minor bug

#### 0.1.4.1

- fix minor

---

### 0.1.3

- now `_train_epoch` and `_eval_epoch` return object `Criteria` instead of `Union[float, Tuple[float, float]]`

- support logging and ploting multi-loss (multi-criterion)

#### 0.1.3.1

- minor fix

#### 0.1.3.2

- add load_config method

### 0.1.2

- support for model that don't use accuacy.

- introduce `register_checking_hook` for `BaseConfig`