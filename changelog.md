# Change Log


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