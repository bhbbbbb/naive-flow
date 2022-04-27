# Change Log

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