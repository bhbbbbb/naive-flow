# Change Log

### 0.3.9

    - Add options to tracker_config
        - Add progress: Literal['tqdm', 'plain', 'none']
        - Add verbose
    - Deprecate controlling verbose using logging level

    - Tracker will not save config in checkpoint anymore.
        - Users can still save their config using custom tracker with custom saving hook


### 0.3.8

    - Add DummyTracker

### 0.3.7

    - seperate register_writer from register_scalar
        - before, we use `writer = tracker.register_scalar(writer, ...)`
        - now, we use
            1. `writer = tracker.register_writer(writer)`
            1. `tracker.register_scalar(...)  `
        - tracker.add_scalar can be use without writer registered for earlystopping
        - add shorthand api to create summary writer
            - writer = tracker.create_summary_writer()

### 0.3.6
    - The behavior introduced in 0.3.5 that earlystopping handler will add best metrics as hparams to summary is removed.
        - Instead, use tracker.get_best_scalars() to get the best metrics and add to summary writer explicitly manually.


### 0.3.5

    - Now the earlystopping handler will add best metrics as hparams to summary
    - Now the message printed to terminal can be turned off
        - E.g.
            1. turn off scalar
                - nf.set_stream_logging_level(nf.LoggingLevel.ON_SCALAR_ADD + 1)
            2. turn off all message
                - nf.set_stream_logging_level(nf.LoggingLevel.NO_STREAM_LOGGING)


### 0.3.4

    - Change the way to initialize tracker
        - Before v0.3.4, tracker won't be fully initialized, and will wait until one of the following function begin called.
            - load_checkpoint  
            - load_best
            - load_latest
            - range
        - Now the tracker is always fully initalized, use the argument `from_checkpoint` to decide how to load the checkpoint
    - Deprecate BaseConfig
    - Seperate checkpoint from BaseTracker
    - Seperate summary_writer from BaseTracker


### 0.3.3

    - Add track_config.save_end

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