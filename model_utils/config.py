from ..config import BaseConfig, UNIMPLEMENTED

class ModelUtilsConfig(BaseConfig):

    # device
    device = UNIMPLEMENTED

    # IMAGE_SHAPE = (224, 224)

    learning_rate = UNIMPLEMENTED


    # num of epochs per checkpoints
    # e.g. 1 stand for save model every epoch
    #      0 for not save until finish
    epochs_per_checkpoint: int = UNIMPLEMENTED

    # dir for saving checkpoints and log files
    log_dir: str = UNIMPLEMENTED

    early_stopping: bool = UNIMPLEMENTED

    # only matter when EARLY_STOPPING is set to True
    early_stopping_threshold: int = UNIMPLEMENTED

    num_class: int = UNIMPLEMENTED
