# pylint: disable=no-self-use
from torch.optim import Adam

from model_utils import BaseModelUtils, Criteria

from .model import FooModel

class TestModelUtils(BaseModelUtils):

    @staticmethod
    def _get_optimizer(model: FooModel, config):
        return Adam(model.parameters())
    
    def _train_epoch(self, train_dataset) -> Criteria:
        return Criteria(0.0)
    
    def _eval_epoch(self, eval_dataset) -> Criteria:
        return Criteria(0.0)
    