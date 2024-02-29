# Fardin Rastakhiz @ 2023

from abc import abstractmethod

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from lightning.pytorch import LightningDataModule
from scripts.configs.ConfigClass import Config



class GraphDataModule(LightningDataModule):

    def __init__(self, config: Config, device, test_size=0.2, val_size=0.15):
        super(GraphDataModule, self).__init__()
        self.test_size = test_size
        self.val_size = val_size
        self.device = device
        self.config = config

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def setup(self, stage: str):
        pass

    @abstractmethod
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    @abstractmethod
    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    @abstractmethod
    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    @abstractmethod
    def teardown(self, stage: str) -> None:
        pass
    
    @abstractmethod
    def zero_rule_baseline():
        pass
