import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from typing import Any

class VIODataModule(LightningDataModule):
    """
    This is only for testing if we can do forward pass correctly with our new models.
    """
    def __init__(self,
                 batch_size=4,
                 num_workers=0,
                 pin_memory=False,
                 train_loader=None,
                 val_loader=None,
                 test_loader=None):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train= train_loader 
        self.data_val = val_loader
        self.data_test = test_loader

        self.batch_size_per_device = batch_size

    def prepare_data(self):
        # download
        pass

    def setup(self, stage):
        # assign train/val datasets for use in dataloaders

        # assign test dataset for use in dataloaders
        pass

    
    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

if __name__ == "__main__":
    _ = DummyVIODataModule()
