from pathlib import Path
from typing import Optional, Sequence

import hydra
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig, ValueNode
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import transforms as transform_lib
from torchvision.datasets import MNIST


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            normalize: bool,
            split: DictConfig,
            num_workers: DictConfig,
            batch_size: DictConfig,
            cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg

        self.data_dir = data_dir
        self.normalize = normalize
        self.split = split
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

    def prepare_data(self) -> None:
        """Saves MNIST files to `data_dir`"""
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None) -> None:

        extra = dict(transform=self.default_transforms) if self.default_transforms else {}
        dataset = MNIST(self.data_dir, train=True, download=False, **extra)
        train_length = len(dataset)

        self.dataset_train, self.dataset_val = random_split(dataset,
                                                            [train_length - self.split.val_split, self.split.val_split])

    def train_dataloader(self):
        """MNIST train set removes a subset to use for validation"""
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size.train,
            shuffle=True,
            num_workers=self.num_workers.train,
            drop_last=True,
            pin_memory=True,
        )

        return loader

    def val_dataloader(self):
        """MNIST val set uses a subset of the training set for validation"""

        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size.val,
            shuffle=False,
            num_workers=self.num_workers.val,
            drop_last=True,
            pin_memory=True,
        )

        return loader

    def test_dataloader(self):
        """MNIST test set uses the test split"""
        extra = dict(transform=self.default_transforms) if self.default_transforms else {}
        dataset = MNIST(self.data_dir, train=False, download=False, **extra)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size.test,
            shuffle=False,
            num_workers=self.num_workers.test,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    @property
    def default_transforms(self):

        if self.normalize:
            mnist_transforms = transform_lib.Compose([
                transform_lib.ToTensor(), transform_lib.Normalize(mean=(0.5,), std=(0.5,))
            ])
        else:
            mnist_transforms = transform_lib.ToTensor()

        return mnist_transforms
