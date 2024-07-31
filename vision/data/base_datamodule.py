from abc import ABC
from abc import abstractmethod
from typing import Optional

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as trans
from vision.util import data_structs as ds


class BaseDataModule(ABC):
    datamodule_id: ds.Dataset
    n_classes: int

    def __init__(self):
        # The normal base data module structure is as follows:
        #   One collects all the possible Data one can train with into a huge set.
        #   This should be split into a set one can do whatever with (train set)
        #       and one fixed test set that is withheld and not used for anything
        #   Based off of this one can split the train set into an arbitrary collection
        #   of parts.
        #
        #   1. A Model that trains on all (labeled) images

        self.dataset_path: str = ""
        self.max_splits = 5

        # Original Raw Datasets & Splits

        self.prepared_already: bool = False
        self.mean: tuple = tuple()
        self.std: tuple = tuple()

        self.image_size: tuple[int, int] | None = None

        # Transformations for the training cases (aka preprocessing + augmentations)
        #   and trans for the validation cases (aka preprocessing)
        self.train_trans: Optional[list[trans.Compose]] = None
        self.val_trans: Optional[list[trans.Compose]] = None

    def get_transforms(self, aug: ds.Augmentation) -> Optional[list[trans.Compose]]:
        if aug == ds.Augmentation.TRAIN:
            return self.train_trans
        elif aug == ds.Augmentation.VAL:
            return self.val_trans
        else:
            return None

    def revert_base_transform(self, image_batch: np.ndarray):
        return (
            (image_batch * np.array(self.std)[None, :, None, None]) + np.array(self.mean)[None, :, None, None]
        ) * 255

    def get_train_val_split(self, split: int, n_samples: int) -> tuple[list[int], list[int]]:
        total_samples = n_samples
        # Inverse order to be backwards compatible
        split = self.max_splits - (split + 1)
        n_val = int(total_samples / self.max_splits)
        all_ids_set = set(range(total_samples))
        val_ids_set = set(range(split * n_val, (split + 1) * n_val))
        train_ids = list(all_ids_set - val_ids_set)
        val_ids = list(val_ids_set)
        return train_ids, val_ids

    @abstractmethod
    def train_dataloader(
        self,
        split: int,
        transform: ds.Augmentation,
        **kwargs,
    ) -> DataLoader:
        """
        Returns the train dataloader with the respective Augmentations.
        Kwargs are: [batch_size, shuffle, pin_memory, num_workers, drop_last]
        """
        pass

    @abstractmethod
    def val_dataloader(
        self,
        split: int,
        transform: ds.Augmentation,
        **kwargs,
    ) -> DataLoader:
        """Returns the val dataloader with the respective Augmentations"""
        pass

    @abstractmethod
    def test_dataloader(
        self,
        transform: ds.Augmentation = ds.Augmentation.VAL,
        **kwargs,
    ) -> DataLoader:
        """Returns the test datalodaer with the respective Augmentations"""
        pass
