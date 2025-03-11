import random

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.datasets import CIFAR100
from vision.data.cifar100_dm import CIFAR100DataModule
from vision.util import data_structs as ds


class RandomLabel_100_C100_DataModule(CIFAR100DataModule):
    random_label_percent: float = 1
    # If I remove split from init call:
    #   I will have to make the Subclasses of the Cifar10BaseMergingModule
    #   override wherever the splitting takes place.
    #   Because this is where the KFold and Disjoint DataModule differ!

    def train_dataloader(self, split: int, transform: ds.Augmentation, **kwargs) -> DataLoader:
        """Get a train dataloader"""
        dataset = CIFAR100(
            root=self.dataset_path,
            train=True,
            download=False,
            transform=self.get_transforms(transform),
        )

        targets = dataset.targets

        # Depending on the random ratio,
        do_random_labels = [random.random() < self.random_label_percent for _ in range(len(targets))]
        new_lbls_pairs = []
        for lbl, do_ in zip(targets, do_random_labels):
            if do_:
                new_lbls_pairs.append(random.randint(0, 99))
            else:
                new_lbls_pairs.append(lbl)

        dataset.targets = new_lbls_pairs

        train_ids, _ = self.get_train_val_split(split, len(dataset))
        dataset = Subset(dataset, train_ids)
        # INFO: Currently does not differentiate into different folds, as the
        #   Dataset comes with a deliberate validation set.
        return DataLoader(dataset=dataset, **kwargs)


class RandomLabel_75_C100_DataModule(RandomLabel_100_C100_DataModule):
    random_label_percent: float = 0.75


class RandomLabel_50_C100_DataModule(RandomLabel_100_C100_DataModule):
    random_label_percent: float = 0.5


class RandomLabel_25_C100_DataModule(RandomLabel_100_C100_DataModule):
    random_label_percent: float = 0.25
