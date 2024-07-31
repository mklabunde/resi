import os
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms as trans
from vision.data.base_datamodule import BaseDataModule
from vision.data.medmnist_ds import DermaMNIST
from vision.randaugment.randaugment import CIFAR10Policy
from vision.randaugment.randaugment import Cutout
from vision.util import data_structs as ds


# from ke.data import auto_augment

# from ke.randaugment.randaugment  import CIFAR10Policy

# from ke.data.auto_augment import CIFAR10Policy

# from ke.data import cutout_aug


class DermaMNISTDataModule(BaseDataModule):
    datamodule_id = ds.Dataset.DermaMNIST
    n_classes = 7

    # If I remove split from init call:
    #   I will have to make the Subclasses of the Cifar10BaseMergingModule
    #   override wherever the splitting takes place.
    #   Because this is where the KFold and Disjoint DataModule differ!

    def __init__(self, advanced_da: bool):
        """ """
        super().__init__()
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)
        self.image_size = (28, 28)
        if advanced_da:
            self.train_trans = trans.Compose(
                [
                    trans.RandomCrop(self.image_size, padding=4, fill=(128, 128, 128)),
                    trans.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    Cutout(size=16),
                    trans.ToTensor(),
                    trans.Normalize(self.mean, self.std),
                ]
            )
        else:
            self.train_trans = trans.Compose(
                [
                    trans.RandomCrop(self.image_size, padding=4, fill=(128, 128, 128)),
                    trans.RandomHorizontalFlip(),
                    Cutout(size=16),
                    trans.ToTensor(),
                    trans.Normalize(self.mean, self.std),
                ]
            )
        self.val_trans = trans.Compose(
            [
                trans.ToTensor(),
                trans.Normalize(self.mean, self.std),
            ]
        )

        if "RAW_DATA" in os.environ:
            dataset_path_p = Path(os.environ["RAW_DATA"]) / "medmnist"
        elif "data" in os.environ:
            dataset_path_p = Path(os.environ["data"]) / "medmnist"
        else:
            raise KeyError(
                "Couldn't find environ variable 'RAW_DATA' or 'data'." "Therefore unable to find CIFAR10 dataset"
            )

        assert dataset_path_p.exists(), f"CIFAR10 dataset not found at {dataset_path_p}"

        dataset_path: str = str(dataset_path_p)
        self.dataset_path = dataset_path

    def train_dataloader(
        self,
        split: int,
        transform: ds.Augmentation,
        **kwargs,
    ) -> DataLoader:
        """Get a train dataloader"""
        if split != 0:
            raise NotImplementedError("MedMNIST has fixed Train Val and Test splits!")
        dataset = DermaMNIST(
            split="train",
            transform=self.get_transforms(transform),
            target_transform=None,
            download=False,
            as_rgb=True,
            root=self.dataset_path,
        )
        return DataLoader(dataset=dataset, **kwargs)

    def val_dataloader(
        self,
        split: int,
        transform: ds.Augmentation,
        **kwargs,
    ) -> DataLoader:
        """Get a validation dataloader"""
        if split != 0:
            raise NotImplementedError("MedMNIST has fixed Train Val and Test splits!")
        dataset = DermaMNIST(
            split="val",
            transform=self.get_transforms(transform),
            target_transform=None,
            download=False,
            as_rgb=True,
            root=self.dataset_path,
        )
        return DataLoader(dataset=dataset, **kwargs)

    def test_dataloader(self, transform: ds.Augmentation = ds.Augmentation.VAL, **kwargs) -> DataLoader:
        dataset = DermaMNIST(
            split="test",
            transform=self.get_transforms(transform),
            target_transform=None,
            download=False,
            as_rgb=True,
            root=self.dataset_path,
        )
        return DataLoader(dataset=dataset, **kwargs)
