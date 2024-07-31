import os
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms as trans
from vision.data.base_datamodule import BaseDataModule
from vision.data.cutout_aug import Cutout
from vision.data.tiny_imagenet_ds import TinyImageNetDataset
from vision.randaugment.randaugment import CIFAR10Policy
from vision.util import data_structs as ds


# from ke.randaugment.randaugment import ImageNetPolicy

# from RandAugment import RandAugment


class TinyImagenetDataModule(BaseDataModule):
    datamodule_id = ds.Dataset.IMAGENET
    n_classes = 200

    # If I remove split from init call:
    #   I will have to make the Subclasses of the Cifar10BaseMergingModule override
    #   wherever the splitting takes place.
    #   Because this is where the KFold and Disjoint DataModule differ!

    def __init__(self, advanced_da: bool):
        """ """
        super().__init__()
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.image_size = (64, 64)

        if advanced_da:
            self.train_trans = trans.Compose(
                [
                    trans.RandomResizedCrop(list(self.image_size)),
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
                    trans.RandomResizedCrop(list(self.image_size)),
                    trans.RandomHorizontalFlip(),
                    Cutout(size=16),
                    trans.ToTensor(),
                    trans.Normalize(self.mean, self.std),
                ]
            )
        self.val_trans = trans.Compose(
            [
                trans.Resize(size=list(self.image_size)),
                trans.ToTensor(),
                trans.Normalize(self.mean, self.std),
            ]
        )
        if "RAW_DATA" in os.environ:
            dataset_path_p = Path(os.environ["RAW_DATA"]) / "tiny-imagenet-200"
        elif "data" in os.environ:
            dataset_path_p = Path(os.environ["data"]) / "tiny-imagenet-200"
        else:
            raise KeyError(
                "Couldn't find environ variable 'RAW_DATA' or 'data'." "Therefore unable to find CIFAR10 dataset"
            )

        assert dataset_path_p.exists(), (
            f"The given path to the Dataset {dataset_path_p} does not"
            f" contain the Data. To download go to "
            f"https://image-net.org/download-images.php"
        )
        dataset_path = str(dataset_path_p)
        self.dataset_path = dataset_path

    def train_dataloader(self, split: int, transform: ds.Augmentation, **kwargs) -> DataLoader:
        """Get a train dataloader"""
        dataset = TinyImageNetDataset(
            root=self.dataset_path,
            split="train",
            transform=self.get_transforms(transform),
        )
        # INFO: Currently does not differentiate into different folds, as the
        #   Dataset comes with a deliberate validation set.
        return DataLoader(dataset=dataset, **kwargs)

    def val_dataloader(self, split: int, transform: ds.Augmentation, **kwargs) -> DataLoader:
        """Get a validation dataloader"""
        dataset = TinyImageNetDataset(
            root=self.dataset_path,
            split="val",
            transform=self.get_transforms(transform),
        )
        return DataLoader(dataset=dataset, **kwargs)

    def test_dataloader(self, transform: ds.Augmentation = ds.Augmentation.VAL, **kwargs) -> DataLoader:
        dataset = TinyImageNetDataset(
            root=self.dataset_path,
            split="test",
            transform=self.get_transforms(transform),
        )
        return DataLoader(dataset=dataset, **kwargs)

    def anchor_dataloader(self, **kwargs) -> DataLoader:
        raise NotImplementedError("No anchor dataloader for TinyImageNet yet")
