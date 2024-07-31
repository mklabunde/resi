import os
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms as trans
from vision.data.base_datamodule import BaseDataModule
from vision.data.imagenet_ds import ImageNetDataset
from vision.randaugment.randaugment import ImageNetPolicy
from vision.util import data_structs as ds


# from ke.randaugment.randaugment import ImageNetPolicy

# from RandAugment import RandAugment


class ImagenetDataModule(BaseDataModule):
    datamodule_id = ds.Dataset.IMAGENET
    n_classes = 1000

    # If I remove split from init call:
    #   I will have to make the Subclasses of the Cifar10BaseMergingModule override
    #   wherever the splitting takes place.
    #   Because this is where the KFold and Disjoint DataModule differ!

    def __init__(self, advanced_da: bool):
        """ """
        super().__init__()
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.image_size = (160, 160)
        if advanced_da:
            self.train_trans = trans.Compose(
                [
                    trans.RandomResizedCrop(list(self.image_size)),
                    trans.RandomHorizontalFlip(),
                    ImageNetPolicy(),
                    trans.ToTensor(),
                    trans.Normalize(self.mean, self.std),
                ]
            )
        else:
            self.train_trans = trans.Compose(
                [
                    trans.RandomResizedCrop(list(self.image_size)),
                    trans.RandomHorizontalFlip(),
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
            dataset_path_p = Path(os.environ["RAW_DATA"]) / "Imagenet"
        elif "data" in os.environ:
            dataset_path_p = Path(os.environ["data"]) / "Imagenet"
        else:
            raise KeyError(
                "Couldn't find environ variable 'RAW_DATA' or 'data'." "Therefore unable to find CIFAR10 dataset"
            )

        assert dataset_path_p.exists(), (
            f"The given path to the Dataset {dataset_path_p} does not"
            f" contain the Data. To download go to "
            f"https://www.kaggle.com/c/imagenet-object-localization-challenge"
        )
        dataset_path = str(dataset_path_p)
        self.dataset_path = dataset_path

    def train_dataloader(self, split: int, transform: ds.Augmentation, **kwargs) -> DataLoader:
        """Get a train dataloader"""
        dataset = ImageNetDataset(
            root=self.dataset_path,
            split="train",
            transform=self.get_transforms(transform),
        )
        # INFO: Currently does not differentiate into different folds, as the
        #   Dataset comes with a deliberate validation set.
        return DataLoader(dataset=dataset, **kwargs)

    def val_dataloader(self, split: int, transform: ds.Augmentation, **kwargs) -> DataLoader:
        """Get a validation dataloader"""
        dataset = ImageNetDataset(
            root=self.dataset_path,
            split="val",
            transform=self.get_transforms(transform),
        )
        return DataLoader(dataset=dataset, **kwargs)

    def test_dataloader(self, transform: ds.Augmentation = ds.Augmentation.VAL, **kwargs) -> DataLoader:
        dataset = ImageNetDataset(
            root=self.dataset_path,
            split="test",
            transform=self.get_transforms(transform),
        )
        return DataLoader(dataset=dataset, **kwargs)
