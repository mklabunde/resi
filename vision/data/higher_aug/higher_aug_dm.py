import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations import Compose
from albumentations import GaussNoise
from albumentations import HorizontalFlip
from albumentations import Normalize
from albumentations import PadIfNeeded
from albumentations import RandomCrop
from albumentations import Resize
from albumentations.pytorch import ToTensorV2
from PIL import Image
from repsim.benchmark.paths import VISION_DATA_PATH
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms as trans
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from vision.data.base_datamodule import BaseDataModule
from vision.data.cifar100_dm import CIFAR100DataModule
from vision.data.cifar10_dm import CIFAR10DataModule
from vision.randaugment.randaugment import CIFAR10Policy
from vision.util import data_structs as ds


# from ke.data import auto_augment

# from ke.randaugment.randaugment  import CIFAR10Policy

# from ke.data.auto_augment import CIFAR10Policy

# from ke.data import cutout_aug


class C100_AlbuDataset(CIFAR100):
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.transform is not None:
            img = self.transform(image=np.array(img))

        if self.target_transform is not None:
            target = self.target_transform(target=target)

        return img["image"], target


class C10_AlbuDataset(CIFAR10):
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.transform is not None:
            img = self.transform(image=np.array(img))

        if self.target_transform is not None:
            target = self.target_transform(target=target)

        return img["image"], target


class Gauss_Max_CIFAR10DataModule(CIFAR10DataModule):
    datamodule_id = ds.Dataset.CIFAR10
    n_classes = 10
    var_limit = (0, 4000)  # (5, 20), (10, 50), (25, 75), (40, 100)

    # If I remove split from init call:
    #   I will have to make the Subclasses of the Cifar10BaseMergingModule
    #   override wherever the splitting takes place.
    #   Because this is where the KFold and Disjoint DataModule differ!

    def __init__(
        self,
        advanced_da: bool,
    ):
        """ """
        super().__init__(advanced_da)

        self.train_trans = Compose(
            [
                RandomCrop(height=self.image_size[0], width=self.image_size[1]),
                HorizontalFlip(),
                GaussNoise(
                    var_limit=0 if self.var_limit is None else self.var_limit,
                    p=0 if self.var_limit is None else 1,
                    always_apply=True,
                ),
                # trans.GaussianBlur(5, sigma=(0.1, 2.0)),
                Normalize(self.mean, self.std),
                ToTensorV2(),
            ]
        )

        self.val_trans = Compose([Normalize(self.mean, self.std), ToTensorV2()])
        self.dataset_path = self.prepare_data()

    def prepare_data(self, **kwargs) -> None:
        if "CIFAR10" in os.environ:
            # Setting the path for this can also be made optional (It's 170 mb afterall)
            dataset_path = os.environ["CIFAR10"]
        else:
            # Test that it is as expected
            dataset_path = os.path.join(VISION_DATA_PATH, "CIFAR10")
            _ = CIFAR10(root=dataset_path, download=True)
        return dataset_path

    def train_dataloader(
        self,
        split: int,
        transform: ds.Augmentation,
        **kwargs,
    ) -> DataLoader:
        """Get a train dataloader"""
        dataset = C10_AlbuDataset(
            root=self.dataset_path,
            train=True,
            download=False,
            transform=self.get_transforms(transform),
        )
        train_ids, _ = self.get_train_val_split(split, len(dataset))
        dataset = Subset(dataset, train_ids)
        return DataLoader(dataset=dataset, **kwargs)

    def val_dataloader(
        self,
        split: int,
        transform: ds.Augmentation,
        **kwargs,
    ) -> DataLoader:
        """Get a validation dataloader"""
        dataset = C10_AlbuDataset(
            root=self.dataset_path,
            train=True,
            download=False,
            transform=self.get_transforms(transform),
        )
        _, val_ids = self.get_train_val_split(split, len(dataset))
        dataset = Subset(dataset, val_ids)
        return DataLoader(dataset=dataset, **kwargs)

    def test_dataloader(self, transform: ds.Augmentation = ds.Augmentation.VAL, **kwargs) -> DataLoader:
        dataset = C10_AlbuDataset(
            root=self.dataset_path,
            train=False,
            download=False,
            transform=self.get_transforms(transform),
        )
        return DataLoader(dataset=dataset, **kwargs)

    def anchor_dataloader(self, **kwargs) -> DataLoader:
        return NotImplementedError()


class Gauss_L_CIFAR10DataModule(Gauss_Max_CIFAR10DataModule):
    var_limit = (0, 3000)


class Gauss_M_CIFAR10DataModule(Gauss_Max_CIFAR10DataModule):
    var_limit = (0, 2000)


class Gauss_S_CIFAR10DataModule(Gauss_Max_CIFAR10DataModule):
    var_limit = (0, 1000)


class Gauss_Off_CIFAR10DataModule(Gauss_Max_CIFAR10DataModule):
    var_limit = None


class Gauss_Max_CIFAR100DataModule(CIFAR100DataModule):
    datamodule_id = ds.Dataset.CIFAR100
    n_classes = 100
    var_limit = (0, 4000)  # (5, 20), (10, 50), (25, 75), (40, 100)

    # If I remove split from init call:
    #   I will have to make the Subclasses of the Cifar10BaseMergingModule
    #   override wherever the splitting takes place.
    #   Because this is where the KFold and Disjoint DataModule differ!

    def __init__(self, advanced_da: bool, is_vit: bool = False):
        """ """
        super().__init__(advanced_da, is_vit)
        if is_vit:
            self.image_size = (224, 224)
            train_trans = [Resize(height=224, width=224)]
            val_trans = [Resize(height=224, width=224)]
        else:
            train_trans = []
            val_trans = []

        self.train_trans = Compose(
            train_trans
            + [
                RandomCrop(height=self.image_size[0], width=self.image_size[1]),
                HorizontalFlip(),
                GaussNoise(
                    var_limit=0 if self.var_limit is None else self.var_limit,
                    p=0 if self.var_limit is None else 1,
                    always_apply=True,
                ),
                # trans.GaussianBlur(5, sigma=(0.1, 2.0)),
                Normalize(self.mean, self.std),
                ToTensorV2(),
            ]
        )

        self.val_trans = Compose(val_trans + [Normalize(self.mean, self.std), ToTensorV2()])
        self.dataset_path = self.prepare_data()

    def prepare_data(self, **kwargs) -> None:
        if "CIFAR100" in os.environ:
            # Setting the path for this can also be made optional (It's 170 mb afterall)
            dataset_path = os.environ["CIFAR100"]
        else:
            # Test that it is as expected
            dataset_path = os.path.join(VISION_DATA_PATH, "CIFAR100")
            _ = CIFAR100(root=dataset_path, download=True)
        return dataset_path

    def train_dataloader(
        self,
        split: int,
        transform: ds.Augmentation,
        **kwargs,
    ) -> DataLoader:
        """Get a train dataloader"""
        dataset = C100_AlbuDataset(
            root=self.dataset_path,
            train=True,
            download=False,
            transform=self.get_transforms(transform),
        )
        train_ids, _ = self.get_train_val_split(split, len(dataset))
        dataset = Subset(dataset, train_ids)
        return DataLoader(dataset=dataset, **kwargs)

    def val_dataloader(
        self,
        split: int,
        transform: ds.Augmentation,
        **kwargs,
    ) -> DataLoader:
        """Get a validation dataloader"""
        dataset = C100_AlbuDataset(
            root=self.dataset_path,
            train=True,
            download=False,
            transform=self.get_transforms(transform),
        )
        _, val_ids = self.get_train_val_split(split, len(dataset))
        dataset = Subset(dataset, val_ids)
        return DataLoader(dataset=dataset, **kwargs)

    def test_dataloader(self, transform: ds.Augmentation = ds.Augmentation.VAL, **kwargs) -> DataLoader:
        dataset = C100_AlbuDataset(
            root=self.dataset_path,
            train=False,
            download=False,
            transform=self.get_transforms(transform),
        )
        return DataLoader(dataset=dataset, **kwargs)

    def anchor_dataloader(self, **kwargs) -> DataLoader:
        return NotImplementedError()


class Gauss_L_CIFAR100DataModule(Gauss_Max_CIFAR100DataModule):
    var_limit = (0, 3000)


class Gauss_M_CIFAR100DataModule(Gauss_Max_CIFAR100DataModule):
    var_limit = (0, 2000)


class Gauss_S_CIFAR100DataModule(Gauss_Max_CIFAR100DataModule):
    var_limit = (0, 1000)


class Gauss_Off_CIFAR100DataModule(Gauss_Max_CIFAR100DataModule):
    var_limit = None


if __name__ == "__main__":
    # Plot the Gauss Max Datamodule and produce some examplary images.
    # Load 20 images for each datamodule and save to disk
    dmoff = Gauss_Off_CIFAR100DataModule(advanced_da=True)
    dms = Gauss_S_CIFAR100DataModule(advanced_da=True)
    dmm = Gauss_M_CIFAR100DataModule(advanced_da=True)
    dml = Gauss_L_CIFAR100DataModule(advanced_da=True)
    dmmax = Gauss_Max_CIFAR100DataModule(advanced_da=True)

    # Load train dataloader for each datamodule
    train_max = dmmax.train_dataloader(split=0, transform=ds.Augmentation.TRAIN, batch_size=1)
    train_l = dml.train_dataloader(split=0, transform=ds.Augmentation.TRAIN, batch_size=1)
    train_m = dmm.train_dataloader(split=0, transform=ds.Augmentation.TRAIN, batch_size=1)
    train_s = dms.train_dataloader(split=0, transform=ds.Augmentation.TRAIN, batch_size=1)
    train_off = dmoff.train_dataloader(split=0, transform=ds.Augmentation.TRAIN, batch_size=1)

    # Save 20 images from each dataloader to disk
    save_dir = os.path.join(os.path.dirname(__file__), "GaussExamplesC100")
    os.makedirs(save_dir, exist_ok=True)

    # Save 20 images from each dataloader to disk
    save_dir = os.path.join(os.path.dirname(__file__), "GaussExamplesC100")
    os.makedirs(save_dir, exist_ok=True)

    for cnt, (name, dl) in enumerate(
        [("off", train_off), ("s", train_s), ("m", train_m), ("l", train_l), ("max", train_max)]
    ):
        for i, batch in enumerate(dl):
            if i >= 20:
                break
            image = torch.squeeze(batch[0])
            save_path = os.path.join(save_dir, f"{cnt}_{name}_image_{i}.png")
            image = torch.squeeze(batch[0]).permute(1, 2, 0).numpy()
            image = (image - image.min()) / (image.max() - image.min())  # Just rescale.
            plt.imsave(save_path, image)
