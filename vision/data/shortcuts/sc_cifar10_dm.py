import os
from pathlib import Path

import torch
from repsim.benchmark.paths import VISION_DATA_PATH
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms as trans
from vision.data.base_datamodule import BaseDataModule
from vision.data.cifar10_dm import CIFAR10DataModule
from vision.data.shortcuts.cifar10_color_dataset import ColorDotCIFAR10
from vision.randaugment.randaugment import CIFAR10Policy
from vision.randaugment.randaugment import Cutout
from vision.util import data_structs as ds


# from ke.data import auto_augment

# from ke.randaugment.randaugment  import CIFAR10Policy

# from ke.data.auto_augment import CIFAR10Policy

# from ke.data import cutout_aug


class ColorDot_100_CIFAR10DataModule(CIFAR10DataModule):
    datamodule_id = ds.Dataset.CIFAR10
    n_classes = 10
    dot_correlation = 100
    dot_diameter = 5

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

        if advanced_da:
            self.train_trans = trans.Compose(
                [
                    trans.RandomCrop(self.image_size, padding=4, fill=(128, 128, 128)),
                    trans.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    trans.ToTensor(),
                    trans.Normalize(self.mean, self.std),
                ]
            )
        else:
            self.train_trans = trans.Compose(
                [
                    trans.RandomCrop(self.image_size, padding=4, fill=(128, 128, 128)),
                    trans.RandomHorizontalFlip(),
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
        self.dataset_path = self.prepare_data()

    def train_dataloader(
        self,
        split: int,
        transform: ds.Augmentation,
        **kwargs,
    ) -> DataLoader:
        """Get a train dataloader"""
        dataset = ColorDotCIFAR10(
            root=self.dataset_path,
            train=True,
            download=False,
            transform=self.get_transforms(transform),
            dot_correlation=self.dot_correlation,
            dot_diameter=self.dot_diameter,
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
        dataset = ColorDotCIFAR10(
            root=self.dataset_path,
            train=True,
            download=False,
            transform=self.get_transforms(transform),
            dot_correlation=self.dot_correlation,
            dot_diameter=self.dot_diameter,
        )
        _, val_ids = self.get_train_val_split(split, len(dataset))
        dataset = Subset(dataset, val_ids)
        return DataLoader(dataset=dataset, **kwargs)

    def test_dataloader(self, transform: ds.Augmentation = ds.Augmentation.VAL, **kwargs) -> DataLoader:
        dataset = ColorDotCIFAR10(
            root=self.dataset_path,
            train=False,
            download=False,
            transform=self.get_transforms(transform),
            dot_correlation=self.dot_correlation,
            dot_diameter=self.dot_diameter,
        )
        return DataLoader(dataset=dataset, **kwargs)

    def anchor_dataloader(self, **kwargs) -> DataLoader:
        return NotImplementedError()


class ColorDot_75_CIFAR10DataModule(ColorDot_100_CIFAR10DataModule):
    dot_correlation = 75


class ColorDot_50_CIFAR10DataModule(ColorDot_100_CIFAR10DataModule):
    dot_correlation = 50


class ColorDot_25_CIFAR10DataModule(ColorDot_100_CIFAR10DataModule):
    dot_correlation = 25


class ColorDot_0_CIFAR10DataModule(ColorDot_100_CIFAR10DataModule):
    dot_correlation = 0


if __name__ == "__main__":

    if "CIFAR10" in os.environ:
        # Setting the path for this can also be made optional (It's 170 mb afterall)
        dataset_path = os.environ["CIFAR10"]
    else:
        # Test that it is as expected
        dataset_path = os.path.join(VISION_DATA_PATH, "CIFAR10")
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    val_trans = trans.Compose([trans.ToTensor(), trans.Normalize(mean, std)])

    for pct in [
        100,
        75,
        50,
        25,
        0,
    ]:
        dataset_c10_1 = ColorDotCIFAR10(
            root=dataset_path,
            train=False,
            download=False,
            transform=val_trans,
            dot_correlation=pct,
            dot_diameter=5,
        )

        dataset_c10_2 = ColorDotCIFAR10(
            root=dataset_path,
            train=False,
            download=False,
            transform=val_trans,
            dot_correlation=pct,
            dot_diameter=5,
        )

        print(f"Testing Color_Dot_C10 {pct}...")
        for i in range(25):
            dl1 = dataset_c10_1[i]
            dl2 = dataset_c10_2[i]

            batch_1 = dl1[0]
            batch_2 = dl2[0]

            assert torch.all(torch.isclose(batch_1, batch_2)), "Batches should be the same!"
        print("...passed!")

    # ColorDot_100_CIFAR10DataModule(advanced_da=True).prepare_data()
