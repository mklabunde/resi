import os
from pathlib import Path

import torch
from loguru import logger
from repsim.benchmark.paths import VISION_DATA_PATH
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms as trans
from vision.data.base_datamodule import BaseDataModule
from vision.data.imagenet100_dm import Imagenet100DataModule
from vision.data.shortcuts.cifar10_color_dataset import ColorDotCIFAR10
from vision.data.shortcuts.imagenet100_color_ds import ColorDotImageNet100Dataset
from vision.randaugment.randaugment import CIFAR10Policy
from vision.randaugment.randaugment import Cutout
from vision.util import data_structs as ds


# from ke.data import auto_augment

# from ke.randaugment.randaugment  import CIFAR10Policy

# from ke.data.auto_augment import CIFAR10Policy

# from ke.data import cutout_aug


class ColorDot_100_IN100Datamodule(Imagenet100DataModule):
    dot_correlation = 100
    dot_diameter = 5

    # If I remove split from init call:
    #   I will have to make the Subclasses of the Cifar10BaseMergingModule
    #   override wherever the splitting takes place.
    #   Because this is where the KFold and Disjoint DataModule differ!

    def train_dataloader(
        self,
        split: int,
        transform: ds.Augmentation,
        **kwargs,
    ) -> DataLoader:
        """Get a train dataloader"""
        dataset = ColorDotImageNet100Dataset(
            root=self.dataset_path,
            split="train",
            kfold_split=0,
            transform=self.get_transforms(transform),
            dot_correlation=self.dot_correlation,
            dot_diameter=self.dot_diameter,
        )
        train_ids, _ = self.get_train_val_split(split, len(dataset))
        dataset = Subset(dataset, train_ids)
        logger.info(f"Length of train dataset: {len(dataset)}")
        return DataLoader(dataset=dataset, **kwargs)

    def val_dataloader(
        self,
        split: int,
        transform: ds.Augmentation,
        **kwargs,
    ) -> DataLoader:
        """Get a validation dataloader"""
        dataset = ColorDotImageNet100Dataset(
            root=self.dataset_path,
            split="val",
            kfold_split=0,
            transform=self.get_transforms(transform),
            dot_correlation=self.dot_correlation,
            dot_diameter=self.dot_diameter,
        )
        _, val_ids = self.get_train_val_split(split, len(dataset))
        dataset = Subset(dataset, val_ids)
        logger.info(f"Length of val dataset: {len(dataset)}")
        return DataLoader(dataset=dataset, **kwargs)

    def test_dataloader(self, transform: ds.Augmentation = ds.Augmentation.VAL, **kwargs) -> DataLoader:
        dataset = ColorDotImageNet100Dataset(
            root=self.dataset_path,
            split="test",
            kfold_split=0,
            transform=self.get_transforms(transform),
            dot_correlation=self.dot_correlation,
            dot_diameter=self.dot_diameter,
        )
        logger.info(f"Length of test dataset: {len(dataset)}")
        return DataLoader(dataset=dataset, **kwargs)


class ColorDot_75_IN100Datamodule(ColorDot_100_IN100Datamodule):
    dot_correlation = 75


class ColorDot_50_IN100Datamodule(ColorDot_100_IN100Datamodule):
    dot_correlation = 50


class ColorDot_25_IN100Datamodule(ColorDot_100_IN100Datamodule):
    dot_correlation = 25


class ColorDot_0_IN100Datamodule(ColorDot_100_IN100Datamodule):
    dot_correlation = 0


if __name__ == "__main__":

    from matplotlib import pyplot as plt
    import numpy as np

    dataset_path = VISION_DATA_PATH
    save_dir = Path(__file__).parent / "example_imgs_IN100"
    save_dir.mkdir(exist_ok=True)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    val_trans = trans.Compose([trans.Resize([224, 224]), trans.ToTensor(), trans.Normalize(mean, std)])

    for pct in [0, 25, 50, 75, 100]:
        dataset_in100_1 = ColorDotImageNet100Dataset(
            root=dataset_path,
            split="train",
            kfold_split=0,
            transform=val_trans,
            dot_correlation=pct,
            dot_diameter=5,
        )

        dataset_in100_2 = ColorDotImageNet100Dataset(
            root=dataset_path,
            split="train",
            kfold_split=0,
            transform=val_trans,
            dot_correlation=pct,
            dot_diameter=5,
        )

        sc_dm = ColorDot_100_IN100Datamodule(True)
        val_dataloader = sc_dm.val_dataloader(0, ds.Augmentation.VAL)

        print(f"Testing Color_Dot_C10 {pct}...")
        for i in range(25):
            dl1 = dataset_in100_1[i]
            dl2 = dataset_in100_2[i]

            image_1 = dl1[0]
            image_2 = dl2[0]

            class_label = dl1[1]
            color_label = dl1[2]

            batch_1_out = image_1 * np.array(std)[:, None, None] + np.array(mean)[:, None, None]

            plt.imshow(batch_1_out.permute(1, 2, 0))
            plt.axis("off")
            plt.title(f"Dot_corr_{pct}_IN100_image_{i}; class: {class_label}; color: {color_label}")
            plt.savefig(os.path.join(save_dir, f"Dot_corr_{pct}_IN100_image_{i}.png"))
            plt.close()

            assert torch.all(torch.isclose(image_1, image_2)), "Batches should be the same!"
        print("...passed!")
