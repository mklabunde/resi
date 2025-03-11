import os
from pathlib import Path

from repsim.benchmark.paths import VISION_DATA_PATH
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms as trans
from torchvision.datasets import CIFAR100
from vision.data.base_datamodule import BaseDataModule
from vision.randaugment.randaugment import CIFAR10Policy
from vision.randaugment.randaugment import Cutout
from vision.util import data_structs as ds


# from ke.data import auto_augment

# from ke.randaugment.randaugment  import CIFAR10Policy

# from ke.data.auto_augment import CIFAR10Policy

# from ke.data import cutout_aug


class CIFAR100DataModule(BaseDataModule):
    datamodule_id = ds.Dataset.CIFAR100
    n_classes = 100

    # If I remove split from init call:
    #   I will have to make the Subclasses of the Cifar10BaseMergingModule
    #   override wherever the splitting takes place.
    #   Because this is where the KFold and Disjoint DataModule differ!

    def __init__(self, advanced_da: bool, is_vit: bool = False):
        """ """
        super().__init__()
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)
        self.image_size = (32, 32)
        if is_vit:
            self.image_size = (224, 224)
            train_trans = [trans.Resize((224, 224))]
            val_trans = [trans.Resize((224, 224))]
        else:
            train_trans = []
            val_trans = []

        if advanced_da:
            self.train_trans = trans.Compose(
                (
                    train_trans  # Optionally add the resize
                    + [
                        trans.RandomCrop(self.image_size, padding=4, fill=(128, 128, 128)),
                        trans.RandomHorizontalFlip(),
                        CIFAR10Policy(),
                        Cutout(size=16),
                        trans.ToTensor(),
                        trans.Normalize(self.mean, self.std),
                    ]
                )
            )
        else:
            self.train_trans = trans.Compose(
                (
                    train_trans  # Optionally add the resize
                    + [
                        trans.RandomCrop(self.image_size, padding=4, fill=(128, 128, 128)),
                        trans.RandomHorizontalFlip(),
                        Cutout(size=16),
                        trans.ToTensor(),
                        trans.Normalize(self.mean, self.std),
                    ]
                )
            )
        self.val_trans = trans.Compose(
            val_trans
            + (
                [
                    trans.ToTensor(),
                    trans.Normalize(self.mean, self.std),
                ]
            )
        )
        self.dataset_path = self.prepare_data()

    def prepare_data(self, **kwargs) -> None:
        if "CIFAR100" in os.environ:
            # Setting the path for this can also be made optional (It's 170 mb afterall)
            dataset_path = os.environ["CIFAR100"]  # Optional pathing for the dataset
        else:
            # Test that it is as expected
            dataset_path = os.path.join(VISION_DATA_PATH, "CIFAR100")
            if "c100_downloaded" not in os.environ.keys():
                _ = CIFAR100(root=dataset_path, download=True)
                os.environ["c100_downloaded"] = "yay"
        return dataset_path

    def train_dataloader(
        self,
        split: int,
        transform: ds.Augmentation,
        **kwargs,
    ) -> DataLoader:
        """Get a train dataloader"""
        dataset = CIFAR100(
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
        dataset = CIFAR100(
            root=self.dataset_path,
            train=True,
            download=False,
            transform=self.get_transforms(transform),
        )
        _, val_ids = self.get_train_val_split(split, len(dataset))
        dataset = Subset(dataset, val_ids)
        return DataLoader(dataset=dataset, **kwargs)

    def test_dataloader(self, transform: ds.Augmentation = ds.Augmentation.VAL, **kwargs) -> DataLoader:
        dataset = CIFAR100(
            root=self.dataset_path,
            train=False,
            download=False,
            transform=self.get_transforms(transform),
        )
        return DataLoader(dataset=dataset, **kwargs)

    def anchor_dataloader(self):
        dataset = CIFAR100(
            root=self.dataset_path, train=True, download=False, transform=self.get_transforms(ds.Augmentation.VAL)
        )
        # Grab the first 1000 samples from the training set (for the anchors)
        anchors: dict[int, tuple] = {}
        samples_per_class = 1000 // self.n_classes
        cnt = 0
        while sum([len(v) for v in anchors.values()]) < 1000:
            im, lbl = dataset[cnt]
            if lbl not in anchors.keys():
                anchors[lbl] = [(im, lbl)]
            elif len(anchors[lbl]) < samples_per_class:
                anchors[lbl].append((im, lbl))
            cnt += 1
        # Flatten the list of lists
        # Anchors are
        preprocessed_anchors = [item for sublist in anchors.values() for item in sublist]
        return DataLoader(dataset=preprocessed_anchors, batch_size=100, shuffle=False)
