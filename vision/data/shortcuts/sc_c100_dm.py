from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms as trans
from vision.data.cifar100_dm import CIFAR100DataModule
from vision.data.shortcuts.cifar100_color_dataset import ColorDotCIFAR100
from vision.randaugment.randaugment import CIFAR10Policy
from vision.util import data_structs as ds


class ColorDot_100_C100Datamodule(CIFAR100DataModule):
    datamodule_id = ds.Dataset.CIFAR100
    dot_correlation = 100
    n_classes = 100
    dot_diameter = 3

    # If I remove split from init call:
    #   I will have to make the Subclasses of the Cifar10BaseMergingModule
    #   override wherever the splitting takes place.
    #   Because this is where the KFold and Disjoint DataModule differ!

    def __init__(
        self,
        advanced_da: bool,
        is_vit: bool = False,
    ):
        """ """
        super().__init__(advanced_da, is_vit)
        if is_vit:
            self.image_size = (224, 224)
            train_trans = [trans.Resize((224, 224))]
            val_trans = [trans.Resize((224, 224))]
        else:
            train_trans = []
            val_trans = []

        if advanced_da:
            self.train_trans = trans.Compose(
                train_trans
                + [
                    trans.RandomCrop(self.image_size, padding=4, fill=(128, 128, 128)),
                    trans.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    trans.ToTensor(),
                    trans.Normalize(self.mean, self.std),
                ]
            )
        else:
            self.train_trans = trans.Compose(
                train_trans
                + [
                    trans.RandomCrop(self.image_size, padding=4, fill=(128, 128, 128)),
                    trans.RandomHorizontalFlip(),
                    trans.ToTensor(),
                    trans.Normalize(self.mean, self.std),
                ]
            )
        self.val_trans = trans.Compose(
            val_trans
            + [
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
        dataset = ColorDotCIFAR100(
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
        dataset = ColorDotCIFAR100(
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
        dataset = ColorDotCIFAR100(
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


class ColorDot_75_C100Datamodule(ColorDot_100_C100Datamodule):
    dot_correlation = 75


class ColorDot_50_C100Datamodule(ColorDot_100_C100Datamodule):
    dot_correlation = 50


class ColorDot_25_C100Datamodule(ColorDot_100_C100Datamodule):
    dot_correlation = 25


class ColorDot_0_C100Datamodule(ColorDot_100_C100Datamodule):
    dot_correlation = 0
