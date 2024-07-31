import random

from torch.utils.data import DataLoader
from vision.data.imagenet100_dm import Imagenet100DataModule
from vision.data.imagenet100_ds import ImageNet100Dataset
from vision.util import data_structs as ds


# from ke.data import auto_augment

# from ke.randaugment.randaugment  import CIFAR10Policy

# from ke.data.auto_augment import CIFAR10Policy

# from ke.data import cutout_aug


class RandomLabel_100_IN100_DataModule(Imagenet100DataModule):
    random_label_percent: float = 1
    # If I remove split from init call:
    #   I will have to make the Subclasses of the Cifar10BaseMergingModule
    #   override wherever the splitting takes place.
    #   Because this is where the KFold and Disjoint DataModule differ!

    def train_dataloader(self, split: int, transform: ds.Augmentation, **kwargs) -> DataLoader:
        """Get a train dataloader"""
        dataset = ImageNet100Dataset(
            root=self.dataset_path,
            split="train",
            kfold_split=split,
            transform=self.get_transforms(transform),
        )

        samples = dataset.samples

        # Depending on the random ratio,
        do_random_labels = [random.random() < self.random_label_percent for _ in range(len(samples))]
        new_im_lbl_pairs = []
        for (im_path, lbl), do_ in zip(samples, do_random_labels):
            if do_:
                new_im_lbl_pairs.append((im_path, random.randint(0, 99)))
            else:
                new_im_lbl_pairs.append((im_path, lbl))

        dataset.samples = new_im_lbl_pairs

        # INFO: Currently does not differentiate into different folds, as the
        #   Dataset comes with a deliberate validation set.
        return DataLoader(dataset=dataset, **kwargs)


class RandomLabel_75_IN100_DataModule(RandomLabel_100_IN100_DataModule):
    random_label_percent: float = 0.75


class RandomLabel_50_IN100_DataModule(RandomLabel_100_IN100_DataModule):
    random_label_percent: float = 0.5


class RandomLabel_25_IN100_DataModule(RandomLabel_100_IN100_DataModule):
    random_label_percent: float = 0.25
