import os
import random
import re
from typing import Any
from typing import Optional
from warnings import warn

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def read_tiny_imagenet_annotation_txt(path: str, wnid_to_id: dict[str:int]) -> dict[str, int]:
    """Reads the tiny imagenet annotation txt file."""
    with open(path, "r") as fd:
        lines = fd.readlines()
    image_to_class_id = {}
    for line in lines:
        line = line.split("\t")
        image_name = line[0]
        image_class_id = wnid_to_id[line[1]]
        image_to_class_id[image_name] = image_class_id
    return image_to_class_id


def read_tiny_imagenet_wnids(path: str) -> list[str]:
    with open(path, "r") as fd:
        lines = fd.readlines()
    wnids = []
    for line in lines:
        wnids.append(line.strip())
    return wnids


class TinyImageNetDataset(Dataset):
    def __init__(self, root: str, split: str, transform: Optional[transforms.Compose]):
        """Creates an instance of the ImageNet Dataset

        :param root: Root folder containing the necessary data & meta files
        :param split: Split indicating if train/val/test images are to be loaded
        :param transform: optional transforms that are to be applied when getting items
        """
        super().__init__()
        assert split in [
            "train",
            "val",
            "test",
        ], "Has to be either 'train', 'val' or 'test"

        self.transforms: transforms.Compose = transform
        # Contains the Path to the image and the class
        self.samples: list[tuple[str, int]] = []
        self.root = root
        self.split = split

        self.sanity_check()
        wnids_of_interest = read_tiny_imagenet_wnids(os.path.join(root, "wnids.txt"))
        self.wnid_class_ids: dict[str:int] = {entry: cnt for cnt, entry in enumerate(wnids_of_interest)}  #

        self.gather_samples()
        random.shuffle(self.samples)

        return

    def sanity_check(self):
        """Validates that the dataset is present and fully loaded.

        :return:
        """
        read_train_wnids = set(read_tiny_imagenet_wnids(os.path.join(self.root, "wnids.txt")))
        found_train_wnids = set(os.listdir(os.path.join(self.root, "train")))

        assert read_train_wnids.issubset(
            found_train_wnids
        ), f"Not all training files exist! Missing: {read_train_wnids.difference(found_train_wnids)}"

    def gather_samples(self):
        """Loads samples into the self.samples list.
        Contains [image_path, class_id].

        :return:
        """

        if self.split == "train":
            for wnid, class_id in self.wnid_class_ids.items():
                current_class_path = os.path.join(self.root, "train", wnid, "images")
                images: list[str] = os.listdir(current_class_path)
                for image in images:
                    if re.match((wnid + r"_\d{0,3}\.JPEG"), image):
                        self.samples.append((os.path.join(current_class_path, image), class_id))
                    else:
                        warn(f"Found unexpected image: {image}")
        elif self.split in ["val", "test"]:
            image_name_and_lbl = read_tiny_imagenet_annotation_txt(
                os.path.join(self.root, "val", "val_annotations.txt"), self.wnid_class_ids
            )
            val_images_path = os.path.join(self.root, "val", "images")
            self.samples = [(os.path.join(val_images_path, k), v) for k, v in image_name_and_lbl.items()]
        else:
            raise ValueError(f"Got faulty split: {self.split} passed.")
        return

    def __getitem__(self, item: int) -> tuple[Any, int]:
        im: Image.Image = Image.open(self.samples[item][0])
        if im.mode != "RGB":
            im = im.convert("RGB")
        trans_im = self.transforms(im)
        lbl = self.samples[item][1]

        return trans_im, lbl

    def __len__(self) -> int:
        return len(self.samples)
