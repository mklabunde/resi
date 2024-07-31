import os
import random
from typing import Any
from typing import Optional

import scipy.io as sio
import xmltodict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.imagenet import parse_devkit_archive


class ImageNetDataset(Dataset):
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
        self.samples: list[tuple[str, int]] = []
        self.root = root
        self.split = split

        self.sanity_check()

        metafile = os.path.join(root, "ILSVRC2012_devkit_t12", "data", "meta.mat")
        meta = sio.loadmat(metafile, squeeze_me=True)["synsets"][:1000]

        self.wnid_to_id = {entry[1]: entry[0] for entry in meta}  #
        self.classes = [value[0] for value in self.wnid_to_id]
        self.wnids = [value[1] for value in self.wnid_to_id]

        self.gather_samples()
        random.shuffle(self.samples)

        return

    def sanity_check(self):
        """Validates that the dataset is present and fully loaded.

        :return:
        """
        # Meta data loading
        if not os.path.exists(os.path.join(self.root, "meta.bin")):
            parse_devkit_archive(self.root)

        base_data_dir = os.path.join(self.root, "ILSVRC", "Data", "CLS-LOC")
        if self.split == "train":
            train_data = os.path.join(base_data_dir, "train")
            n_dirs = len(os.listdir(train_data))
            if n_dirs != 1000:
                raise ValueError(f"Expected 1000 directories, found {n_dirs}")
        elif self.split in ["val", "test"]:
            val_data = os.path.join(base_data_dir, "val")
            n_samples = len(os.listdir(val_data))
            if n_samples != 50000:
                raise ValueError(f"Expected 1000 directories, found {n_samples}")

        return

    def gather_samples(self):
        """Loads samples into the self.samples list.
        Contains [image_path, class_id].

        :return:
        """
        data_root_dir = os.path.join(self.root, "ILSVRC", "Data", "CLS-LOC")
        ann_root_dir = os.path.join(self.root, "ILSVRC", "Annotations", "CLS-LOC")
        if self.split == "train":
            data_dir = os.listdir(os.path.join(data_root_dir, "train"))
            for wnid in data_dir:
                current_class_path = os.path.join(data_root_dir, "train", wnid)
                images: list[str] = os.listdir(current_class_path)
                class_id = self.wnid_to_id[wnid] - 1
                for image in images:
                    self.samples.append((os.path.join(current_class_path, image), class_id))
        elif self.split in ["val", "test"]:
            data_dir = os.path.join(data_root_dir, "val")
            ann_dir = os.path.join(ann_root_dir, "val")
            for content in os.listdir(ann_dir):
                with open(os.path.join(ann_dir, content), "rb") as fd:
                    content = xmltodict.parse(fd, force_list={"object"})
                    filename = content["annotation"]["filename"]
                    class_ids = []
                    for obj in content["annotation"]["object"]:
                        class_ids.append(self.wnid_to_id[obj["name"]])
                    assert all(
                        [class_ids[0] == cid for cid in class_ids]
                    ), f"class ids are not the same for case {filename}"
                    class_id = self.wnid_to_id[content["annotation"]["object"][0]["name"]] - 1
                image_path = os.path.join(data_dir, filename + ".JPEG")
                self.samples.append((image_path, class_id))
        # elif self.split == "test":
        #     data_dir = os.path.join(data_root_dir, "test")
        #     for content in os.listdir(data_dir):
        #         self.samples.append((os.path.join(data_dir, content), -1))
        #         # INFO: -1 as label if no class given
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
