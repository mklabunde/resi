import os
import random
from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from vision.data.imagenet100_ds import IN100_LABELS
from vision.data.shortcuts.shortcut_transforms import ColorDotShortcut
from vision.util.file_io import load_json


class ColorDotImageNet100Dataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        kfold_split: int,
        transform: Optional[transforms.Compose],
        dot_correlation: int = 100,
        dot_diameter=5,
    ):
        """Creates an instance of the ImageNet Dataset

        :param root: Root folder containing the necessary data & meta files
        :param split: Split indicating if train/val/test images are to be loaded
        :param transform: optional transforms that are to be applied when getting items
        """
        super().__init__()
        assert split in ["train", "val", "test"], "Has to be either 'train' or 'val' "

        self.resize_transform = transforms.Resize((224, 224))
        self.transforms: transforms.Compose = transform
        self.samples: list[tuple[Path, int]] = []
        self.root: Path = Path(root) / "Imagenet100"

        self.max_kfold_split: int = 10
        self.kfold_split = kfold_split

        self.sanity_check()

        metafile = IN100_LABELS
        classes = list(sorted(metafile.keys()))  # Always the same classes
        self.wnid_to_id = {dk: cnt for cnt, dk in enumerate(classes)}

        self.gather_samples(split)
        self.samples = list(sorted(self.samples))
        rng = np.random.default_rng(32)
        rng.shuffle(self.samples)

        # Returns all the samples in tuples of (path, label)
        self._color_sc_gen = ColorDotShortcut(
            n_classes=100,
            n_channels=3,
            image_size=(224, 224),
            dataset_mean=0,
            dataset_std=1,
            correlation_prob=dot_correlation / 100.0,
            dot_diameter=dot_diameter,
        )
        # Save the coordinates and the color for each sample
        self.color_dot_coords = [self._color_sc_gen._get_color_dot_coords(sample[1]) for sample in self.samples]

        return

    def sanity_check(self):
        """Validates that the dataset is present and all samples exist.

        :return:
        """
        assert os.path.exists(self.root), f"Dataset not found at path {self.root}"

        expected_keys = set(IN100_LABELS.keys())
        actual_keys = set(os.listdir(self.root / "train"))

        assert actual_keys.issuperset(expected_keys), f"Expected keys: {expected_keys} Actual keys: {actual_keys}"

        for data_dir, n_data in zip(["train", "val"], [1300, 50]):
            train_data = self.root / data_dir
            train_data_class_dirs = list(train_data.iterdir())
            train_data_class_dirs = [d for d in train_data_class_dirs if d.is_dir()]
            n_dirs = len(train_data_class_dirs)
            if n_dirs != 100:
                raise ValueError(f"Expected 100 directories, found {n_dirs}")
            for data_subdir in train_data_class_dirs:
                samples = [
                    s
                    for s in list(data_subdir.iterdir())
                    if (s.name.endswith(".JPEG")) and (not s.name.startswith("._"))
                ]
                if len(samples) != n_data:
                    raise ValueError(
                        f"Expected {n_data} {data_dir} images! " f"Found {len(samples)} in {data_subdir.name}"
                    )

        return

    def gather_samples(self, split: str):
        """Loads samples into the self.samples list.
        Contains [image_path, class_id].

        :return:
        """
        data_root_dir = self.root
        if split in ["train", "val"]:
            data_dir = data_root_dir / "train"
        elif split == "test":
            data_dir = data_root_dir / "val"
        else:
            raise ValueError(f"Got faulty split: {split} passed.")

        all_samples = []
        for wnid, class_id in self.wnid_to_id.items():
            class_path = data_dir / wnid
            images: list[tuple[Path, int]] = [
                (cp, class_id) for cp in class_path.iterdir() if cp.name.endswith(".JPEG")
            ]
            all_samples.extend(images)
        self.samples = all_samples
        logger.info(f"Using '{len(self.samples)}' of split '{split}' samples")
        logger.info(f"Shape of samples: '{len(self.samples[0])}' ")
        assert len(self.samples) > 0, f"No samples found in the dataset. Checked path: {data_dir}"
        return

    def __getitem__(self, item: int) -> tuple[Any, int, int]:
        try:
            im: Image.Image = Image.open(self.samples[item][0])
        except IndexError as e:
            logger.info(f"Item id: {item}")
            logger.info(f"Length of samples: {len(self.samples)}")
            logger.info(f"Shape {len(self.samples[0])}")
            raise e

        if im.mode != "RGB":
            im = im.convert("RGB")
        im_resized = self.resize_transform(im)
        x_center, y_center, color, color_label, cls_lbl = self.color_dot_coords[item]
        sc_mask, color_mask = self._color_sc_gen._color_dot_from_coords(x_center, y_center, color, dtype=np.uint8)
        im_np_resized = self._color_sc_gen.apply_shortcut(np.array(im_resized), sc_mask, color_mask)
        trans_im = self.transforms(Image.fromarray(im_np_resized, mode="RGB"))
        lbl = self.samples[item][1]
        return trans_im, lbl, int(color_label)

    def __len__(self) -> int:
        return len(self.samples)
