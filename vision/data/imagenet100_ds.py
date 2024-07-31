import json
import os
import random
import shutil
from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from vision.util.file_io import load_json

IN100_LABELS = {
    "n01968897": "chambered nautilus, pearly nautilus, nautilus",
    "n01770081": "harvestman, daddy longlegs, Phalangium opilio",
    "n01818515": "macaw",
    "n02011460": "bittern",
    "n01496331": "electric ray, crampfish, numbfish, torpedo",
    "n01847000": "drake",
    "n01687978": "agama",
    "n01740131": "night snake, Hypsiglena torquata",
    "n01537544": "indigo bunting, indigo finch, indigo bird, Passerina cyanea",
    "n01491361": "tiger shark, Galeocerdo cuvieri",
    "n02007558": "flamingo",
    "n01735189": "garter snake, grass snake",
    "n01630670": "common newt, Triturus vulgaris",
    "n01440764": "tench, Tinca tinca",
    "n01819313": "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita",
    "n02002556": "white stork, Ciconia ciconia",
    "n01667778": "terrapin",
    "n01755581": "diamondback, diamondback rattlesnake, Crotalus adamanteus",
    "n01924916": "flatworm, platyhelminth",
    "n01751748": "sea snake",
    "n01984695": "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish",
    "n01729977": "green snake, grass snake",
    "n01614925": "bald eagle, American eagle, Haliaeetus leucocephalus",
    "n01608432": "kite",
    "n01443537": "goldfish, Carassius auratus",
    "n01770393": "scorpion",
    "n01855672": "goose",
    "n01560419": "bulbul",
    "n01592084": "chickadee",
    "n01914609": "sea anemone, anemone",
    "n01582220": "magpie",
    "n01667114": "mud turtle",
    "n01985128": "crayfish, crawfish, crawdad, crawdaddy",
    "n01820546": "lorikeet",
    "n01773797": "garden spider, Aranea diademata",
    "n02006656": "spoonbill",
    "n01986214": "hermit crab",
    "n01484850": "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",
    "n01749939": "green mamba",
    "n01828970": "bee eater",
    "n02018795": "bustard",
    "n01695060": "Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis",
    "n01729322": "hognose snake, puff adder, sand viper",
    "n01677366": "common iguana, iguana, Iguana iguana",
    "n01734418": "king snake, kingsnake",
    "n01843383": "toucan",
    "n01806143": "peacock",
    "n01773549": "barn spider, Araneus cavaticus",
    "n01775062": "wolf spider, hunting spider",
    "n01728572": "thunder snake, worm snake, Carphophis amoenus",
    "n01601694": "water ouzel, dipper",
    "n01978287": "Dungeness crab, Cancer magister",
    "n01930112": "nematode, nematode worm, roundworm",
    "n01739381": "vine snake",
    "n01883070": "wombat",
    "n01774384": "black widow, Latrodectus mactans",
    "n02037110": "oystercatcher, oyster catcher",
    "n01795545": "black grouse",
    "n02027492": "red-backed sandpiper, dunlin, Erolia alpina",
    "n01531178": "goldfinch, Carduelis carduelis",
    "n01944390": "snail",
    "n01494475": "hammerhead, hammerhead shark",
    "n01632458": "spotted salamander, Ambystoma maculatum",
    "n01698640": "American alligator, Alligator mississipiensis",
    "n01675722": "banded gecko",
    "n01877812": "wallaby, brush kangaroo",
    "n01622779": "great grey owl, great gray owl, Strix nebulosa",
    "n01910747": "jellyfish",
    "n01860187": "black swan, Cygnus atratus",
    "n01796340": "ptarmigan",
    "n01833805": "hummingbird",
    "n01685808": "whiptail, whiptail lizard",
    "n01756291": "sidewinder, horned rattlesnake, Crotalus cerastes",
    "n01514859": "hen",
    "n01753488": "horned viper, cerastes, sand viper, horned asp, Cerastes cornutus",
    "n02058221": "albatross, mollymawk",
    "n01632777": "axolotl, mud puppy, Ambystoma mexicanum",
    "n01644900": "tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui",
    "n02018207": "American coot, marsh hen, mud hen, water hen, Fulica americana",
    "n01664065": "loggerhead, loggerhead turtle, Caretta caretta",
    "n02028035": "redshank, Tringa totanus",
    "n02012849": "crane",
    "n01776313": "tick",
    "n02077923": "sea lion",
    "n01774750": "tarantula",
    "n01742172": "boa constrictor, Constrictor constrictor",
    "n01943899": "conch",
    "n01798484": "prairie chicken, prairie grouse, prairie fowl",
    "n02051845": "pelican",
    "n01824575": "coucal",
    "n02013706": "limpkin, Aramus pictus",
    "n01955084": "chiton, coat-of-mail shell, sea cradle, polyplacophore",
    "n01773157": "black and gold garden spider, Argiope aurantia",
    "n01665541": "leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea",
    "n01498041": "stingray",
    "n01978455": "rock crab, Cancer irroratus",
    "n01693334": "green lizard, Lacerta viridis",
    "n01950731": "sea slug, nudibranch",
    "n01829413": "hornbill",
    "n01514668": "cock",
}


class ImageNet100Dataset(Dataset):
    def __init__(self, root: str | Path, split: str, kfold_split: int, transform: Optional[transforms.Compose]):
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
        ], "Has to be either 'train', 'val' or test"

        self.transforms: transforms.Compose = transform
        self.samples: list[tuple[Path, int]] = []
        self.root: Path = Path(root) / "Imagenet100"

        self.max_kfold_split: int = 10
        self.kfold_split = kfold_split

        self.sanity_check()

        metafile = IN100_LABELS
        classes = list(sorted(metafile.keys()))  # Always the same classes
        self.wnid_to_id = {dk: cnt for cnt, dk in enumerate(classes)}

        # Returns all the samples in tuples of (path, label)
        self.gather_samples(split)
        if split in ["train", "val"]:
            self.draw_kfold_subset(split, kfold_split)
        self.samples = list(sorted(self.samples))
        rng = np.random.default_rng(32)
        rng.shuffle(self.samples)
        return

    def draw_kfold_subset(self, split: str, kf_split: int) -> None:
        """Draws a split from the class in deterministic fashion.

        :param split: Split to draw
        :param kf_split: Use the kfold split to train/val
        :return:
        """
        tmp_samples = []
        for wnid in self.wnid_to_id.values():
            current_samples = [sample for sample in self.samples if sample[1] == wnid]
            n_cur_samples = len(current_samples)
            if kf_split == self.max_kfold_split:
                max_id_to_draw = n_cur_samples
            else:
                max_id_to_draw = (n_cur_samples // self.max_kfold_split) * (kf_split + 1)
            min_id_to_draw = (n_cur_samples // self.max_kfold_split) * kf_split
            val_samples = set(current_samples[min_id_to_draw:max_id_to_draw])
            train_samples = set(current_samples) - val_samples
            if split == "val":
                tmp_samples.extend(list(val_samples))
            else:
                tmp_samples.extend(list(train_samples))
        self.samples = tmp_samples
        assert len(self.samples) > 0, f"No samples remain in dataset after kfold subset drawing."

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

        logger.info(f"Collecting samples from {data_dir}")
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

    def __getitem__(self, item: int) -> tuple[Any, int]:
        im: Image.Image = Image.open(self.samples[item][0])
        if im.mode != "RGB":
            im = im.convert("RGB")
        trans_im = self.transforms(im)
        lbl = self.samples[item][1]

        return trans_im, lbl

    def __len__(self) -> int:
        return len(self.samples)


# Deprecated
# Dataset was originally downloaded from Kaggle
# So download here https://www.kaggle.com/datasets/ambityga/imagenet100/data
# Then merge the 'train.X1' to 'train.X4' to one 'train' folder
# Also rename 'val.X1' to 'val'
def create_IN100_datset_from_IN1k(in100_outpath: Path, path_to_in1k: str | Path):

    expected_path_to_train = ["ILSVRC", "Data", "CLS-LOC", "train"]
    train_path = Path(path_to_in1k)
    for p in expected_path_to_train:
        train_path = train_path / p
        assert train_path.exists(), f"Path {train_path} does not exist."
    expected_val_path = train_path.parent / "val"
    assert expected_val_path.exists(), f"Path {expected_val_path} does not exist."

    wnids = list(IN100_LABELS.keys())

    in100_outpath.mkdir(exist_ok=True, parents=True)
    for split, path in zip(["train", "val"], [train_path, expected_val_path]):
        for wnid in tqdm(wnids, desc="Copying 100 IN1k classes to create IN100."):
            outpath = in100_outpath / split / wnid
            outpath.mkdir(exist_ok=True, parents=True)
            for img in (path / wnid).iterdir():
                img_outpath = outpath / img.name
                shutil.copy(img, img_outpath)


if __name__ == "__main__":
    # Test the ImageNet100Dataset
    root = Path(os.environ["RAW_DATA"]) / "Imagenet100" / "train"
    wnids = os.listdir(root)
    print(json.dumps(wnids, indent=4))
