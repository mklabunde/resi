from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Literal

from vision.util import name_conventions as nc
from vision.util.status_check import output_json_has_nans

# Static Naming conventions for writing out files and finding the files again.

logger = logging.getLogger(__name__)


def load_json(filepath: str | Path) -> Any:
    """Load the json again

    :param filepath:
    :return:
    """
    with open(str(filepath)) as f:
        ret = json.load(f)
    return ret


@dataclass(frozen=True)
class GroupMetrics:
    all_to_all: list[list[float]] | object
    all_to_all_mean: float
    last_to_others: list[float] | object
    last_to_others_mean: float
    last_to_first: float


class Dataset(Enum):
    """Info which dataset should be used"""

    TEST = "TEST"
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"
    SPLITCIFAR100 = "SplitCIFAR100"
    IMAGENET = "ImageNet"
    IMAGENET100 = "ImageNet100"
    DermaMNIST = "DermaMNIST"
    TinyIMAGENET = "TinyImageNet"
    RandomLabelC10 = "RandomLabelCIFAR10"
    CDOT100 = "ColorDot_100_CIFAR10DataModule"
    CDOT75 = "ColorDot_75_CIFAR10DataModule"
    CDOT50 = "ColorDot_50_CIFAR10DataModule"
    CDOT25 = "ColorDot_25_CIFAR10DataModule"
    CDOT0 = "ColorDot_0_CIFAR10DataModule"
    INCDOT100 = "ColorDot_100_ImageNet100DataModule"
    INCDOT75 = "ColorDot_75_ImageNet100DataModule"
    INCDOT50 = "ColorDot_50_ImageNet100DataModule"
    INCDOT25 = "ColorDot_25_ImageNet100DataModule"
    INCDOT0 = "ColorDot_0_ImageNet100DataModule"
    GaussMAX = "Gauss_Max_CIFAR10DataModule"
    GaussL = "Gauss_L_CIFAR10DataModule"
    GaussM = "Gauss_M_CIFAR10DataModule"
    GaussS = "Gauss_S_CIFAR10DataModule"
    GaussOff = "Gauss_Off_CIFAR10DataModule"
    INGaussMAX = "Gauss_Max_ImageNet100DataModule"
    INGaussL = "Gauss_L_ImageNet100DataModule"
    INGaussM = "Gauss_M_ImageNet100DataModule"
    INGaussS = "Gauss_S_ImageNet100DataModule"
    INGaussOff = "Gauss_Off_ImageNet100DataModule"
    INRLABEL100 = "RandomLabel_100_IN100_DataModule"
    INRLABEL75 = "RandomLabel_75_IN100_DataModule"
    INRLABEL50 = "RandomLabel_50_IN100_DataModule"
    INRLABEL25 = "RandomLabel_25_IN100_DataModule"


class BaseArchitecture(Enum):
    VGG11 = "VGG11"
    VGG16 = "VGG16"
    VGG19 = "VGG19"
    DYNVGG19 = "DYNVGG19"
    RESNET18 = "ResNet18"
    RESNET34 = "ResNet34"
    RESNET50 = "ResNet50"
    RESNET101 = "ResNet101"
    DYNRESNET101 = "DYNResNet101"
    DENSENET121 = "DenseNet121"
    DENSENET161 = "DenseNet161"
    VIT_B16 = "ViT_B16"
    VIT_B32 = "ViT_B32"
    VIT_L16 = "ViT_L16"
    VIT_L32 = "ViT_L32"


class Augmentation(Enum):
    TRAIN = "train"
    VAL = "val"
    NONE = "none"


@dataclass
class ArchitectureInfo:
    arch_type_str: str
    arch_kwargs: dict
    checkpoint: str | Path | None
    hooks: tuple[Hook] | None


@dataclass(frozen=True)
class ModelInfo:
    """
    Most important ModelInfo class. This contains all the parameters,
    paths and other information needed to load a model and access its results.
    """

    # Basic information
    seed: int
    architecture: str
    dataset: str
    setting_identifier: str

    # Basic paths
    path_root: Path
    path_ckpt: Path = field(init=False)
    path_activations: Path = field(init=False)

    path_pd_train: Path = field(init=False)
    path_pd_test: Path = field(init=False)
    path_gt_train: Path = field(init=False)
    path_gt_test: Path = field(init=False)
    path_output_json: Path = field(init=False)
    path_last_metrics_json: Path = field(init=False)
    path_train_log: Path = field(init=False)
    path_train_info_json: Path = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "path_ckpt", self.path_root / nc.CKPT_DIR_NAME / nc.STATIC_CKPT_NAME)
        object.__setattr__(self, "path_activations", self.path_root / nc.ACTI_DIR_NAME)
        object.__setattr__(self, "path_pd_train", self.path_root / nc.ACTI_DIR_NAME / nc.MODEL_TRAIN_PD_TMPLT)
        object.__setattr__(self, "path_pd_test", self.path_root / nc.ACTI_DIR_NAME / nc.MODEL_TEST_PD_TMPLT)
        object.__setattr__(self, "path_gt_train", self.path_root / nc.ACTI_DIR_NAME / nc.MODEL_TRAIN_GT_TMPLT)
        object.__setattr__(self, "path_gt_test", self.path_root / nc.ACTI_DIR_NAME / nc.MODEL_TEST_GT_TMPLT)
        object.__setattr__(self, "path_output_json", self.path_root / nc.OUTPUT_TMPLT)
        object.__setattr__(self, "path_last_metrics_json", self.path_root / nc.LAST_METRICS_TMPLT)
        object.__setattr__(self, "path_train_log", self.path_root / nc.LOG_DIR)
        object.__setattr__(self, "path_train_info_json", self.path_root / nc.KE_INFO_FILE)

    def has_checkpoint(self):
        """Checks whether model has a checkpoint."""
        return self.path_ckpt.exists()

    def finished_training(self) -> bool:
        """Checks whether the model has finished training."""
        return self.path_output_json.exists()

    def has_final_metrics(self) -> bool:
        """
        Check if the model has `final_metrics.json` written.
        This is an indicator that the final test_set evaluation crashed
        (e.g. due to not pulling changes to test_dataloader in-time...)
        """
        return self.path_last_metrics_json.exists()

    def info_file_exists(self) -> bool:
        """Checks whether the info file exists."""
        return self.path_train_info_json.exists()

    def has_predictions(self) -> bool:
        """
        Returns true if model has prediction logits and groundtruths of the test set already.
        """
        preds = self.path_predictions_test
        gts = self.path_groundtruths_test
        return preds.exists() and gts.exists()


@dataclass
class Params:
    """Dataclass containing all hyperparameters needed for a basic training"""

    num_epochs: int
    batch_size: int
    label_smoothing: bool
    label_smoothing_val: float
    architecture_name: str
    save_last_checkpoint: bool
    momentum: float
    learning_rate: float
    nesterov: bool
    weight_decay: float
    cosine_annealing: bool
    gamma: float
    split: int
    dataset: str
    advanced_da: bool = True
    optimizer: dict[str, Any] | None = None
    gradient_clip: float | None = None


@dataclass
class Hook:
    architecture_index: int
    name: str
    keys: list[str]
    n_channels: int = 0
    downsampling_steps: int = -1  # == Undefined! - Sometimes dynamic --> Set when initialized the model!
    resolution: tuple[int, int] = (0, 0)  # == Undefined! - Depends on Dataset --> Set when initialized!
    resolution_relative_depth: float = -1
    at_input: bool = False
