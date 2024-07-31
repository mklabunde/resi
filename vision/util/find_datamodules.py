from vision.data.base_datamodule import BaseDataModule
from vision.data.cifar100_dm import CIFAR100DataModule
from vision.data.cifar10_dm import CIFAR10DataModule
from vision.data.higher_aug.higher_aug_dm import Gauss_L_CIFAR10DataModule
from vision.data.higher_aug.higher_aug_dm import Gauss_M_CIFAR10DataModule
from vision.data.higher_aug.higher_aug_dm import Gauss_Max_CIFAR10DataModule
from vision.data.higher_aug.higher_aug_dm import Gauss_Off_CIFAR10DataModule
from vision.data.higher_aug.higher_aug_dm import Gauss_S_CIFAR10DataModule
from vision.data.higher_aug.higher_aug_IN100_dm import Gauss_L_Imagenet100DataModule
from vision.data.higher_aug.higher_aug_IN100_dm import Gauss_M_Imagenet100DataModule
from vision.data.higher_aug.higher_aug_IN100_dm import Gauss_Max_Imagenet100DataModule
from vision.data.higher_aug.higher_aug_IN100_dm import Gauss_Off_Imagenet100DataModule
from vision.data.higher_aug.higher_aug_IN100_dm import Gauss_S_Imagenet100DataModule
from vision.data.imagenet100_dm import Imagenet100DataModule
from vision.data.imagenet_dm import ImagenetDataModule
from vision.data.medmnist_dm import DermaMNISTDataModule
from vision.data.random_labels.rl_c10_dm import RandomLabel_CIFAR10DataModule
from vision.data.random_labels.rl_in100_dm import RandomLabel_100_IN100_DataModule
from vision.data.random_labels.rl_in100_dm import RandomLabel_25_IN100_DataModule
from vision.data.random_labels.rl_in100_dm import RandomLabel_50_IN100_DataModule
from vision.data.random_labels.rl_in100_dm import RandomLabel_75_IN100_DataModule
from vision.data.shortcuts.sc_cifar10_dm import ColorDot_0_CIFAR10DataModule
from vision.data.shortcuts.sc_cifar10_dm import ColorDot_100_CIFAR10DataModule
from vision.data.shortcuts.sc_cifar10_dm import ColorDot_25_CIFAR10DataModule
from vision.data.shortcuts.sc_cifar10_dm import ColorDot_50_CIFAR10DataModule
from vision.data.shortcuts.sc_cifar10_dm import ColorDot_75_CIFAR10DataModule
from vision.data.shortcuts.sc_in100_dm import ColorDot_0_IN100Datamodule
from vision.data.shortcuts.sc_in100_dm import ColorDot_100_IN100Datamodule
from vision.data.shortcuts.sc_in100_dm import ColorDot_25_IN100Datamodule
from vision.data.shortcuts.sc_in100_dm import ColorDot_50_IN100Datamodule
from vision.data.shortcuts.sc_in100_dm import ColorDot_75_IN100Datamodule
from vision.data.test_dm import TestDataModule
from vision.data.tiny_imagenet_dm import TinyImagenetDataModule
from vision.util import data_structs as ds


def get_datamodule(dataset: ds.Dataset | str, advanced_da: bool = True) -> BaseDataModule:
    """Returns the datamodule specified by the Dataset and the train/val/test split."""
    if isinstance(dataset, str):
        dataset = ds.Dataset(dataset)
    if dataset == ds.Dataset.CIFAR10:
        return CIFAR10DataModule(advanced_da)
    elif dataset == ds.Dataset.TEST:
        return TestDataModule()
    elif dataset == ds.Dataset.CIFAR100:
        return CIFAR100DataModule(advanced_da)
    elif dataset == ds.Dataset.IMAGENET:
        return ImagenetDataModule(advanced_da)
    elif dataset == ds.Dataset.IMAGENET100:
        return Imagenet100DataModule(advanced_da)
    elif dataset == ds.Dataset.TinyIMAGENET:
        return TinyImagenetDataModule(advanced_da)
    elif dataset == ds.Dataset.DermaMNIST:
        return DermaMNISTDataModule(advanced_da)
    elif dataset == ds.Dataset.SPLITCIFAR100:
        raise NotImplementedError()
    elif dataset == ds.Dataset.RandomLabelC10:
        return RandomLabel_CIFAR10DataModule(advanced_da)
    elif dataset == ds.Dataset.CDOT100:
        return ColorDot_100_CIFAR10DataModule(advanced_da)
    elif dataset == ds.Dataset.CDOT75:
        return ColorDot_75_CIFAR10DataModule(advanced_da)
    elif dataset == ds.Dataset.CDOT50:
        return ColorDot_50_CIFAR10DataModule(advanced_da)
    elif dataset == ds.Dataset.CDOT25:
        return ColorDot_25_CIFAR10DataModule(advanced_da)
    elif dataset == ds.Dataset.CDOT0:
        return ColorDot_0_CIFAR10DataModule(advanced_da)
    elif dataset == ds.Dataset.GaussMAX:
        return Gauss_Max_CIFAR10DataModule(advanced_da)
    elif dataset == ds.Dataset.GaussL:
        return Gauss_L_CIFAR10DataModule(advanced_da)
    elif dataset == ds.Dataset.GaussM:
        return Gauss_M_CIFAR10DataModule(advanced_da)
    elif dataset == ds.Dataset.GaussS:
        return Gauss_S_CIFAR10DataModule(advanced_da)
    elif dataset == ds.Dataset.GaussOff:
        return Gauss_Off_CIFAR10DataModule(advanced_da)
    elif dataset == ds.Dataset.INGaussMAX:
        return Gauss_Max_Imagenet100DataModule(advanced_da)
    elif dataset == ds.Dataset.INGaussL:
        return Gauss_L_Imagenet100DataModule(advanced_da)
    elif dataset == ds.Dataset.INGaussM:
        return Gauss_M_Imagenet100DataModule(advanced_da)
    elif dataset == ds.Dataset.INGaussS:
        return Gauss_S_Imagenet100DataModule(advanced_da)
    elif dataset == ds.Dataset.INGaussOff:
        return Gauss_Off_Imagenet100DataModule(advanced_da)
    elif dataset == ds.Dataset.INCDOT100:
        return ColorDot_100_IN100Datamodule(advanced_da)
    elif dataset == ds.Dataset.INCDOT75:
        return ColorDot_75_IN100Datamodule(advanced_da)
    elif dataset == ds.Dataset.INCDOT50:
        return ColorDot_50_IN100Datamodule(advanced_da)
    elif dataset == ds.Dataset.INCDOT25:
        return ColorDot_25_IN100Datamodule(advanced_da)
    elif dataset == ds.Dataset.INCDOT0:
        return ColorDot_0_IN100Datamodule(advanced_da)
    elif dataset == ds.Dataset.INRLABEL100:
        return RandomLabel_100_IN100_DataModule(advanced_da)
    elif dataset == ds.Dataset.INRLABEL75:
        return RandomLabel_75_IN100_DataModule(advanced_da)
    elif dataset == ds.Dataset.INRLABEL50:
        return RandomLabel_50_IN100_DataModule(advanced_da)
    elif dataset == ds.Dataset.INRLABEL25:
        return RandomLabel_25_IN100_DataModule(advanced_da)
    else:
        raise ValueError


def get_min_max_shortcut_datamodules(
    dataset: ds.Dataset | str, advanced_da: bool = True
) -> tuple[BaseDataModule, BaseDataModule]:
    if isinstance(dataset, str):
        dataset = ds.Dataset(dataset)

    if dataset in [ds.Dataset.CDOT0, ds.Dataset.CDOT75, ds.Dataset.CDOT50, ds.Dataset.CDOT25, ds.Dataset.CDOT100]:
        return get_datamodule(ds.Dataset.CDOT0, advanced_da), get_datamodule(ds.Dataset.CDOT100, advanced_da)
    elif dataset in [
        ds.Dataset.INCDOT0,
        ds.Dataset.INCDOT75,
        ds.Dataset.INCDOT50,
        ds.Dataset.INCDOT25,
        ds.Dataset.INCDOT100,
    ]:
        return get_datamodule(ds.Dataset.INCDOT0, advanced_da), get_datamodule(ds.Dataset.INCDOT100, advanced_da)
    else:
        raise NotImplementedError("Only implemented for shortcut datasets.")
