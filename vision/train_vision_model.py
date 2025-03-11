from argparse import ArgumentParser
from functools import partial

from loguru import logger
from vision.arch.arch_loading import load_model_from_info_file
from vision.data.random_labels.rl_c10_dm import RandomLabel_CIFAR10DataModule
from vision.losses.dummy_loss import DummyLoss
from vision.training.ke_train_modules.base_training_module import BaseLightningModule
from vision.training.ke_train_modules.shortcut_lightning_module import ShortcutLightningModule
from vision.training.trainers.base_trainer import BaseTrainer
from vision.training.trainers.shortcut_trainer import ShortcutTrainer
from vision.util import data_structs as ds
from vision.util import default_params as dp
from vision.util import find_architectures as fa
from vision.util import find_datamodules as fd
from vision.util.default_parser_args import add_vision_training_params
from vision.util.file_io import get_vision_model_info

STANDARD_DATAMODULES = [
    ds.Dataset.TinyIMAGENET,
    ds.Dataset.CIFAR10,
    ds.Dataset.CIFAR100,
    ds.Dataset.IMAGENET100,
]

SHORTCUT_DATAMODULES = [
    ds.Dataset.CDOT100,
    ds.Dataset.CDOT75,
    ds.Dataset.CDOT50,
    ds.Dataset.CDOT25,
    ds.Dataset.CDOT0,
]

C100_SHORTCUT_DATAMODULES = [
    ds.Dataset.C100CDOT100,
    ds.Dataset.C100CDOT75,
    ds.Dataset.C100CDOT50,
    ds.Dataset.C100CDOT25,
    ds.Dataset.C100CDOT0,
]

IN_SHORTCUT_DATAMODULES = [
    ds.Dataset.INCDOT100,
    ds.Dataset.INCDOT75,
    ds.Dataset.INCDOT50,
    ds.Dataset.INCDOT25,
    ds.Dataset.INCDOT0,
]

IN_RANDOMLABEL_DATAMODULES = [
    ds.Dataset.INRLABEL100,
    ds.Dataset.INRLABEL75,
    ds.Dataset.INRLABEL50,
    ds.Dataset.INRLABEL25,
]

C100_RANDOMLABEL_DATAMODULES = [
    ds.Dataset.C100RLABEL100,
    ds.Dataset.C100RLABEL75,
    ds.Dataset.C100RLABEL50,
    ds.Dataset.C100RLABEL25,
]

AUGMENTATION_DATAMODULES = [
    ds.Dataset.GaussMAX,
    ds.Dataset.GaussL,
    ds.Dataset.GaussM,
    ds.Dataset.GaussS,
    ds.Dataset.GaussOff,
]

C100_AUGMENTATION_DATAMODULES = [
    ds.Dataset.C100GaussMAX,
    ds.Dataset.C100GaussL,
    ds.Dataset.C100GaussM,
    ds.Dataset.C100GaussS,
    ds.Dataset.C100GaussOff,
]

IN_AUGMENTATION_DATAMODULES = [
    ds.Dataset.INGaussMAX,
    ds.Dataset.INGaussL,
    ds.Dataset.INGaussM,
    ds.Dataset.INGaussS,
    ds.Dataset.INGaussOff,
]


def load_model_and_datamodule(model_info: ds.ModelInfo, is_vit: bool):
    """Load instances of the model and the datamodule from the infos of the info_file."""
    datamodule = fd.get_datamodule(dataset=model_info.dataset, is_vit=is_vit)
    params = dp.get_default_parameters(model_info.architecture, model_info.dataset)
    arch_kwargs = dp.get_default_arch_params(model_info.dataset, is_vit)
    if model_info.info_file_exists():
        loaded_model = load_model_from_info_file(model_info)
    else:
        architecture = fa.get_base_arch(model_info.architecture)
        loaded_model = architecture(**arch_kwargs)
    return loaded_model, datamodule, params, arch_kwargs


def train_vision_model(
    architecture_name: str, train_dataset: str, seed_id: int, setting_identifier: str, overwrite: bool = False
):
    model_info: ds.ModelInfo = get_vision_model_info(
        architecture_name=architecture_name,
        dataset=train_dataset,
        seed_id=seed_id,
        setting_identifier=setting_identifier,
    )

    is_vit = False
    if architecture_name in ["ViT_B32", "ViT_L32"]:
        is_vit = True

    if model_info.finished_training() and not overwrite:
        logger.info("Model already trained, skipping.")
        return  # No need to train the model again if it exists

    loaded_model, datamodule, params, arch_params = load_model_and_datamodule(model_info, is_vit=is_vit)
    if ds.Dataset(train_dataset) in (SHORTCUT_DATAMODULES + C100_SHORTCUT_DATAMODULES + IN_SHORTCUT_DATAMODULES):
        lnm_cls = ShortcutLightningModule
        no_sc_dm, full_sc_dm = fd.get_min_max_shortcut_datamodules(train_dataset, is_vit=is_vit)
        trainer_cls = partial(ShortcutTrainer, no_sc_datamodule=no_sc_dm, full_sc_datamodule=full_sc_dm)
    else:
        lnm_cls = BaseLightningModule
        trainer_cls = BaseTrainer

    if isinstance(datamodule, RandomLabel_CIFAR10DataModule):
        datamodule.rng_seed = (seed_id + 1) * 123  # Different rng seeds for different seeds.

    lightning_mod = lnm_cls(
        model_info=model_info,
        network=loaded_model,
        save_checkpoints=True,
        params=params,
        hparams=arch_params,
        loss=DummyLoss(),
        log=True,
    )

    trainer = trainer_cls(
        model=lightning_mod,
        datamodule=datamodule,
        model_info=model_info,
        arch_params=arch_params,
    )
    just_load_checkpoint = False
    if model_info.has_final_metrics():
        just_load_checkpoint = True
    trainer.train(just_load_checkpoint)


if __name__ == "__main__":
    parser = ArgumentParser()
    add_vision_training_params(parser)
    args = parser.parse_args()
    train_vision_model(args.architecture, args.dataset, args.seed, args.setting_identifier, args.overwrite)
