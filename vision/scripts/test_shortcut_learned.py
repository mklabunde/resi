from argparse import ArgumentParser

import torch
from loguru import logger
from vision.arch.arch_loading import load_model_from_info_file
from vision.losses.dummy_loss import DummyLoss
from vision.training.ke_train_modules.base_training_module import BaseLightningModule
from vision.training.ke_train_modules.shortcut_lightning_module import ShortcutLightningModule
from vision.training.trainers.base_trainer import BaseTrainer
from vision.util import data_structs as ds
from vision.util import default_params as dp
from vision.util import find_architectures as fa
from vision.util import find_datamodules as fd
from vision.util.default_parser_args import add_vision_training_params
from vision.util.file_io import get_vision_model_info

SHORTCUT_DATAMODULES = [
    ds.Dataset.CDOT100,
    ds.Dataset.CDOT75,
    ds.Dataset.CDOT50,
    ds.Dataset.CDOT25,
    ds.Dataset.CDOT0,
]


def load_model_and_datamodule(model_info: ds.ModelInfo, load_ckpt: bool = False, is_vit: bool = False):
    """Load instances of the model and the datamodule from the infos of the info_file."""
    datamodule = fd.get_datamodule(dataset=model_info.dataset)
    params = dp.get_default_parameters(model_info.architecture, model_info.dataset)
    arch_kwargs = dp.get_default_arch_params(model_info.dataset, is_vit)
    if model_info.info_file_exists():
        loaded_model = load_model_from_info_file(model_info, load_ckpt=load_ckpt)
    else:
        architecture = fa.get_base_arch(model_info.architecture)
        loaded_model = architecture(**arch_kwargs)
    return loaded_model, datamodule, params, arch_kwargs


def test_model_learned_shortcut(
    architecture_name: str, train_dataset: str, seed_id: int, setting_identifier: str, overwrite: bool = False
):
    model_info: ds.ModelInfo = get_vision_model_info(
        architecture_name=architecture_name,
        dataset=train_dataset,
        seed_id=seed_id,
        setting_identifier=setting_identifier,
    )

    loaded_model, datamodule, params, arch_params = load_model_and_datamodule(model_info, load_ckpt=True)

    uncorrelated_shortcut_datamodule = fd.get_datamodule(ds.Dataset.CDOT0)
    fully_correlated_shortcut_datamodule = fd.get_datamodule(ds.Dataset.CDOT100)
    # uncorrelated_shortcut_datamodule = fd.get_datamodule(ds.Dataset.CDOT0)

    lightning_mod = ShortcutLightningModule(
        model_info=model_info,
        network=loaded_model,
        save_checkpoints=True,
        params=params,
        hparams=arch_params,
        loss=DummyLoss(),
        log=True,
    )
    trainer = BaseTrainer(
        model=lightning_mod,
        datamodule=datamodule,
        model_info=model_info,
        arch_params=arch_params,
    )

    uncorr_perf = trainer.evaluate(uncorrelated_shortcut_datamodule, mode="val")
    train_perf = trainer.evaluate(datamodule, mode="val")
    fully_corr_perf = trainer.evaluate(fully_correlated_shortcut_datamodule, mode="val")

    return


if __name__ == "__main__":
    parser = ArgumentParser()
    add_vision_training_params(parser)
    args = parser.parse_args()
    test_model_learned_shortcut(args.architecture, args.dataset, args.seed, args.setting_identifier, args.overwrite)
