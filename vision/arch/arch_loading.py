from __future__ import annotations

from pathlib import Path
from typing import Type

import torch
from vision.arch.abstract_acti_extr import AbsActiExtrArch
from vision.arch.arch_utils import deserialize_architecture_info
from vision.arch.ke_architectures.feature_approximation import FAArch
from vision.util import data_structs as ds
from vision.util import file_io
from vision.util import find_architectures as fa
from vision.util import name_conventions as nc
from vision.util.data_structs import ArchitectureInfo
from vision.util.file_io import load_json
from vision.util.find_architectures import get_base_arch


def instantiate_kemodule_from_path(source_data_path: Path, source_ckpt_path: Path):
    """Loads the FAArch structure only no checkpoints loaded!."""
    ckpt_dir_path = source_ckpt_path / nc.CKPT_DIR_NAME
    approx_infos: list[dict] = []
    approx_ckpts: list[Path] = []
    for i in range(int((len(list(ckpt_dir_path.iterdir())) - 1) / 2)):
        approx_infos.append(file_io.load(ckpt_dir_path / nc.APPROX_CKPT_INFO_NAME.format(i)))
        approx_ckpts.append(ckpt_dir_path / nc.APPROX_CKPT_NAME.format(i))
    src_arch_infos = [deserialize_architecture_info(i) for i in approx_infos[0]["architecture_infos"]]
    for arch_info in src_arch_infos:
        arch_info.checkpoint = arch_info.checkpoint.replace(
            "/dkfz/cluster/gpu/checkpoints/OE0441/t006d", "/mnt/cluster-checkpoint"
        )
    output_json = file_io.load_json(source_data_path / nc.OUTPUT_TMPLT)
    info_json = file_io.load_json(source_data_path / nc.KE_INFO_FILE)

    tbt_arch_info = ArchitectureInfo(
        arch_type_str=output_json["architecture_name"],
        arch_kwargs={
            "n_cls": output_json["n_cls"],
            "in_ch": output_json["in_ch"],
            "input_resolution": output_json["input_resolution"],
            "early_downsampling": output_json["early_downsampling"],
            "global_average_pooling": output_json["global_average_pooling"],
        },
        checkpoint=None,
        hooks=src_arch_infos[0].hooks,  # Assumes hooks are the same!
    )

    ke_module = FAArch(
        old_model_info=src_arch_infos,
        new_model_info=tbt_arch_info,
        aggregate_old_reps=info_json["aggregate_source_reps"],
        transfer_depth=info_json["trans_depth"],
        transfer_kernel_width=info_json["trans_kernel"],
    )

    return ke_module


def load_kemodule_model(source_data_path: Path, source_ckpt_path: Path) -> FAArch:
    """Loads the FAArch and
    the checkpoints needed to restore behavior as at end of training."""
    ckpt_dir_path = source_ckpt_path / nc.CKPT_DIR_NAME
    approx_infos: list[dict] = []
    approx_ckpts: list[Path] = []
    for i in range(int((len(list(ckpt_dir_path.iterdir())) - 1) / 2)):
        approx_infos.append(file_io.load(ckpt_dir_path / nc.APPROX_CKPT_INFO_NAME.format(i)))
        approx_ckpts.append(ckpt_dir_path / nc.APPROX_CKPT_NAME.format(i))
    src_arch_infos = [deserialize_architecture_info(i) for i in approx_infos[0]["architecture_infos"]]
    info_json = file_io.load_json(source_data_path / nc.KE_INFO_FILE)

    ke_module = instantiate_kemodule_from_path(source_data_path, source_ckpt_path)
    ke_module.load_individual_state_dicts(
        tbt_ckpt=info_json["path_ckpt"].replace(
            "/dkfz/cluster/gpu/checkpoints/OE0441/t006d", "/mnt/cluster-checkpoint"
        ),
        approx_layer_ckpts=approx_ckpts,
        source_arch_ckpts=[
            ai.checkpoint.replace("/dkfz/cluster/gpu/checkpoints/OE0441/t006d", "/mnt/cluster-checkpoint")
            for ai in src_arch_infos
        ],
    )

    return ke_module


def load_model(source_data_path: Path, source_ckpt_path: Path) -> AbsActiExtrArch:
    """
    Loads the architecture and checkpoint based off the path given and returns the initialized
    ActivationExtractionArchitecture!
    """
    oj = load_json(source_data_path / nc.OUTPUT_TMPLT)
    architecture_class = get_base_arch(ds.BaseArchitecture(oj["architecture_name"]))
    architecture_inst: AbsActiExtrArch = architecture_class(
        n_cls=oj["n_cls"],
        in_ch=oj["in_ch"],
        input_resolution=oj["input_resolution"],
        early_downsampling=oj["early_downsampling"],
        global_average_pooling=oj["global_average_pooling"],
    )
    ckpt_path = source_ckpt_path / nc.CKPT_DIR_NAME / nc.STATIC_CKPT_NAME
    architecture_inst.load_state_dict(torch.load(ckpt_path))
    return architecture_inst


def strip_state_dict_of_keys(state_dict: dict) -> dict:
    """Removes the `net` value from the keys in the state_dict

    Example: original contains: "net.features.0.weight"
        current model expects: "features.0.weight"

    :return:
    """
    new_dict = {}
    for key, val in state_dict.items():
        new_dict[".".join(key.split(".")[1:])] = val

    return new_dict


def load_model_from_info_file(model_info: ds.ModelInfo, load_ckpt: bool) -> AbsActiExtrArch:
    """Loads model from a BasicTrainingInfo file.
    :param model_info: Model configuration file to load from.
    :param load_ckpt: Flag if the ckpt should be loaded as well.
    """
    arch: Type[AbsActiExtrArch] = fa.get_base_arch(model_info.architecture)
    oj = load_json(model_info.path_output_json)
    arch_instance: AbsActiExtrArch = arch(
        n_cls=oj["n_cls"],
        in_ch=oj["in_ch"],
        input_resolution=oj["input_resolution"],
        early_downsampling=oj["early_downsampling"],
        global_average_pooling=oj["global_average_pooling"],
    )

    if load_ckpt:
        try:
            arch_instance.load_state_dict(torch.load(model_info.path_ckpt, map_location="cpu"))
        except RuntimeError:
            try:
                arch_instance.load_state_dict(
                    strip_state_dict_of_keys(torch.load(model_info.path_ckpt, map_location="cpu"))
                )
            except RuntimeError as e1:
                raise e1
    if torch.cuda.is_available():
        arch_instance = arch_instance.cuda()

    return arch_instance
