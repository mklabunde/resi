from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from vision.arch.abstract_acti_extr import AbsActiExtrArch
from vision.util.data_structs import ArchitectureInfo
from vision.util.data_structs import Hook


def create_module(
    arch: AbsActiExtrArch,
    ckpt: str | Path | None = None,
    first_hook: Hook | None = None,
    second_hook: Hook | None = None,
) -> nn.Module:
    """
    Creates a partial nn.Module from a given AbsActiExtrArch (AbstractActivationExtractionArchitecture)
    that goes from first_hook to second_hook.
    If one spl
    """
    if ckpt is not None:
        state_dict = torch.load(str(ckpt))
        try:
            arch.load_state_dict(state_dict)
        except RuntimeError:
            try:
                cut_state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
                arch.load_state_dict(cut_state_dict)
            except RuntimeError as e:
                raise e

    if first_hook is not None and second_hook is not None:
        res_arch = arch.get_intermediate_module(arch, first_hook.keys, second_hook.keys)
    elif first_hook is None and second_hook is not None:
        res_arch = arch.get_intermediate_module(arch, None, second_hook.keys)
    elif first_hook is not None and second_hook is None:
        res_arch = arch.get_intermediate_module(arch, first_hook.keys, None)
    else:
        res_arch = arch
    return res_arch


def serialize_architecture_info(arch_infos: ArchitectureInfo) -> dict:
    arch_info_hook_info = tuple([asdict(h) for h in arch_infos.hooks])
    archinfo_dict = asdict(arch_infos)
    archinfo_dict["hooks"] = arch_info_hook_info
    archinfo_dict["checkpoint"] = str(archinfo_dict["checkpoint"])
    return archinfo_dict


def deserialize_architecture_info(archinfo_dict: dict) -> ArchitectureInfo:
    deserialized_hooks = []
    for h in archinfo_dict["hooks"]:
        deserialized_hooks.append(Hook(**h))
    archinfo_dict["hooks"] = tuple(deserialized_hooks)
    return ArchitectureInfo(**archinfo_dict)
