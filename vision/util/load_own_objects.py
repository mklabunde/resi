from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from vision.data.base_datamodule import BaseDataModule
from vision.util import data_structs as ds
from vision.util import name_conventions as nc
from vision.util.find_datamodules import get_datamodule


def load_json(filepath: str | Path) -> Any:
    """Load the json again

    :param filepath:
    :return:
    """
    with open(str(filepath)) as f:
        ret = json.load(f)
    return ret


def load_datamodule_from_info(model_info: ds.ModelInfo) -> BaseDataModule:
    """
    Returns an instance of the datamodule that was used in training of the trained model from the path.
    """
    oj = load_json(model_info.path_output_json)
    dataset = ds.Dataset(oj["dataset"])
    is_vit = True if model_info.architecture in ["ViT_B32", "ViT_L32"] else False
    return get_datamodule(dataset, is_vit)
