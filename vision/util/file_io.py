from __future__ import annotations

import collections
import json
import logging
import os
import pickle
import re
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from warnings import warn

import numpy as np
from repsim.benchmark.paths import VISION_MODEL_PATH
from vision.util import data_structs as ds
from vision.util import name_conventions as nc


logger = logging.getLogger(__name__)


def all_paths_exists(*args: Path):
    for a in args:
        if not a.exists():
            return False
    return True


def get_experiments_data_root_path():
    try:
        EXPERIMENTS_ROOT_PATH = os.environ["DATA_RESULTS_FOLDER"]
    except KeyError:
        raise KeyError("Could not find 'DATA_RESULTS_FOLDER'")
    return EXPERIMENTS_ROOT_PATH


def get_experiments_checkpoints_root_path() -> str:
    try:
        CHECKPOINT_ROOT_PATH = os.environ["CHECKPOINTS_FOLDER"]
    except KeyError:
        raise KeyError("Could not find 'INPUT_DATA_RESULTS_FOLDER'")
    return CHECKPOINT_ROOT_PATH


def save(
    obj: Any,
    path: str | Path,
    filename: str | None = None,
    overwrite: bool = True,
    make_dirs: bool = True,
) -> None:
    """Saves an data to disk. If a filename is given the path is considered to an directory.
    If the filename is not given the path has to have an extension that is supported and gets detected automatically.


    :param obj: Data to be saved
    :param path: Path to a file to save the data to or to a directory.
    :param filename: Optional. can be given should path lead to a directory
     and specifies the filename specifically.
    :param overwrite: Flag if overwriting should take place if a file
     should already exist.
    :param make_dirs: Flag if not existing directories should be created
    :return:
    :raises RuntimeError: If the file should exist and overwrite is not set to true.
    :raises ValueError: If the path leads to a file
    """
    p = str(path)
    _, ext = os.path.splitext(p)

    if (filename is None) and (ext == ""):
        raise ValueError("Expected either a filename in the path or a filename with extension. Can't have neither.")
    elif (filename is None) and (ext != ""):
        dirpath, filename = os.path.split(p)
        if not make_dirs:
            assert os.path.exists(dirpath), f"Given directory does not exist! Path: {dirpath}"
        else:
            os.makedirs(dirpath, exist_ok=True)
    else:
        dirpath = p
        if os.path.isfile(dirpath):
            raise ValueError(
                "Path to a file was given AND a filename is provided." " Filename should be None in this case though!"
            )
        os.makedirs(dirpath, exist_ok=True)

    full_path: str = os.path.join(dirpath, filename)
    extension = os.path.splitext(filename)[1]

    if not overwrite and os.path.isfile(full_path):
        raise FileExistsError(
            "Expected not existing file, found file with identical name." " Set overwrite to true to ignore this."
        )
    else:
        if extension == ".json":
            save_json(obj, full_path)
        elif extension == ".npz":
            save_npz(obj, full_path)
        elif extension == ".pkl":
            save_pickle(obj, full_path)
        elif extension == ".csv":
            save_csv(obj, full_path)
        elif extension == ".npy":
            save_np(obj, full_path)
        else:
            supported_extensions = "json npz pkl csv".split(" ")
            raise NotImplementedError(
                "The given extensions is supported. Supported are: {}".format(supported_extensions)
            )


def load(path: str | Path, filename: str | None = None, mmap=None) -> Any:
    """Basic loading method of the comp Manager.
    Retrieves and loads the file from the specified directory, depending on the
     file extension.
    """
    p = str(path)
    if filename is None:
        path, filename = os.path.split(p)
    else:
        if os.path.isfile(p):
            raise ValueError("Path to a file was given. Filename should be None in this case!")
        else:
            p = os.path.join(p, filename)

    extension = os.path.splitext(filename)[-1]
    if not os.path.exists(p):
        raise ValueError(f"Given path does not exists: {p}")
    else:
        if extension == ".npz":
            return load_npz(p)
        elif extension == ".json":
            return load_json(p)
        elif extension == ".pkl":
            return load_pickle(p)
        elif extension == ".csv":
            return load_csv(p)
        elif extension == ".npy":
            return load_np(p, mmap)
        else:
            supported_extensions = "json npz pkl csv".split(" ")
            raise NotImplementedError(
                f"Loading given extension is not supported!" f" Given: {extension}, Supported:{supported_extensions}"
            )


def strip_state_dict_of_keys(state_dict: dict) -> OrderedDict:
    """Removes the `net` value from the keys in the state_dict

    Example: original contains: "net.features.0.weight"
        current model expects: "features.0.weight"

    :return:
    """
    new_dict = collections.OrderedDict()
    for key, val in state_dict.items():
        new_dict[".".join(key.split(".")[1:])] = val

    return new_dict


def get_vision_model_info(
    dataset: str,
    architecture_name: str,
    seed_id: int,
    setting_identifier: str = "FIRST_MODELS",
) -> ds.ModelInfo:
    """Return the checkpoint of the group id if it already exists!"""
    root_path: Path = (
        Path(VISION_MODEL_PATH)
        / "vision_models_simbench"
        / nc.MODEL_DIR.format(setting_identifier, dataset, architecture_name)
        / nc.MODEL_SEED_ID_DIR.format(seed_id)
    )

    model = ds.ModelInfo(
        architecture=architecture_name,
        dataset=dataset,
        seed=seed_id,
        setting_identifier=setting_identifier,
        path_root=root_path,
    )

    return model


def first_model_trained(first_model: ds.ModelInfo) -> bool:
    """Return true if the info file and checkpoint exists."""
    return first_model.path_train_info_json.exists() and first_model.path_ckpt.exists()


def get_corresponding_first_model(model_info: ds.ModelInfo):
    """Return the corresponding first model info."""
    first_model_dir_name = nc.KE_MODEL_DIR.format(model_info.dataset, model_info.architecture)
    dir_name = nc.KE_MODEL_GROUP_ID_DIR.format(model_info.seed)
    first_data_path = model_info.path_data_root.parent.parent / first_model_dir_name / dir_name
    first_ckpt_path = model_info.path_ckpt_root.parent.parent / first_model_dir_name / dir_name
    return ds.ModelInfo(
        dir_name=nc.KE_MODEL_GROUP_ID_DIR.format(model_info.seed),
        architecture=model_info.architecture,
        dataset=model_info.dataset,
        learning_rate=model_info.learning_rate,
        split=model_info.split,
        weight_decay=model_info.weight_decay,
        batch_size=model_info.batch_size,
        path_data_root=first_data_path,
        path_ckpt_root=first_ckpt_path,
        seed=model_info.seed,
        model_id=0,
    )


def save_pickle(data: Any, filepath: str) -> None:
    """Save Python object to pickle.

    :param data: Data to be saved
    :param filepath: Path to save the object to
    :return: None
    """
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    return


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def load_pickle(filepath: str) -> Any:
    """Loads the pickled file from the filepath.

    :param filepath: Path to the file to load
    :return: loaded python object.
    """
    with open(filepath, "rb") as f:
        ret = pickle.load(f)
    return ret


def save_json(data: Any, filepath: Path | str) -> None:
    """

    :param data:
    :param filepath:
    :return:
    """
    with open(str(filepath), "w") as f:
        json.dump(data, f, indent=4)
    return


def load_json(filepath: str | Path) -> Any:
    """Load the json again

    :param filepath:
    :return:
    """
    with open(str(filepath)) as f:
        ret = json.load(f)
    return ret


def save_npz(data: dict, filepath: str | Path) -> None:
    # np.savez_compressed(filepath, **data)
    np.savez(str(filepath), **data)
    return


def save_np(data: np.ndarray, filepath: str | Path) -> None:
    # np.savez_compressed(filepath, **data)
    np.save(str(filepath), data)
    return


def load_np(filepath: str | Path, mmap: str = None):
    data = np.load(str(filepath), mmap_mode=mmap)
    return data


def load_npz(filepath: str, mmap: str = None) -> np.ndarray | Iterable | int | float | tuple | dict | np.memmap:
    data = np.load(filepath, mmap_mode=mmap)
    return data


def save_csv(data: np.ndarray, filepath: str) -> None:
    """Saves np.ndarray into csv file."""

    np.savetxt(filepath, data)  # noqa: type
    return


def load_csv(filepath: str) -> np.ndarray:
    """Loads the csv file into a np.ndarray

    :param filepath:
    :return:
    """
    return np.loadtxt(filepath)
