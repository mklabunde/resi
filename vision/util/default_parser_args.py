from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from dataclasses import replace

import git
from vision.util import data_structs as ds


def get_git_hash_of_repo() -> str:
    if "data" in os.environ:
        raise NotImplementedError()
        # repo_path = "/home/t006d/Code/"
        # repo = git.Repo(repo_path, search_parent_directories=True)
    else:
        repo = git.Repo(search_parent_directories=True)

    sha = repo.head.object.hexsha
    return sha


def add_vision_training_params(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-d",
        "--dataset",
        default=ds.Dataset.CIFAR10.value,
        nargs="?",
        choices=[c.value for c in list(ds.Dataset)],
        type=str,
        help="Dataset name to be trained on.",
    )
    parser.add_argument(
        "-a",
        "--architecture",
        choices=[c.value for c in list(ds.BaseArchitecture)],
        default=ds.BaseArchitecture.RESNET50.value,
        type=str,
        nargs="?",
        help="Name of the architecture to train.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        required=True,
        help="To differentiate between different groups with same config (for MC runs)",
    )
    parser.add_argument(
        "-sid",
        "--setting_identifier",
        type=str,
        required=True,
        default="Normal",
        help="Split of the Dataset to train on",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        type=str2bool,
        default=False,
        help="Whether to overwrite the model if it already exists.",
    )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
