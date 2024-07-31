import os
from argparse import ArgumentParser
from collections.abc import Sequence
from typing import Literal

import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt
from repsim.benchmark.model_selection import get_grouped_models
from repsim.benchmark.paths import EXPERIMENT_RESULTS_PATH
from repsim.benchmark.paths import LOG_PATH
from repsim.benchmark.registry import ALL_TRAINED_MODELS
from repsim.utils import VisionModel
from vision.train_vision_model import IN_AUGMENTATION_DATAMODULES
from vision.train_vision_model import IN_RANDOMLABEL_DATAMODULES
from vision.train_vision_model import IN_SHORTCUT_DATAMODULES
from vision.train_vision_model import STANDARD_DATAMODULES
from vision.util.data_structs import load_json
from vision.util.data_structs import ModelInfo


settings = Literal["Shortcut", "GaussNoise", "RandomLabel"]


def get_grouping_dataset(group_setting: settings) -> tuple[list, dict[str, str]]:
    if group_setting == "GaussNoise":
        datasets = [
            "Gauss_Max_ImageNet100DataModule",
            "Gauss_M_ImageNet100DataModule",
            "Gauss_S_ImageNet100DataModule",
            # "Gauss_Off_ImageNet100DataModule",
        ]
        mapping = {
            "Gauss_Max_ImageNet100DataModule": "Noise: Max",
            # "Gauss_L_ImageNet100DataModule": "Noise: Large",
            "Gauss_M_ImageNet100DataModule": "Noise: Medium",
            "Gauss_S_ImageNet100DataModule": "Noise: Small",
            # "Gauss_Off_ImageNet100DataModule": "Noise: Off",
        }

    elif group_setting == "RandomLabel":
        datasets = [
            "RandomLabel_100_IN100_DataModule",
            "RandomLabel_50_IN100_DataModule",
            "ImageNet100",
        ]  # Add ImageNet100
        mapping = {
            "RandomLabel_100_IN100_DataModule": "Random Labels: 100%",
            # "RandomLabel_75_IN100_DataModule": "Random Labels: 75%",
            "RandomLabel_50_IN100_DataModule": "Random Labels: 50%",
            # "RandomLabel_25_IN100_DataModule": "Random Labels: 25%",
            "ImageNet100": "Random Labels: 0%",
        }
    elif group_setting == "Shortcut":
        datasets = [
            "ColorDot_100_ImageNet100DataModule",
            "ColorDot_75_ImageNet100DataModule",
            "ColorDot_0_ImageNet100DataModule",
        ]
        mapping = {
            "ColorDot_100_ImageNet100DataModule": "Shortcut: 100%",
            "ColorDot_75_ImageNet100DataModule": "Shortcut: 75%",
            # "ColorDot_50_ImageNet100DataModule": "Shortcut: 50%",
            # "ColorDot_25_ImageNet100DataModule": "Shortcut: 50%",
            "ColorDot_0_ImageNet100DataModule": "Shortcut: 0%",
        }
    else:
        raise NotImplementedError(f"Group setting {group_setting} not implemented.")
    return datasets, mapping


def run(group_setting: settings):

    out_file = EXPERIMENT_RESULTS_PATH / f"model_accuracies_{group_setting}.csv"

    setting_seeds = [0, 1, 2, 3, 4]
    dataset, mapping = get_grouping_dataset(group_setting)
    grouped_models: list[list[Sequence[VisionModel]]] = get_grouped_models(
        models=ALL_TRAINED_MODELS,
        filter_key_vals={"train_dataset": dataset, "seed": setting_seeds, "domain": "VISION"},
        separation_keys={"architecture"},
    )

    model: VisionModel
    vision_info: ModelInfo
    model_group: Sequence[VisionModel]
    all_results: list[dict] = []
    for arch_group in grouped_models:
        model_group = arch_group[0]
        architecture = model_group[0].architecture
        for model in model_group:
            vision_info = model.get_vision_model_info()
            seed = model.seed
            train_dataset = model.train_dataset
            if not vision_info.path_output_json.exists():
                logger.info(f"Missing Seed {seed} of {architecture} trained on  {train_dataset}")
                continue
            vision_info_json = load_json(vision_info.path_output_json)
            if group_setting == "Shortcut":
                accuracy = vision_info_json["test_no_sc"]["no_shortcut"]["accuracy"]
                # The json can be hard to understand, so here is a breakdown for later:
                # the output contains "test_no_sc" and "test_full_sc" which are the interesting keys
                #   Each one contains "no_shortcut" and "shortcut", which refers again contains metrics.
                #       This is because the trainer evaluates two things:
                #           1. How is the accuracy of the model to the image label --> "no_shortcut"  (Should probably have been named "image_label_acc")
                #           2. How is the accuracy of the model to the shortcut label --> "shortcut"  (Should probably have been named "shortcut_label_acc")
                #       But in the test setting we use different dataloaders, the uncorrelated one is the "test_no_sc"
                #           and the one with full correlation is the "test_full_sc".
                #           One can also see that the in the "test_full_sc" case the "no_shortcut" and "shortcut" accuracy are equal
                #               as the shortcut label is always the same as the image label
            elif group_setting == "GaussNoise":
                accuracy = vision_info_json["test"]["accuracy"]
            elif group_setting == "RandomLabel":
                accuracy = vision_info_json["test"]["accuracy"]
            all_results.append(
                {
                    "Architecture": architecture,
                    "Train_Dataset": train_dataset,
                    "Train Setting": mapping[train_dataset],
                    "Seed": seed,
                    "Accuracy": accuracy,
                }
            )
    return all_results


def plot_df(group_setting: str, setting_result: list[dict]):

    mapping = get_grouping_dataset(group_setting)[1]
    df = pd.DataFrame(setting_result)
    out_file = EXPERIMENT_RESULTS_PATH / f"model_accuracies_{group_setting}.csv"
    df.to_csv(out_file, index=False)

    plt.figure(figsize=(10, 6))
    plt.grid(True, axis="y")
    sns.set_theme("paper", style="white", font_scale=1.5)
    sns.stripplot(data=df, x="Architecture", y="Accuracy", hue="Train Setting", hue_order=mapping.values())
    plt.title(f"{group_setting} Model Accuracies")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig(EXPERIMENT_RESULTS_PATH / f"model_accuracies_{group_setting}.pdf", bbox_inches="tight")
    plt.savefig(EXPERIMENT_RESULTS_PATH / f"model_accuracies_{group_setting}.png", bbox_inches="tight", dpi=300)
    # plt.savefig(EXPERIMENT_RESULTS_PATH / f"model_accuracies_{group_setting}.pdf")


# def plot_joint(group_setting: str, all_results: list[tuple[str, list[dict]]):


if __name__ == "__main__":
    all_results = []
    for setting in ["GaussNoise", "Shortcut", "RandomLabel"]:
        setting_result = run(setting)
        plot_df(setting, setting_result)
        all_results.append((setting, setting_result))
