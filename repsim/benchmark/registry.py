from collections.abc import Sequence
from typing import get_args

import repsim.benchmark.paths
import repsim.nlp
import repsim.utils
from repsim.benchmark.types_globals import AUGMENTATION_100_SETTING
from repsim.benchmark.types_globals import AUGMENTATION_25_SETTING
from repsim.benchmark.types_globals import AUGMENTATION_50_SETTING
from repsim.benchmark.types_globals import AUGMENTATION_75_SETTING
from repsim.benchmark.types_globals import DEFAULT_SEEDS
from repsim.benchmark.types_globals import GRAPH_ARCHITECTURE_TYPE
from repsim.benchmark.types_globals import GRAPH_DATASET_TRAINED_ON
from repsim.benchmark.types_globals import GRAPH_DOMAIN
from repsim.benchmark.types_globals import RANDOM_LABEL_100_SETTING
from repsim.benchmark.types_globals import RANDOM_LABEL_25_SETTING
from repsim.benchmark.types_globals import RANDOM_LABEL_50_SETTING
from repsim.benchmark.types_globals import RANDOM_LABEL_75_SETTING
from repsim.benchmark.types_globals import SETTING_IDENTIFIER
from repsim.benchmark.types_globals import STANDARD_SETTING
from repsim.utils import GraphModel
from repsim.utils import MNLI
from repsim.utils import NLPModel
from repsim.utils import SST2
from repsim.utils import TrainedModel
from repsim.utils import VisionModel

NLP_TRAIN_DATASETS = {
    "sst2": SST2("sst2"),
    "sst2_sc_rate0558": SST2("sst2_sc_rate0558", shortcut_rate=0.558, shortcut_seed=0),
    "sst2_sc_rate0668": SST2("sst2_sc_rate0668", shortcut_rate=0.668, shortcut_seed=0),
    "sst2_sc_rate0779": SST2("sst2_sc_rate0779", shortcut_rate=0.779, shortcut_seed=0),
    "sst2_sc_rate0889": SST2("sst2_sc_rate0889", shortcut_rate=0.889, shortcut_seed=0),
    "sst2_sc_rate10": SST2("sst2_sc_rate10", shortcut_rate=1.0, shortcut_seed=0),
    "sst2_mem_rate025": SST2(
        "sst2_mem_rate025",
        path=str(repsim.benchmark.paths.NLP_DATA_PATH / "memorizing" / "sst2_labels5_strength025"),
        memorization_rate=0.25,
        memorization_n_new_labels=5,
        memorization_seed=0,
    ),
    "sst2_mem_rate05": SST2(
        "sst2_mem_rate05",
        path=str(repsim.benchmark.paths.NLP_DATA_PATH / "memorizing" / "sst2_labels5_strength05"),
        memorization_rate=0.5,
        memorization_n_new_labels=5,
        memorization_seed=0,
    ),
    "sst2_mem_rate075": SST2(
        "sst2_mem_rate075",
        path=str(repsim.benchmark.paths.NLP_DATA_PATH / "memorizing" / "sst2_labels5_strength075"),
        memorization_rate=0.75,
        memorization_n_new_labels=5,
        memorization_seed=0,
    ),
    "sst2_mem_rate10": SST2(
        "sst2_mem_rate10",
        path=str(repsim.benchmark.paths.NLP_DATA_PATH / "memorizing" / "sst2_labels5_strength10"),
        memorization_rate=1.0,
        memorization_n_new_labels=5,
        memorization_seed=0,
    ),
    "sst2_aug_rate025": SST2(
        "sst2_aug_rate025",
        path=str(repsim.benchmark.paths.NLP_DATA_PATH / "robustness" / "sst2_eda_strength025"),
        feature_column="augmented",
        augmentation_rate=0.25,
        augmentation_type="eda",
    ),
    "sst2_aug_rate05": SST2(
        "sst2_aug_rate05",
        path=str(repsim.benchmark.paths.NLP_DATA_PATH / "robustness" / "sst2_eda_strength05"),
        feature_column="augmented",
        augmentation_rate=0.5,
        augmentation_type="eda",
    ),
    "sst2_aug_rate075": SST2(
        "sst2_aug_rate075",
        path=str(repsim.benchmark.paths.NLP_DATA_PATH / "robustness" / "sst2_eda_strength075"),
        feature_column="augmented",
        augmentation_rate=0.75,
        augmentation_type="eda",
    ),
    "sst2_aug_rate10": SST2(
        "sst2_aug_rate10",
        path=str(repsim.benchmark.paths.NLP_DATA_PATH / "robustness" / "sst2_eda_strength10"),
        feature_column="augmented",
        augmentation_rate=1.0,
        augmentation_type="eda",
    ),
    "mnli_aug_rate025": MNLI(
        "mnli_aug_rate025",
        path=str(repsim.benchmark.paths.NLP_DATA_PATH / "robustness" / "mnli_eda_strength025"),
        feature_column="augmented",
        augmentation_rate=0.25,
        augmentation_type="eda",
    ),
    "mnli_aug_rate05": MNLI(
        "mnli_aug_rate05",
        path=str(repsim.benchmark.paths.NLP_DATA_PATH / "robustness" / "mnli_eda_strength05"),
        feature_column="augmented",
        augmentation_rate=0.5,
        augmentation_type="eda",
    ),
    "mnli_aug_rate075": MNLI(
        "mnli_aug_rate075",
        path=str(repsim.benchmark.paths.NLP_DATA_PATH / "robustness" / "mnli_eda_strength075"),
        feature_column="augmented",
        augmentation_rate=0.75,
        augmentation_type="eda",
    ),
    "mnli_aug_rate10": MNLI(
        "mnli_aug_rate10",
        path=str(repsim.benchmark.paths.NLP_DATA_PATH / "robustness" / "mnli_eda_strength10"),
        feature_column="augmented",
        augmentation_rate=1.0,
        augmentation_type="eda",
    ),
    "mnli_sc_rate0354": MNLI("mnli_sc_rate0354", shortcut_rate=0.354, shortcut_seed=0),
    "mnli_sc_rate05155": MNLI("mnli_sc_rate05155", shortcut_rate=0.5155, shortcut_seed=0),
    "mnli_sc_rate0677": MNLI("mnli_sc_rate0677", shortcut_rate=0.677, shortcut_seed=0),
    "mnli_sc_rate08385": MNLI("mnli_sc_rate08385", shortcut_rate=0.8385, shortcut_seed=0),
    "mnli_sc_rate1": MNLI("mnli_sc_rate1", shortcut_rate=1.0, shortcut_seed=0),
    "mnli_mem_rate025": MNLI(
        "mnli_mem_rate025",
        path=str(repsim.benchmark.paths.NLP_DATA_PATH / "memorizing" / "glue__mnli_labels5_strength025"),
        memorization_rate=0.25,
        memorization_n_new_labels=5,
        memorization_seed=0,
    ),
    "mnli_mem_rate05": MNLI(
        "mnli_mem_rate05",
        path=str(repsim.benchmark.paths.NLP_DATA_PATH / "memorizing" / "glue__mnli_labels5_strength05"),
        memorization_rate=0.5,
        memorization_n_new_labels=5,
        memorization_seed=0,
    ),
    "mnli_mem_rate075": MNLI(
        "mnli_mem_rate075",
        path=str(repsim.benchmark.paths.NLP_DATA_PATH / "memorizing" / "glue__mnli_labels5_strength075"),
        memorization_rate=0.75,
        memorization_n_new_labels=5,
        memorization_seed=0,
    ),
    "mnli_mem_rate10": MNLI(
        "mnli_mem_rate10",
        path=str(repsim.benchmark.paths.NLP_DATA_PATH / "memorizing" / "glue__mnli_labels5_strength10"),
        memorization_rate=1.0,
        memorization_n_new_labels=5,
        memorization_seed=0,
    ),
    "mnli": MNLI("mnli"),
}
NLP_REPRESENTATION_DATASETS = {
    "sst2": SST2("sst2", split="validation"),
    "sst2_sc_rate0": SST2(
        name="sst2_sc_rate0",
        # The local version would be useful if the modified tokenizer is saved with the trained models. But it's not,
        # so the shortcuts are added on the fly.
        # local_path=str(repsim.benchmark.paths.NLP_DATA_PATH / "shortcut" / "sst2_sc_rate0"),
        split="validation",
        shortcut_rate=0,
        shortcut_seed=0,
    ),
    "sst2_sc_rate0558": SST2(name="sst2_sc_rate0558", split="validation", shortcut_rate=0.558, shortcut_seed=0),
    "sst2_mem_rate0": SST2(name="sst2_mem_rate0", split="validation"),
    "sst2_aug_rate0": SST2(name="sst2_aug_rate0", split="validation"),
    "mnli": MNLI(name="mnli", split="validation_matched"),
    "mnli_aug_rate0": MNLI(name="mnli_aug_rate0", split="validation_matched"),
    "mnli_mem_rate0": MNLI(name="mnli_mem_rate0", split="validation_matched"),
    "mnli_sc_rate0354": MNLI(
        name="mnli_sc_rate0354", split="validation_matched", shortcut_rate=0.354, shortcut_seed=0
    ),
}


def all_trained_vision_models() -> list[VisionModel]:
    all_trained_vision_models = []
    for i in range(10):
        for arch in ["VGG11", "VGG19", "ResNet18", "ResNet34", "ResNet101", "ViT_B32", "ViT_L32"]:
            for dataset in ["CIFAR10", "CIFAR100", "ImageNet100"]:
                for identifier in [STANDARD_SETTING]:
                    all_trained_vision_models.append(
                        VisionModel(
                            domain="VISION",
                            architecture=arch,
                            train_dataset=dataset,
                            identifier=identifier,
                            seed=i,
                        )
                    )
    for i in range(5):
        for arch in ["ResNet18", "ResNet34", "VGG11"]:
            for dataset in [
                "ColorDot_100_CIFAR10DataModule",
                "ColorDot_75_CIFAR10DataModule",
                "ColorDot_50_CIFAR10DataModule",
                "ColorDot_25_CIFAR10DataModule",
                "ColorDot_0_CIFAR10DataModule",
            ]:
                for identifier in ["Shortcut"]:
                    all_trained_vision_models.append(
                        VisionModel(
                            domain="VISION",
                            architecture=arch,
                            train_dataset=dataset,
                            identifier=identifier,
                            seed=i,
                        )
                    )
    for i in range(5):
        for arch in ["ResNet18", "ResNet34", "ResNet101" "VGG11"]:
            for dataset in [
                "Gauss_Max_CIFAR10DataModule",
                "Gauss_L_CIFAR10DataModule",
                "Gauss_M_CIFAR10DataModule",
                "Gauss_S_CIFAR10DataModule",
                "Gauss_Off_CIFAR10DataModule",  # N
            ]:
                for identifier in ["GaussNoise"]:
                    all_trained_vision_models.append(
                        VisionModel(
                            domain="VISION",
                            architecture=arch,
                            train_dataset=dataset,
                            identifier=identifier,
                            seed=i,
                        )
                    )
    for i in range(5):
        for arch in ["VGG11", "VGG19", "ResNet18", "ResNet34", "ResNet101", "ViT_B32", "ViT_L32"]:
            for dataset in [
                "Gauss_Max_ImageNet100DataModule",
                "Gauss_L_ImageNet100DataModule",
                "Gauss_M_ImageNet100DataModule",
                "Gauss_S_ImageNet100DataModule",
                "Gauss_Off_ImageNet100DataModule",  # N
            ]:
                all_trained_vision_models.append(
                    VisionModel(
                        domain="VISION",
                        architecture=arch,
                        train_dataset=dataset,
                        identifier="GaussNoise",
                        seed=i,
                    )
                )
    for i in range(5):
        for arch in ["VGG11", "VGG19", "ResNet18", "ResNet34", "ResNet101", "ViT_B32", "ViT_L32"]:
            for dataset in [
                "ColorDot_100_ImageNet100DataModule",
                "ColorDot_75_ImageNet100DataModule",
                "ColorDot_50_ImageNet100DataModule",
                "ColorDot_25_ImageNet100DataModule",
                "ColorDot_0_ImageNet100DataModule",  # N
            ]:
                all_trained_vision_models.append(
                    VisionModel(
                        domain="VISION",
                        architecture=arch,
                        train_dataset=dataset,
                        identifier="Shortcut",
                        seed=i,
                    )
                )
    for i in range(5):
        for arch in ["VGG11", "VGG19", "ResNet18", "ResNet34", "ResNet101", "ViT_B32", "ViT_L32"]:
            for dataset in [
                "RandomLabel_100_IN100_DataModule",
                "RandomLabel_75_IN100_DataModule",
                "RandomLabel_50_IN100_DataModule",
                "RandomLabel_25_IN100_DataModule",
            ]:
                all_trained_vision_models.append(
                    VisionModel(
                        domain="VISION",
                        architecture=arch,
                        train_dataset=dataset,
                        identifier="Randomlabel",
                        seed=i,
                    )
                )

    return all_trained_vision_models


def all_trained_nlp_models() -> Sequence[NLPModel]:
    base_sst2_models = [
        NLPModel(
            train_dataset="sst2",
            identifier=STANDARD_SETTING,
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "standard" / f"sst2_pretrain{i}_finetune{i}"),
            tokenizer_name=f"google/multiberts-seed_{i}",
            token_pos=0,
        )
        for i in range(10)
    ]
    base_mnli_models = [
        NLPModel(
            train_dataset="mnli",  # type:ignore
            identifier=STANDARD_SETTING,
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "standard" / f"glue__mnli_pre{i}_ft{i}"),
            tokenizer_name=f"google/multiberts-seed_{i}",
            token_pos=0,
        )
        for i in range(10)
    ]

    shortcut_sst2_models = []
    for seed in range(10):
        for rate in ["0558", "0668", "0779", "0889", "10"]:
            shortcut_sst2_models.append(
                NLPModel(
                    identifier=f"Shortcut_{rate}",  # type:ignore
                    seed=seed,
                    train_dataset=f"sst2_sc_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH / "shortcut" / f"sst2_pre{seed}_ft{seed}_scrate{rate}"
                    ),
                    tokenizer_name=f"google/multiberts-seed_{seed}",
                    token_pos=0,  # only CLS token has been validated as different
                )
            )
    shortcut_mnli_models = []
    for seed in range(5):
        for rate in ["0354", "05155", "0677", "08385", "1"]:
            shortcut_mnli_models.append(
                NLPModel(
                    identifier=f"Shortcut_{rate}",  # type:ignore
                    seed=seed,
                    train_dataset=f"mnli_sc_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "shortcut"
                        / f"glue__mnli_pre{seed}_ft{seed}_scrate{rate}"
                    ),
                    tokenizer_name=f"google/multiberts-seed_{seed}",
                    token_pos=0,  # only CLS token has been validated as different
                )
            )

    memorizing_sst2_models = []
    rate_to_setting = {
        "025": RANDOM_LABEL_25_SETTING,
        "05": RANDOM_LABEL_50_SETTING,
        "075": RANDOM_LABEL_75_SETTING,
        "10": RANDOM_LABEL_100_SETTING,
    }
    for seed in range(5):
        for rate in ["025", "05", "075", "10"]:
            memorizing_sst2_models.append(
                NLPModel(
                    identifier=rate_to_setting[rate],  # type:ignore
                    seed=seed,
                    train_dataset=f"sst2_mem_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "memorizing"
                        / f"sst2_pre{seed}_ft{seed}_labels5_strength{rate}"
                    ),
                    tokenizer_name=f"google/multiberts-seed_{seed}",
                    token_pos=0,  # only CLS token has been validated as different
                )
            )
    memorizing_sst2_models += [
        NLPModel(
            train_dataset="sst2",
            identifier="RandomLabels_0",
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "standard" / f"sst2_pretrain{i}_finetune{i}"),
            tokenizer_name=f"google/multiberts-seed_{i}",
            token_pos=0,
        )
        for i in range(5)
    ]

    memorizing_mnli_models = []
    rate_to_setting = {
        "025": RANDOM_LABEL_25_SETTING,
        "05": RANDOM_LABEL_50_SETTING,
        "075": RANDOM_LABEL_75_SETTING,
        "10": RANDOM_LABEL_100_SETTING,
    }
    for seed in range(5):
        for rate in ["025", "05", "075", "10"]:
            memorizing_mnli_models.append(
                NLPModel(
                    identifier=rate_to_setting[rate],  # type:ignore
                    seed=seed,
                    train_dataset=f"mnli_mem_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "memorizing"
                        / f"glue__mnli_pre{seed}_ft{seed}_labels5_strength{rate}"
                    ),
                    tokenizer_name=f"google/multiberts-seed_{seed}",
                    token_pos=0,  # only CLS token has been validated as different
                )
            )
    memorizing_mnli_models += [
        NLPModel(
            train_dataset="mnli",
            identifier="RandomLabels_0",
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "standard" / f"glue__mnli_pre{i}_ft{i}"),
            tokenizer_name=f"google/multiberts-seed_{i}",
            token_pos=0,
        )
        for i in range(5)
    ]

    rate_to_setting = {
        "025": AUGMENTATION_25_SETTING,
        "05": AUGMENTATION_50_SETTING,
        "075": AUGMENTATION_75_SETTING,
        "10": AUGMENTATION_100_SETTING,
    }
    augmented_sst2_models = []
    for seed in range(10):
        for rate in ["025", "05", "075", "10"]:
            augmented_sst2_models.append(
                NLPModel(
                    identifier=rate_to_setting[rate],  # type:ignore
                    seed=seed,
                    train_dataset=f"sst2_aug_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "augmentation"
                        / f"sst2_pre{seed}_ft{seed}_eda_strength{rate}"
                    ),
                    tokenizer_name=f"google/multiberts-seed_{seed}",
                    token_pos=0,  # only CLS token has been validated as different
                )
            )
    augmented_sst2_models += [
        NLPModel(
            train_dataset="sst2",
            identifier="Augmentation_0",
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "standard" / f"sst2_pretrain{i}_finetune{i}"),
            tokenizer_name=f"google/multiberts-seed_{i}",
            token_pos=0,
        )
        for i in range(10)
    ]

    augmented_mnli_models = []
    for seed in range(5):  # TODO: train more models
        for rate in ["025", "05", "075", "10"]:
            augmented_mnli_models.append(
                NLPModel(
                    identifier=rate_to_setting[rate],  # type:ignore
                    seed=seed,
                    train_dataset=f"mnli_aug_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "augmentation"
                        / f"glue__mnli_pre{seed}_ft{seed}_eda_strength{rate}"
                    ),
                    tokenizer_name=f"google/multiberts-seed_{seed}",
                    token_pos=0,  # only CLS token has been validated as different
                )
            )
    augmented_mnli_models += [
        NLPModel(
            train_dataset="mnli",
            identifier="Augmentation_0",
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "standard" / f"glue__mnli_pre{i}_ft{i}"),
            tokenizer_name=f"google/multiberts-seed_{i}",
            token_pos=0,
        )
        for i in range(5)
    ]

    return (
        base_sst2_models
        + base_mnli_models
        + shortcut_sst2_models
        + shortcut_mnli_models
        + memorizing_sst2_models
        + memorizing_mnli_models
        + augmented_sst2_models
        + augmented_mnli_models
    )


def all_trained_graph_models() -> list[TrainedModel]:
    all_trained_models = []

    for i in DEFAULT_SEEDS:
        for arch in get_args(GRAPH_ARCHITECTURE_TYPE):
            for dataset in get_args(GRAPH_DATASET_TRAINED_ON):
                for setting in list(get_args(SETTING_IDENTIFIER)):
                    all_trained_models.append(
                        GraphModel(
                            domain=GRAPH_DOMAIN,
                            architecture=arch,
                            train_dataset=dataset,
                            identifier=setting,
                            seed=i,
                            additional_kwargs={},
                        )
                    )
    for i in [5, 6, 7, 8, 9]:
        for arch in get_args(GRAPH_ARCHITECTURE_TYPE):
            for dataset in get_args(GRAPH_DATASET_TRAINED_ON):
                all_trained_models.append(
                    GraphModel(
                        domain=GRAPH_DOMAIN,
                        architecture=arch,
                        train_dataset=dataset,
                        identifier=STANDARD_SETTING,
                        seed=i,
                        additional_kwargs={},
                    )
                )
    return all_trained_models


ALL_TRAINED_MODELS: list[TrainedModel | NLPModel] = []
ALL_TRAINED_MODELS.extend(all_trained_vision_models())
ALL_TRAINED_MODELS.extend(all_trained_nlp_models())
ALL_TRAINED_MODELS.extend(all_trained_graph_models())
