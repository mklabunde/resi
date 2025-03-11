from collections.abc import Sequence
from typing import get_args

import repsim.benchmark.paths
import repsim.nlp
import repsim.utils
from repsim.benchmark.types_globals import AUGMENTATION_100_SETTING
from repsim.benchmark.types_globals import AUGMENTATION_25_SETTING
from repsim.benchmark.types_globals import AUGMENTATION_50_SETTING
from repsim.benchmark.types_globals import AUGMENTATION_75_SETTING
from repsim.benchmark.types_globals import CORA_DATASET_NAME
from repsim.benchmark.types_globals import DEFAULT_SEEDS
from repsim.benchmark.types_globals import GAT_MODEL_NAME
from repsim.benchmark.types_globals import GCN_MODEL_NAME
from repsim.benchmark.types_globals import GRAPH_ARCHITECTURE_TYPE
from repsim.benchmark.types_globals import GRAPH_DATASET_TRAINED_ON
from repsim.benchmark.types_globals import GRAPH_DOMAIN
from repsim.benchmark.types_globals import GRAPH_EXPERIMENT_FIVE_GROUPS_DICT
from repsim.benchmark.types_globals import GRAPHSAGE_MODEL_NAME
from repsim.benchmark.types_globals import LABEL_EXPERIMENT_NAME
from repsim.benchmark.types_globals import MULTI_LAYER_SETTING
from repsim.benchmark.types_globals import PGNN_MODEL_NAME
from repsim.benchmark.types_globals import RANDOM_LABEL_100_SETTING
from repsim.benchmark.types_globals import RANDOM_LABEL_25_SETTING
from repsim.benchmark.types_globals import RANDOM_LABEL_50_SETTING
from repsim.benchmark.types_globals import RANDOM_LABEL_75_SETTING
from repsim.benchmark.types_globals import SETTING_IDENTIFIER
from repsim.benchmark.types_globals import SHORTCUT_EXPERIMENT_NAME
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
    "sst2_sft": SST2(
        name="sst2_sft",
        local_path=str(repsim.benchmark.paths.NLP_DATA_PATH / "llm_sft" / "standard" / "sst2"),
        split="train",
        feature_column="sft",
    ),
    "sst2_sft_sc_rate0558": SST2(
        name="sst2_sft_sc_rate0558",
        local_path=str(repsim.benchmark.paths.NLP_DATA_PATH / "llm_sft" / "shortcut" / "sst2_sc_rate0558"),
        split="train",
        feature_column="sft",
        shortcut_rate=0.558,
        shortcut_seed=0,
    ),
    "sst2_sft_sc_rate0889": SST2(
        name="sst2_sft_sc_rate0889",
        local_path=str(repsim.benchmark.paths.NLP_DATA_PATH / "llm_sft" / "shortcut" / "sst2_sc_rate0889"),
        split="train",
        feature_column="sft",
        shortcut_rate=0.889,
        shortcut_seed=0,
    ),
    "sst2_sft_sc_rate10": SST2(
        name="sst2_sft_sc_rate10",
        local_path=str(repsim.benchmark.paths.NLP_DATA_PATH / "llm_sft" / "shortcut" / "sst2_sc_rate10"),
        split="train",
        feature_column="sft",
        shortcut_rate=1.0,
        shortcut_seed=0,
    ),
    "sst2_sft_mem_rate10": SST2(
        name="sst2_sft_mem_rate10",
        local_path=str(repsim.benchmark.paths.NLP_DATA_PATH / "llm_sft" / "memorization" / "sst2_rate10"),
        split="train",
        feature_column="sft",
        memorization_rate=1.0,
        memorization_seed=0,
    ),
    "sst2_sft_mem_rate075": SST2(
        name="sst2_sft_mem_rate075",
        local_path=str(repsim.benchmark.paths.NLP_DATA_PATH / "llm_sft" / "memorization" / "sst2_rate075"),
        split="train",
        feature_column="sft",
        memorization_rate=0.75,
        memorization_seed=0,
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
    "mnli_sc_rate10": MNLI("mnli_sc_rate1", shortcut_rate=1.0, shortcut_seed=0),  # alias
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
    "mnli_sft": MNLI(
        name="mnli_sft",
        local_path=str(repsim.benchmark.paths.NLP_DATA_PATH / "llm_sft" / "standard" / "mnli"),
        split="train",
        feature_column="sft",
    ),
    "mnli_sft_sc_rate0354": MNLI(
        name="mnli_sft_sc_rate0354",
        local_path=str(repsim.benchmark.paths.NLP_DATA_PATH / "llm_sft" / "shortcut" / "mnli_sc_rate0354"),
        split="train",
        feature_column="sft",
        shortcut_rate=0.354,
        shortcut_seed=0,
    ),
    "mnli_sft_sc_rate08385": MNLI(
        name="mnli_sft_sc_rate08385",
        local_path=str(repsim.benchmark.paths.NLP_DATA_PATH / "llm_sft" / "shortcut" / "mnli_sc_rate08385"),
        split="train",
        feature_column="sft",
        shortcut_rate=0.8385,
        shortcut_seed=0,
    ),
    "mnli_sft_sc_rate10": MNLI(
        name="mnli_sft_sc_rate10",
        local_path=str(repsim.benchmark.paths.NLP_DATA_PATH / "llm_sft" / "shortcut" / "mnli_sc_rate10"),
        split="train",
        feature_column="sft",
        shortcut_rate=1.0,
        shortcut_seed=0,
    ),
    "mnli_sft_mem_rate10": MNLI(
        name="mnli_sft_mem_rate10",
        local_path=str(repsim.benchmark.paths.NLP_DATA_PATH / "llm_sft" / "memorization" / "mnli_rate10"),
        split="train",
        feature_column="sft",
        memorization_rate=1.0,
        memorization_seed=0,
    ),
    "mnli_sft_mem_rate075": MNLI(
        name="mnli_sft_mem_rate075",
        local_path=str(repsim.benchmark.paths.NLP_DATA_PATH / "llm_sft" / "memorization" / "mnli_rate075"),
        split="train",
        feature_column="sft",
        memorization_rate=0.75,
        memorization_seed=0,
    ),
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
    "sst2_sft": SST2(
        name="sst2_sft",
        local_path=str(repsim.benchmark.paths.NLP_DATA_PATH / "llm_sft" / "standard" / "sst2"),
        split="validation",
        feature_column="sft",
    ),
    "sst2_sft_sc_rate0558": SST2(
        name="sst2_sft_sc_rate0558",
        local_path=str(repsim.benchmark.paths.NLP_DATA_PATH / "llm_sft" / "shortcut" / "sst2_sc_rate0558"),
        split="validation",
        feature_column="sft",
        shortcut_rate=0.558,
        shortcut_seed=0,
    ),
    "sst2_sft_mem_rate0": SST2(
        name="sst2_sft_mem_rate0",
        local_path=str(repsim.benchmark.paths.NLP_DATA_PATH / "llm_sft" / "standard" / "sst2"),
        split="validation",
        feature_column="sft",
    ),
    "mnli": MNLI(name="mnli", split="validation_matched"),
    "mnli_aug_rate0": MNLI(name="mnli_aug_rate0", split="validation_matched"),
    "mnli_mem_rate0": MNLI(name="mnli_mem_rate0", split="validation_matched"),
    "mnli_sc_rate0354": MNLI(
        name="mnli_sc_rate0354", split="validation_matched", shortcut_rate=0.354, shortcut_seed=0
    ),
    "mnli_sft": MNLI(
        name="mnli_sft",
        local_path=str(repsim.benchmark.paths.NLP_DATA_PATH / "llm_sft" / "standard" / "mnli"),
        split="validation_matched",
        feature_column="sft",
    ),
    "mnli_sft_sc_rate0354": MNLI(
        name="mnli_sft_sc_rate0354",
        local_path=str(repsim.benchmark.paths.NLP_DATA_PATH / "llm_sft" / "shortcut" / "mnli_sc_rate0354"),
        split="validation_matched",
        feature_column="sft",
        shortcut_rate=0.354,
        shortcut_seed=0,
    ),
    "mnli_sft_mem_rate0": SST2(
        name="mnli_sft_mem_rate0",
        local_path=str(repsim.benchmark.paths.NLP_DATA_PATH / "llm_sft" / "standard" / "mnli"),
        split="validation_matched",
        feature_column="sft",
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
        for arch in ["ResNet18", "ResNet34", "ResNet101", "VGG11", "VGG19", "ViT_B32", "ViT_L32"]:
            for dataset in [
                "ColorDot_100_C100DataModule",
                "ColorDot_75_C100DataModule",
                "ColorDot_50_C100DataModule",
                "ColorDot_25_C100DataModule",
                "ColorDot_0_C100DataModule",
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
        for arch in ["ResNet18", "ResNet34", "ResNet101", "VGG11"]:
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
        for arch in ["ResNet18", "ResNet34", "ResNet101", "VGG11", "VGG19", "ViT_B32", "ViT_L32"]:
            for dataset in [
                "Gauss_Max_CIFAR100DataModule",
                "Gauss_L_CIFAR100DataModule",
                "Gauss_M_CIFAR100DataModule",
                "Gauss_S_CIFAR100DataModule",
                "Gauss_Off_CIFAR100DataModule",  # N
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
    for i in range(5):
        for arch in ["VGG11", "VGG19", "ResNet18", "ResNet34", "ResNet101", "ViT_B32", "ViT_L32"]:
            for dataset in [
                "RandomLabel_100_C100_DataModule",
                "RandomLabel_75_C100_DataModule",
                "RandomLabel_50_C100_DataModule",
                "RandomLabel_25_C100_DataModule",
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
    base_sst2_models = (
        [
            NLPModel(
                train_dataset="sst2",
                identifier=STANDARD_SETTING,
                seed=i,
                path=str(
                    repsim.benchmark.paths.NLP_MODEL_PATH / "bert" / "standard" / f"sst2_pretrain{i}_finetune{i}"
                ),
                tokenizer_name=f"google/multiberts-seed_{i}",
                token_pos=0,
            )
            for i in range(10)
        ]
        + [
            NLPModel(
                architecture="albert-base-v2",
                train_dataset="sst2",
                identifier=STANDARD_SETTING,
                seed=ft_seed,
                path=str(
                    repsim.benchmark.paths.NLP_MODEL_PATH
                    / "albert"
                    / "standard"
                    / f"sst2_pre{pretrain_seed}_ft{ft_seed}"
                ),
                tokenizer_name="albert/albert-base-v2",
                token_pos=0,
            )
            for pretrain_seed, ft_seed in zip([0] * 10, range(123, 133))
        ]
        + [
            NLPModel(
                architecture="smollm2-1.7b",
                model_type="causal-lm",
                train_dataset="sst2_sft",  # type:ignore
                identifier=STANDARD_SETTING,
                seed=seed,
                path=str(
                    repsim.benchmark.paths.NLP_SMOLLM_PATH / f"ft_smollm2_1-7b_sst2_seed{seed}_bs16_ff/checkpoint-500"
                ),
                tokenizer_name="HuggingFaceTB/SmolLM2-1.7B",
                token_pos=-1,
            )
            for seed in range(10)
        ]
    )
    base_mnli_models = (
        [
            NLPModel(
                train_dataset="mnli",  # type:ignore
                identifier=STANDARD_SETTING,
                seed=i,
                path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "bert" / "standard" / f"glue__mnli_pre{i}_ft{i}"),
                tokenizer_name=f"google/multiberts-seed_{i}",
                token_pos=0,
            )
            for i in range(10)
        ]
        + [
            NLPModel(
                architecture="albert-base-v2",
                train_dataset="mnli",  # type:ignore
                identifier=STANDARD_SETTING,
                seed=i,
                path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "albert" / "standard" / f"glue__mnli_pre0_ft{i}"),
                tokenizer_name="albert/albert-base-v2",
                token_pos=0,
            )
            for i in range(10)
        ]
        + [
            NLPModel(
                architecture="smollm2-1.7b",
                model_type="causal-lm",
                train_dataset="mnli_sft",  # type:ignore
                identifier=STANDARD_SETTING,
                seed=seed,
                path=str(
                    repsim.benchmark.paths.NLP_SMOLLM_PATH / f"ft_smollm2_1-7b_mnli_seed{seed}_bs16_ff/checkpoint-500"
                ),
                tokenizer_name="HuggingFaceTB/SmolLM2-1.7B",
                token_pos=-1,
            )
            for seed in range(10)
        ]
    )

    shortcut_sst2_models = []
    for seed in range(10):
        for rate in ["0558", "0668", "0779", "0889", "10"]:
            shortcut_sst2_models.append(
                NLPModel(
                    identifier=f"Shortcut_{rate}",  # type:ignore
                    seed=seed,
                    train_dataset=f"sst2_sc_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "bert"
                        / "shortcut"
                        / f"sst2_pre{seed}_ft{seed}_scrate{rate}"
                    ),
                    tokenizer_name=f"google/multiberts-seed_{seed}",
                    token_pos=0,  # only CLS token has been validated as different
                )
            )
    for seed in range(5):
        for rate in ["0558", "0889", "10"]:
            shortcut_sst2_models.append(
                NLPModel(
                    architecture="albert-base-v2",
                    identifier=f"Shortcut_{rate}",  # type:ignore
                    seed=seed,
                    train_dataset=f"sst2_sc_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "albert"
                        / "shortcut"
                        / f"sst2_pre0_ft{seed}_scrate{rate}"
                    ),
                    tokenizer_name="albert/albert-base-v2",
                    token_pos=0,  # only CLS token has been validated as different
                )
            )
    for seed in range(10):
        for rate in ["0558", "10"]:
            rateId = rate if rate == "0558" else ""
            shortcut_sst2_models.append(
                NLPModel(
                    architecture="smollm2-1.7b",
                    model_type="causal-lm",
                    identifier=f"Shortcut_{rate}",  # type:ignore
                    seed=seed,
                    train_dataset=f"sst2_sft_sc_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_SMOLLM_PATH
                        / f"ft_smollm2_1-7b_sst2-shortcut{rateId}_seed{seed}_bs16_ff/checkpoint-500"
                    ),
                    tokenizer_name="HuggingFaceTB/SmolLM2-1.7B",
                    token_pos=-1,  # only CLS token has been validated as different
                )
            )
    for seed in range(5):
        for rate in ["0889"]:
            rateId = rate if rate == "0889" else ""
            shortcut_sst2_models.append(
                NLPModel(
                    architecture="smollm2-1.7b",
                    model_type="causal-lm",
                    identifier=f"Shortcut_{rate}",  # type:ignore
                    seed=seed,
                    train_dataset=f"sst2_sft_sc_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_SMOLLM_PATH
                        / f"ft_smollm2_1-7b_sst2-shortcut{rateId}_seed{seed}_bs16_ff/checkpoint-500"
                    ),
                    tokenizer_name="HuggingFaceTB/SmolLM2-1.7B",
                    token_pos=-1,  # only CLS token has been validated as different
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
                        / "bert"
                        / "shortcut"
                        / f"glue__mnli_pre{seed}_ft{seed}_scrate{rate}"
                    ),
                    tokenizer_name=f"google/multiberts-seed_{seed}",
                    token_pos=0,  # only CLS token has been validated as different
                )
            )
    for seed in range(5):
        for rate in ["0354", "08385", "10"]:
            shortcut_mnli_models.append(
                NLPModel(
                    architecture="albert-base-v2",
                    identifier=f"Shortcut_{rate}",  # type:ignore
                    seed=seed,
                    train_dataset=f"mnli_sc_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "albert"
                        / "shortcut"
                        / f"glue__mnli_pre0_ft{seed}_scrate{rate}"
                    ),
                    tokenizer_name="albert/albert-base-v2",
                    token_pos=0,  # only CLS token has been validated as different
                )
            )
    for seed in range(5):
        for rate in ["0354", "08385", "10"]:
            shortcut_mnli_models.append(
                NLPModel(
                    architecture="smollm2-1.7b",
                    model_type="causal-lm",
                    identifier=f"Shortcut_{rate}",  # type:ignore
                    seed=seed,
                    train_dataset=f"mnli_sft_sc_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_SMOLLM_PATH
                        / f"ft_smollm2_1-7b_mnli-shortcut{rate}_seed{seed}_bs16_ff/checkpoint-500"
                    ),
                    tokenizer_name="HuggingFaceTB/SmolLM2-1.7B",
                    token_pos=-1,  # only CLS token has been validated as different
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
                        / "bert"
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
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "bert" / "standard" / f"sst2_pretrain{i}_finetune{i}"),
            tokenizer_name=f"google/multiberts-seed_{i}",
            token_pos=0,
        )
        for i in range(5)
    ]
    for seed in range(5):
        for rate in ["075", "10"]:
            if rate == "075" and seed == 0:
                continue  # seed 0 has mem100-like behavior. We add seed 6 below.
            memorizing_sst2_models.append(
                NLPModel(
                    architecture="albert-base-v2",
                    identifier=rate_to_setting[rate],  # type:ignore
                    seed=seed,
                    train_dataset=f"sst2_mem_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "albert"
                        / "memorizing"
                        / f"sst2_pre0_ft{seed}_labels5_strength{rate}"
                    ),
                    tokenizer_name="albert/albert-base-v2",
                    token_pos=0,  # only CLS token has been validated as different
                )
            )
    memorizing_sst2_models.append(
        NLPModel(
            architecture="albert-base-v2",
            train_dataset="sst2_mem_rate075",  # type:ignore
            identifier="RandomLabels_75",
            seed=6,
            path=str(
                repsim.benchmark.paths.NLP_MODEL_PATH / "albert" / "memorizing" / f"sst2_pre0_ft6_labels5_strength075"
            ),
            tokenizer_name="albert/albert-base-v2",
            token_pos=0,
        )
    )
    memorizing_sst2_models += [
        NLPModel(
            architecture="albert-base-v2",
            train_dataset="sst2",
            identifier="RandomLabels_0",
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "albert" / "standard" / f"sst2_pre0_ft{i}"),
            tokenizer_name="albert/albert-base-v2",
            token_pos=0,
        )
        for i in range(123, 128)
    ]
    memorizing_sst2_models += (
        [
            NLPModel(
                architecture="smollm2-1.7b",
                model_type="causal-lm",
                train_dataset="sst2_sft",  # type:ignore
                identifier="RandomLabels_0",
                seed=seed,
                path=str(
                    repsim.benchmark.paths.NLP_SMOLLM_PATH / f"ft_smollm2_1-7b_sst2_seed{seed}_bs16_ff/checkpoint-500"
                ),
                tokenizer_name="HuggingFaceTB/SmolLM2-1.7B",
                token_pos=-1,
            )
            for seed in range(10)
        ]
        + [
            NLPModel(
                architecture="smollm2-1.7b",
                model_type="causal-lm",
                train_dataset="sst2_sft_mem_rate075",  # type:ignore
                identifier=RANDOM_LABEL_75_SETTING,
                seed=seed,
                path=str(
                    repsim.benchmark.paths.NLP_SMOLLM_PATH
                    / f"ft_smollm2_1-7b_sst2-mem075_seed{seed}_bs16_ff/checkpoint-500"
                ),
                tokenizer_name="HuggingFaceTB/SmolLM2-1.7B",
                token_pos=-1,
            )
            for seed in range(5)
        ]
        + [
            NLPModel(
                architecture="smollm2-1.7b",
                model_type="causal-lm",
                train_dataset="sst2_sft_mem_rate10",  # type:ignore
                identifier=RANDOM_LABEL_100_SETTING,
                seed=seed,
                path=str(
                    repsim.benchmark.paths.NLP_SMOLLM_PATH
                    / f"ft_smollm2_1-7b_sst2-mem10_seed{seed}_bs16_ff/checkpoint-500"
                ),
                tokenizer_name="HuggingFaceTB/SmolLM2-1.7B",
                token_pos=-1,
            )
            for seed in range(5)
        ]
    )

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
                        / "bert"
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
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "bert" / "standard" / f"glue__mnli_pre{i}_ft{i}"),
            tokenizer_name=f"google/multiberts-seed_{i}",
            token_pos=0,
        )
        for i in range(5)
    ]
    for seed in range(5):
        for rate in ["075", "10"]:
            memorizing_mnli_models.append(
                NLPModel(
                    architecture="albert-base-v2",
                    identifier=rate_to_setting[rate],  # type:ignore
                    seed=seed,
                    train_dataset=f"mnli_mem_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "albert"
                        / "memorizing"
                        / f"glue__mnli_pre0_ft{seed}_labels5_strength{rate}"
                    ),
                    tokenizer_name="albert/albert-base-v2",
                    token_pos=0,  # only CLS token has been validated as different
                )
            )
    memorizing_mnli_models += [
        NLPModel(
            architecture="albert-base-v2",
            train_dataset="mnli",
            identifier="RandomLabels_0",
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "albert" / "standard" / f"glue__mnli_pre0_ft{i}"),
            tokenizer_name="albert/albert-base-v2",
            token_pos=0,
        )
        for i in range(5)
    ]
    for seed in range(5):
        for rate in ["075", "10"]:
            memorizing_mnli_models.append(
                NLPModel(
                    architecture="smollm2-1.7b",
                    model_type="causal-lm",
                    train_dataset=f"mnli_sft_mem_rate{rate}",  # type:ignore
                    identifier=rate_to_setting[rate],  # type:ignore
                    seed=seed,
                    path=str(
                        repsim.benchmark.paths.NLP_SMOLLM_PATH
                        / f"ft_smollm2_1-7b_mnli-mem{rate}_seed{seed}_bs16_ff/checkpoint-500"
                    ),
                    tokenizer_name="HuggingFaceTB/SmolLM2-1.7B",
                    token_pos=-1,
                )
            )
    memorizing_mnli_models += [
        NLPModel(
            architecture="smollm2-1.7b",
            model_type="causal-lm",
            train_dataset="mnli_sft",  # type:ignore
            identifier="RandomLabels_0",
            seed=seed,
            path=str(
                repsim.benchmark.paths.NLP_SMOLLM_PATH / f"ft_smollm2_1-7b_mnli_seed{seed}_bs16_ff/checkpoint-500"
            ),
            tokenizer_name="HuggingFaceTB/SmolLM2-1.7B",
            token_pos=-1,
        )
        for seed in range(5)
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
                        / "bert"
                        / "augmentation"
                        / f"sst2_pre{seed}_ft{seed}_eda_strength{rate}"
                    ),
                    tokenizer_name=f"google/multiberts-seed_{seed}",
                    token_pos=0,  # only CLS token has been validated as different
                )
            )
    for seed in range(5):
        for rate in ["10"]:
            augmented_sst2_models.append(
                NLPModel(
                    architecture="albert-base-v2",
                    identifier=rate_to_setting[rate],  # type:ignore
                    seed=seed,
                    train_dataset=f"sst2_aug_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "albert"
                        / "augmentation"
                        / f"sst2_pre0_ft{seed}_eda_strength{rate}"
                    ),
                    tokenizer_name="albert/albert-base-v2",
                    token_pos=0,  # only CLS token has been validated as different
                )
            )
    augmented_sst2_models += [
        NLPModel(
            train_dataset="sst2",
            identifier="Augmentation_0",
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "bert" / "standard" / f"sst2_pretrain{i}_finetune{i}"),
            tokenizer_name=f"google/multiberts-seed_{i}",
            token_pos=0,
        )
        for i in range(10)
    ] + [
        NLPModel(
            architecture="albert-base-v2",
            train_dataset="sst2",
            identifier="Augmentation_0",
            seed=ft_seed,
            path=str(
                repsim.benchmark.paths.NLP_MODEL_PATH / "albert" / "standard" / f"sst2_pre{pretrain_seed}_ft{ft_seed}"
            ),
            tokenizer_name="albert/albert-base-v2",
            token_pos=0,
        )
        for pretrain_seed, ft_seed in zip([0] * 10, range(123, 133))
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
                        / "bert"
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
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "bert" / "standard" / f"glue__mnli_pre{i}_ft{i}"),
            tokenizer_name=f"google/multiberts-seed_{i}",
            token_pos=0,
        )
        for i in range(5)
    ]
    for seed in range(5):  # TODO: train more models
        for rate in ["10"]:
            augmented_mnli_models.append(
                NLPModel(
                    architecture="albert-base-v2",
                    identifier=rate_to_setting[rate],  # type:ignore
                    seed=seed,
                    train_dataset=f"mnli_aug_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "albert"
                        / "augmentation"
                        / f"glue__mnli_pre0_ft{seed}_eda_strength{rate}"
                    ),
                    tokenizer_name="albert/albert-base-v2",
                    token_pos=0,  # only CLS token has been validated as different
                )
            )
    augmented_mnli_models += [
        NLPModel(
            architecture="albert-base-v2",
            train_dataset="mnli",
            identifier="Augmentation_0",
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "albert" / "standard" / f"glue__mnli_pre0_ft{i}"),
            tokenizer_name="albert/albert-base-v2",
            token_pos=0,
        )
        for i in range(5)
    ]

    # mean pooled representations
    base_sst2_models_meanpooled = [
        NLPModel(
            train_dataset="sst2",
            identifier=STANDARD_SETTING,
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "bert" / "standard" / f"sst2_pretrain{i}_finetune{i}"),
            tokenizer_name=f"google/multiberts-seed_{i}",
            token_pos="mean",
            additional_kwargs={
                "token_pos": "mean"
            },  # important because added to id. Otherwise not distinguishable from CLS-token model
        )
        for i in range(10)
    ] + [
        NLPModel(
            architecture="albert-base-v2",
            train_dataset="sst2",
            identifier=STANDARD_SETTING,
            seed=ft_seed,
            path=str(
                repsim.benchmark.paths.NLP_MODEL_PATH / "albert" / "standard" / f"sst2_pre{pretrain_seed}_ft{ft_seed}"
            ),
            tokenizer_name="albert/albert-base-v2",
            token_pos="mean",
            additional_kwargs={"token_pos": "mean"},
        )
        for pretrain_seed, ft_seed in zip([0] * 10, range(123, 133))
    ]

    shortcut_sst2_models_meanpooled = []
    for seed in range(10):
        for rate in ["0558", "0668", "0779", "0889", "10"]:
            shortcut_sst2_models_meanpooled.append(
                NLPModel(
                    identifier=f"Shortcut_{rate}",  # type:ignore
                    seed=seed,
                    train_dataset=f"sst2_sc_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "bert"
                        / "shortcut"
                        / f"sst2_pre{seed}_ft{seed}_scrate{rate}"
                    ),
                    tokenizer_name=f"google/multiberts-seed_{seed}",
                    token_pos="mean",
                    additional_kwargs={"token_pos": "mean"},
                )
            )
    for seed in range(5):
        for rate in ["0558", "0889", "10"]:
            shortcut_sst2_models_meanpooled.append(
                NLPModel(
                    architecture="albert-base-v2",
                    identifier=f"Shortcut_{rate}",  # type:ignore
                    seed=seed,
                    train_dataset=f"sst2_sc_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "albert"
                        / "shortcut"
                        / f"sst2_pre0_ft{seed}_scrate{rate}"
                    ),
                    tokenizer_name="albert/albert-base-v2",
                    token_pos="mean",
                    additional_kwargs={"token_pos": "mean"},
                )
            )

    memorizing_sst2_models_meanpooled = []
    rate_to_setting = {
        "025": RANDOM_LABEL_25_SETTING,
        "05": RANDOM_LABEL_50_SETTING,
        "075": RANDOM_LABEL_75_SETTING,
        "10": RANDOM_LABEL_100_SETTING,
    }
    for seed in range(5):
        for rate in ["025", "05", "075", "10"]:
            memorizing_sst2_models_meanpooled.append(
                NLPModel(
                    identifier=rate_to_setting[rate],  # type:ignore
                    seed=seed,
                    train_dataset=f"sst2_mem_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "bert"
                        / "memorizing"
                        / f"sst2_pre{seed}_ft{seed}_labels5_strength{rate}"
                    ),
                    tokenizer_name=f"google/multiberts-seed_{seed}",
                    token_pos="mean",
                    additional_kwargs={"token_pos": "mean"},
                )
            )
    memorizing_sst2_models_meanpooled += [
        NLPModel(
            train_dataset="sst2",
            identifier="RandomLabels_0",
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "bert" / "standard" / f"sst2_pretrain{i}_finetune{i}"),
            tokenizer_name=f"google/multiberts-seed_{i}",
            token_pos="mean",
            additional_kwargs={"token_pos": "mean"},
        )
        for i in range(5)
    ]
    for seed in range(5):
        for rate in ["075", "10"]:
            if rate == "075" and seed == 0:
                continue  # seed 0 has mem100-like behavior. We add seed 6 below.
            memorizing_sst2_models_meanpooled.append(
                NLPModel(
                    architecture="albert-base-v2",
                    identifier=rate_to_setting[rate],  # type:ignore
                    seed=seed,
                    train_dataset=f"sst2_mem_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "albert"
                        / "memorizing"
                        / f"sst2_pre0_ft{seed}_labels5_strength{rate}"
                    ),
                    tokenizer_name="albert/albert-base-v2",
                    token_pos="mean",
                    additional_kwargs={"token_pos": "mean"},
                )
            )
    memorizing_sst2_models_meanpooled.append(
        NLPModel(
            architecture="albert-base-v2",
            train_dataset="sst2_mem_rate075",  # type:ignore
            identifier="RandomLabels_75",
            seed=6,
            path=str(
                repsim.benchmark.paths.NLP_MODEL_PATH / "albert" / "memorizing" / f"sst2_pre0_ft6_labels5_strength075"
            ),
            tokenizer_name="albert/albert-base-v2",
            token_pos="mean",
            additional_kwargs={"token_pos": "mean"},
        )
    )
    memorizing_sst2_models_meanpooled += [
        NLPModel(
            architecture="albert-base-v2",
            train_dataset="sst2",
            identifier="RandomLabels_0",
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "albert" / "standard" / f"sst2_pre0_ft{i}"),
            tokenizer_name="albert/albert-base-v2",
            token_pos="mean",
            additional_kwargs={"token_pos": "mean"},
        )
        for i in range(123, 128)
    ]

    rate_to_setting = {
        "025": AUGMENTATION_25_SETTING,
        "05": AUGMENTATION_50_SETTING,
        "075": AUGMENTATION_75_SETTING,
        "10": AUGMENTATION_100_SETTING,
    }
    augmented_sst2_models_meanpooled = []
    for seed in range(10):
        for rate in ["025", "05", "075", "10"]:
            augmented_sst2_models_meanpooled.append(
                NLPModel(
                    identifier=rate_to_setting[rate],  # type:ignore
                    seed=seed,
                    train_dataset=f"sst2_aug_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "bert"
                        / "augmentation"
                        / f"sst2_pre{seed}_ft{seed}_eda_strength{rate}"
                    ),
                    tokenizer_name=f"google/multiberts-seed_{seed}",
                    token_pos="mean",
                    additional_kwargs={"token_pos": "mean"},
                )
            )
    for seed in range(5):
        for rate in ["10"]:
            augmented_sst2_models_meanpooled.append(
                NLPModel(
                    architecture="albert-base-v2",
                    identifier=rate_to_setting[rate],  # type:ignore
                    seed=seed,
                    train_dataset=f"sst2_aug_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "albert"
                        / "augmentation"
                        / f"sst2_pre0_ft{seed}_eda_strength{rate}"
                    ),
                    tokenizer_name="albert/albert-base-v2",
                    token_pos="mean",
                    additional_kwargs={"token_pos": "mean"},
                )
            )
    augmented_sst2_models_meanpooled += [
        NLPModel(
            train_dataset="sst2",
            identifier="Augmentation_0",
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "bert" / "standard" / f"sst2_pretrain{i}_finetune{i}"),
            tokenizer_name=f"google/multiberts-seed_{i}",
            token_pos="mean",
            additional_kwargs={"token_pos": "mean"},
        )
        for i in range(10)
    ] + [
        NLPModel(
            architecture="albert-base-v2",
            train_dataset="sst2",
            identifier="Augmentation_0",
            seed=ft_seed,
            path=str(
                repsim.benchmark.paths.NLP_MODEL_PATH / "albert" / "standard" / f"sst2_pre{pretrain_seed}_ft{ft_seed}"
            ),
            tokenizer_name="albert/albert-base-v2",
            token_pos="mean",
            additional_kwargs={"token_pos": "mean"},
        )
        for pretrain_seed, ft_seed in zip([0] * 10, range(123, 133))
    ]

    base_mnli_models_meanpooled = [
        NLPModel(
            train_dataset="mnli",  # type:ignore
            identifier=STANDARD_SETTING,
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "bert" / "standard" / f"glue__mnli_pre{i}_ft{i}"),
            tokenizer_name=f"google/multiberts-seed_{i}",
            token_pos="mean",
            additional_kwargs={"token_pos": "mean"},
        )
        for i in range(10)
    ] + [
        NLPModel(
            architecture="albert-base-v2",
            train_dataset="mnli",  # type:ignore
            identifier=STANDARD_SETTING,
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "albert" / "standard" / f"glue__mnli_pre0_ft{i}"),
            tokenizer_name="albert/albert-base-v2",
            token_pos="mean",
            additional_kwargs={"token_pos": "mean"},
        )
        for i in range(10)
    ]
    shortcut_mnli_models_meanpooled = []
    for seed in range(5):
        for rate in ["0354", "05155", "0677", "08385", "1"]:
            shortcut_mnli_models_meanpooled.append(
                NLPModel(
                    identifier=f"Shortcut_{rate}",  # type:ignore
                    seed=seed,
                    train_dataset=f"mnli_sc_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "bert"
                        / "shortcut"
                        / f"glue__mnli_pre{seed}_ft{seed}_scrate{rate}"
                    ),
                    tokenizer_name=f"google/multiberts-seed_{seed}",
                    token_pos="mean",
                    additional_kwargs={"token_pos": "mean"},
                )
            )
    for seed in range(5):
        for rate in ["0354", "08385", "10"]:
            shortcut_mnli_models_meanpooled.append(
                NLPModel(
                    architecture="albert-base-v2",
                    identifier=f"Shortcut_{rate}",  # type:ignore
                    seed=seed,
                    train_dataset=f"mnli_sc_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "albert"
                        / "shortcut"
                        / f"glue__mnli_pre0_ft{seed}_scrate{rate}"
                    ),
                    tokenizer_name="albert/albert-base-v2",
                    token_pos="mean",
                    additional_kwargs={"token_pos": "mean"},
                )
            )
    memorizing_mnli_models_meanpooled = []
    rate_to_setting = {
        "025": RANDOM_LABEL_25_SETTING,
        "05": RANDOM_LABEL_50_SETTING,
        "075": RANDOM_LABEL_75_SETTING,
        "10": RANDOM_LABEL_100_SETTING,
    }
    for seed in range(5):
        for rate in ["025", "05", "075", "10"]:
            memorizing_mnli_models_meanpooled.append(
                NLPModel(
                    identifier=rate_to_setting[rate],  # type:ignore
                    seed=seed,
                    train_dataset=f"mnli_mem_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "bert"
                        / "memorizing"
                        / f"glue__mnli_pre{seed}_ft{seed}_labels5_strength{rate}"
                    ),
                    tokenizer_name=f"google/multiberts-seed_{seed}",
                    token_pos="mean",
                    additional_kwargs={"token_pos": "mean"},
                )
            )
    memorizing_mnli_models_meanpooled += [
        NLPModel(
            train_dataset="mnli",
            identifier="RandomLabels_0",
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "bert" / "standard" / f"glue__mnli_pre{i}_ft{i}"),
            tokenizer_name=f"google/multiberts-seed_{i}",
            token_pos="mean",
            additional_kwargs={"token_pos": "mean"},
        )
        for i in range(5)
    ]
    for seed in range(5):
        for rate in ["075", "10"]:
            memorizing_mnli_models_meanpooled.append(
                NLPModel(
                    architecture="albert-base-v2",
                    identifier=rate_to_setting[rate],  # type:ignore
                    seed=seed,
                    train_dataset=f"mnli_mem_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "albert"
                        / "memorizing"
                        / f"glue__mnli_pre0_ft{seed}_labels5_strength{rate}"
                    ),
                    tokenizer_name="albert/albert-base-v2",
                    token_pos="mean",
                    additional_kwargs={"token_pos": "mean"},
                )
            )
    memorizing_mnli_models_meanpooled += [
        NLPModel(
            architecture="albert-base-v2",
            train_dataset="mnli",
            identifier="RandomLabels_0",
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "albert" / "standard" / f"glue__mnli_pre0_ft{i}"),
            tokenizer_name="albert/albert-base-v2",
            token_pos="mean",
            additional_kwargs={"token_pos": "mean"},
        )
        for i in range(5)
    ]

    rate_to_setting = {
        "025": AUGMENTATION_25_SETTING,
        "05": AUGMENTATION_50_SETTING,
        "075": AUGMENTATION_75_SETTING,
        "10": AUGMENTATION_100_SETTING,
    }
    augmented_mnli_models_meanpooled = []
    for seed in range(5):  # TODO: train more models
        for rate in ["025", "05", "075", "10"]:
            augmented_mnli_models_meanpooled.append(
                NLPModel(
                    identifier=rate_to_setting[rate],  # type:ignore
                    seed=seed,
                    train_dataset=f"mnli_aug_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "bert"
                        / "augmentation"
                        / f"glue__mnli_pre{seed}_ft{seed}_eda_strength{rate}"
                    ),
                    tokenizer_name=f"google/multiberts-seed_{seed}",
                    token_pos="mean",
                    additional_kwargs={"token_pos": "mean"},
                )
            )
    augmented_mnli_models_meanpooled += [
        NLPModel(
            train_dataset="mnli",
            identifier="Augmentation_0",
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "bert" / "standard" / f"glue__mnli_pre{i}_ft{i}"),
            tokenizer_name=f"google/multiberts-seed_{i}",
            token_pos="mean",
            additional_kwargs={"token_pos": "mean"},
        )
        for i in range(5)
    ]
    for seed in range(5):  # TODO: train more models
        for rate in ["10"]:
            augmented_mnli_models_meanpooled.append(
                NLPModel(
                    architecture="albert-base-v2",
                    identifier=rate_to_setting[rate],  # type:ignore
                    seed=seed,
                    train_dataset=f"mnli_aug_rate{rate}",  # type:ignore
                    path=str(
                        repsim.benchmark.paths.NLP_MODEL_PATH
                        / "albert"
                        / "augmentation"
                        / f"glue__mnli_pre0_ft{seed}_eda_strength{rate}"
                    ),
                    tokenizer_name="albert/albert-base-v2",
                    token_pos="mean",
                    additional_kwargs={"token_pos": "mean"},
                )
            )
    augmented_mnli_models_meanpooled += [
        NLPModel(
            architecture="albert-base-v2",
            train_dataset="mnli",
            identifier="Augmentation_0",
            seed=i,
            path=str(repsim.benchmark.paths.NLP_MODEL_PATH / "albert" / "standard" / f"glue__mnli_pre0_ft{i}"),
            tokenizer_name="albert/albert-base-v2",
            token_pos="mean",
            additional_kwargs={"token_pos": "mean"},
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
        + base_sst2_models_meanpooled
        + shortcut_sst2_models_meanpooled
        + memorizing_sst2_models_meanpooled
        + augmented_sst2_models_meanpooled
        + base_mnli_models_meanpooled
        + shortcut_mnli_models_meanpooled
        + memorizing_mnli_models_meanpooled
        + augmented_mnli_models_meanpooled
    )


def all_trained_graph_models() -> list[TrainedModel]:
    all_trained_models = []

    for i in DEFAULT_SEEDS:
        for arch in [GCN_MODEL_NAME, GRAPHSAGE_MODEL_NAME, GAT_MODEL_NAME]:
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
        for arch in [GCN_MODEL_NAME, GRAPHSAGE_MODEL_NAME, GAT_MODEL_NAME]:
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

    # ADDITIONAL P-GNN EXPERIMENTS
    for i in DEFAULT_SEEDS:
        for setting in (
            GRAPH_EXPERIMENT_FIVE_GROUPS_DICT[LABEL_EXPERIMENT_NAME]
            + GRAPH_EXPERIMENT_FIVE_GROUPS_DICT[SHORTCUT_EXPERIMENT_NAME]
            + [MULTI_LAYER_SETTING]
        ):
            all_trained_models.append(
                GraphModel(
                    domain=GRAPH_DOMAIN,
                    architecture=PGNN_MODEL_NAME,
                    train_dataset=CORA_DATASET_NAME,
                    identifier=setting,
                    seed=i,
                    additional_kwargs={},
                )
            )
    for i in [5, 6, 7, 8, 9]:
        all_trained_models.append(
            GraphModel(
                domain=GRAPH_DOMAIN,
                architecture=PGNN_MODEL_NAME,
                train_dataset=CORA_DATASET_NAME,
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
