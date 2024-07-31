import argparse
import os
from typing import get_args
from typing import List

import yaml
from repsim.benchmark.types_globals import AUGMENTATION_EXPERIMENT_NAME
from repsim.benchmark.types_globals import BENCHMARK_EXPERIMENTS_LIST
from repsim.benchmark.types_globals import DEFAULT_SEEDS
from repsim.benchmark.types_globals import EXPERIMENT_IDENTIFIER
from repsim.benchmark.types_globals import EXPERIMENT_SEED
from repsim.benchmark.types_globals import GRAPH_DATASET_TRAINED_ON
from repsim.benchmark.types_globals import GRAPH_EXPERIMENT_DEFAULT_DICT
from repsim.benchmark.types_globals import GRAPH_EXPERIMENT_FIVE_GROUPS_DICT
from repsim.benchmark.types_globals import GRAPH_EXPERIMENT_TWO_GROUPS_DICT
from repsim.benchmark.types_globals import GROUP_SEPARATION_EXPERIMENT
from repsim.benchmark.types_globals import LABEL_EXPERIMENT_NAME
from repsim.benchmark.types_globals import LAYER_EXPERIMENT_NAME
from repsim.benchmark.types_globals import MONOTONICITY_EXPERIMENT
from repsim.benchmark.types_globals import OUTPUT_CORRELATION_EXPERIMENT
from repsim.benchmark.types_globals import OUTPUT_CORRELATION_EXPERIMENT_NAME
from repsim.benchmark.types_globals import SHORTCUT_EXPERIMENT_NAME
from repsim.measures import ALL_MEASURES
from repsim.run import run

CONFIG_INCLUDED_MEASURES_KEY = "included_measures"
CONFIG_EXCLUDED_MEASURES_KEY = "excluded_measures"
CONFIG_THREADS_KEY = "threads"
CONFIG_CACHE_DISK_KEY = "cache_to_disk"
CONFIG_CACHE_MEMORY_KEY = "cache_to_mem"
CONFIG_RERUN_NANS_KEY = "rerun_nans"
CONFIG_EXTRACT_REPS_ONLY_KEY = "only_extract_reps"
CONFIG_REPRESENTATION_DATASET_KEY = "representation_dataset"

CONFIG_EXPERIMENTS_KEY = "experiments"
CONFIG_EXPERIMENTS_NAME_SUBKEY = "name"
CONFIG_EXPERIMENTS_TYPE_SUBKEY = "type"
CONFIG_EXPERIMENTS_USE_ACC_SUBKEY = "use_acc_comparison"
CONFIG_EXPERIMENTS_FILTER_SUBKEY = "filter_key_vals"
CONFIG_EXPERIMENTS_TRAIN_DATA_SUBKEY = "train_dataset"
CONFIG_EXPERIMENTS_SEEDS_SUBKEY = "seed"
CONFIG_EXPERIMENTS_IDENTIFIER_SUBKEY = "identifier"
CONFIG_EXPERIMENTS_GROUPING_SUBKEY = "grouping_keys"
CONFIG_EXPERIMENTS_SEPARATION_SUBKEY = "separation_keys"
CONFIG_EXPERIMENTS_REPRESENTATION_DATA_SUBKEY = "representation_dataset"
CONFIG_EXPERIMENTS_DOMAIN_SUBKEY = "domain"

CONFIG_RAW_RESULTS_FILENAME_KEY = "raw_results_filename"

CONFIG_RES_TABLE_CREATION_KEY = "table_creation"
CONFIG_RES_TABLE_SAVE_SUBKEY = "save_full_df"
CONFIG_RES_TABLE_FILENAME_SUBKEY = "full_df_filename"
CONFIG_AGG_TABLE_SAVE_SUBKEY = "save_aggregated_df"
CONFIG_AGG_TABLE_INDEX_SUBKEY = "row_index"
CONFIG_AGG_TABLE_COLUMNS_SUBKEY = "columns"
CONFIG_AGG_TABLE_VALUE_SUBKEY = "value_key"
CONFIG_AGG_TABLE_FILENAME_SUBKEY = "filename"

EXPERIMENT_TYPE_DICT = {
    OUTPUT_CORRELATION_EXPERIMENT_NAME: OUTPUT_CORRELATION_EXPERIMENT,
    LAYER_EXPERIMENT_NAME: MONOTONICITY_EXPERIMENT,
    SHORTCUT_EXPERIMENT_NAME: GROUP_SEPARATION_EXPERIMENT,
    AUGMENTATION_EXPERIMENT_NAME: GROUP_SEPARATION_EXPERIMENT,
    LABEL_EXPERIMENT_NAME: GROUP_SEPARATION_EXPERIMENT,
}


def BASE_FILE_NAME(test, dataset, groups: int = 3, measures: List = None):
    fname_base = f"graphs_{test}_{dataset}"
    if groups != 3 and test != OUTPUT_CORRELATION_EXPERIMENT:
        fname_base += f"_{groups}groups"
    if measures is not None:
        for m in measures:
            fname_base += f"_{m}"
    return fname_base


def PARQUET_FILE_NAME(test, dataset, measures: List = None):
    return f"{BASE_FILE_NAME(test, dataset, measures = measures)}.parquet"


def FULL_DF_FILE_NAME(test, dataset, groups: int = 3, measures: List = None):
    return f"{BASE_FILE_NAME(test, dataset, groups, measures)}_full.csv"


def AGG_DF_FILE_NAME(test, dataset, groups: int = 3, measures: List = None):
    return f"{BASE_FILE_NAME(test, dataset, groups, measures)}.csv"


def YAML_CONFIG_FILE_NAME(test, dataset, groups: int = 3, measures: List = None):
    return f"{BASE_FILE_NAME(test, dataset, groups, measures)}.yaml"


def build_graph_config(
    test: EXPERIMENT_IDENTIFIER,
    dataset: GRAPH_DATASET_TRAINED_ON,
    measures: List = None,
    save_to_memory=True,
    save_to_disk=False,
    groups: int = 3,
):
    if groups == 5:
        experiment_settings = GRAPH_EXPERIMENT_FIVE_GROUPS_DICT[test]
    elif groups == 3:
        experiment_settings = GRAPH_EXPERIMENT_DEFAULT_DICT[test]
    else:
        experiment_settings = GRAPH_EXPERIMENT_TWO_GROUPS_DICT[test]

    experiment_type = EXPERIMENT_TYPE_DICT[test]
    seeds = list(get_args(EXPERIMENT_SEED)) if test == OUTPUT_CORRELATION_EXPERIMENT_NAME else DEFAULT_SEEDS

    save_agg_table = True if experiment_type != OUTPUT_CORRELATION_EXPERIMENT else False
    yaml_dict = {
        CONFIG_THREADS_KEY: 1,
        CONFIG_CACHE_MEMORY_KEY: save_to_memory,
        CONFIG_CACHE_DISK_KEY: save_to_disk,
        CONFIG_EXTRACT_REPS_ONLY_KEY: False,
        CONFIG_EXPERIMENTS_KEY: [
            {
                CONFIG_EXPERIMENTS_NAME_SUBKEY: f"{test} {dataset}",
                CONFIG_EXPERIMENTS_TYPE_SUBKEY: experiment_type,
                CONFIG_REPRESENTATION_DATASET_KEY: dataset,
                CONFIG_EXPERIMENTS_FILTER_SUBKEY: {
                    CONFIG_EXPERIMENTS_IDENTIFIER_SUBKEY: experiment_settings,
                    CONFIG_EXPERIMENTS_TRAIN_DATA_SUBKEY: [dataset],
                    CONFIG_EXPERIMENTS_SEEDS_SUBKEY: seeds,
                    CONFIG_EXPERIMENTS_DOMAIN_SUBKEY: "GRAPHS",
                },
                CONFIG_EXPERIMENTS_GROUPING_SUBKEY: ["identifier"],
                CONFIG_EXPERIMENTS_SEPARATION_SUBKEY: ["architecture"],
            },
        ],
        CONFIG_RAW_RESULTS_FILENAME_KEY: PARQUET_FILE_NAME(test, dataset, measures=measures),
        #
        CONFIG_RES_TABLE_CREATION_KEY: {
            CONFIG_RES_TABLE_SAVE_SUBKEY: True,
            CONFIG_RES_TABLE_FILENAME_SUBKEY: FULL_DF_FILE_NAME(test, dataset, groups, measures),
            CONFIG_AGG_TABLE_SAVE_SUBKEY: save_agg_table,
            CONFIG_AGG_TABLE_INDEX_SUBKEY: "similarity_measure",
            CONFIG_AGG_TABLE_COLUMNS_SUBKEY: ["quality_measure", "architecture"],
            CONFIG_AGG_TABLE_VALUE_SUBKEY: "value",
            CONFIG_AGG_TABLE_FILENAME_SUBKEY: AGG_DF_FILE_NAME(test, dataset, groups, measures),
        },
    }
    if measures is None:
        yaml_dict[CONFIG_EXCLUDED_MEASURES_KEY] = ["GeometryScore"]
    else:
        yaml_dict[CONFIG_INCLUDED_MEASURES_KEY] = measures

    if test == OUTPUT_CORRELATION_EXPERIMENT_NAME:
        yaml_dict[CONFIG_EXPERIMENTS_KEY][0][CONFIG_EXPERIMENTS_USE_ACC_SUBKEY] = True

    return yaml_dict


def parse_args():
    """Parses arguments given to script

    Returns:
        dict-like object -- Dict-like object containing all given arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=list(get_args(GRAPH_DATASET_TRAINED_ON)),
        help="Datasets used in evaluation.",
    )
    parser.add_argument(
        "-t",
        "--test",
        type=str,
        choices=BENCHMARK_EXPERIMENTS_LIST,
        help="Test to run.",
    )
    parser.add_argument(
        "-m",
        "--measures",
        type=str,
        nargs="*",
        choices=list(ALL_MEASURES.keys()),
        default=None,
        help="Measures to test - if none are specified, all benchmark measures will be used.",
    )
    parser.add_argument(
        "--groups",
        type=int,
        choices=[2, 3, 5],
        default=3,
        help="Number of groups to separate per test.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    n_groups = args.groups

    yaml_config = build_graph_config(test=args.test, dataset=args.dataset, measures=args.measures, groups=n_groups)

    config_path = os.path.join(
        "configs", "graphs", YAML_CONFIG_FILE_NAME(test=args.test, dataset=args.dataset, groups=n_groups)
    )
    with open(config_path, "w") as file:
        yaml.dump(yaml_config, file)

    run(config_path=config_path)
