import os
from argparse import ArgumentParser
from collections.abc import Sequence

import yaml
from loguru import logger
from repsim.benchmark.group_separation_experiment import GroupSeparationExperiment
from repsim.benchmark.model_selection import _filter_models
from repsim.benchmark.model_selection import _separate_models_by_keys
from repsim.benchmark.model_selection import get_grouped_models
from repsim.benchmark.monotonicity_experiment import MonotonicityExperiment
from repsim.benchmark.output_correlation_experiment import OutputCorrelationExperiment
from repsim.benchmark.paths import EXPERIMENT_RESULTS_PATH
from repsim.benchmark.paths import LOG_PATH
from repsim.benchmark.registry import ALL_TRAINED_MODELS
from repsim.benchmark.registry import TrainedModel
from repsim.benchmark.utils import create_pivot_excel_table
from repsim.benchmark.utils import save_full_table
from repsim.measures import ALL_MEASURES
from repsim.measures import FUNCTIONAL_SIMILARITY_MEASURES
from repsim.measures.utils import RepresentationalSimilarityMeasure


def read_yaml_config(config_path: str) -> dict:
    """
    Read a yaml file and return the dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # ------------ If no measures provided we default to all measures ------------ #
    return config


def get_measures(config: dict) -> list[RepresentationalSimilarityMeasure]:
    include_key = "included_measures"
    exclude_key = "excluded_measures"
    assert not (
        include_key in config and exclude_key in config
    ), "Cannot include and exclude measures at the same time"
    used_measures = None
    if include_key in config:
        if config[include_key] == "all":
            used_measures = list(ALL_MEASURES.values())
        else:
            if isinstance(config[include_key], str):
                used_measures = [ALL_MEASURES[config[include_key]]]
            else:
                used_measures = [ALL_MEASURES[measurename] for measurename in config[include_key]]
    elif exclude_key in config:
        if isinstance(config[exclude_key], str):
            excluded_measures = [config[exclude_key]]
        else:
            excluded_measures = config[exclude_key]
        used_measures = [measure for name, measure in ALL_MEASURES.items() if name not in excluded_measures]
    else:
        used_measures = list(ALL_MEASURES.values())

    logger.info(f"Using measures: {[m.name for m in used_measures]}")
    return used_measures


def verify_config(config: dict) -> None:
    """
    Raise error if config is not valid
    """
    if "excluded_measures" in config:
        assert isinstance(
            config["excluded_measures"], (str, list)
        ), "Excluded measures should be a string or a list of strings"
        assert "included_measures" not in config, "Cannot include and exclude measures at the same time"
        assert all(
            measure in ALL_MEASURES for measure in config["excluded_measures"]
        ), "Some measures in excluded_measures are not valid"
    if "included_measures" in config:
        assert isinstance(
            config["included_measures"], (str, list)
        ), "Included measures should be a string or a list of strings"
        assert "excluded_measures" not in config, "Cannot include and exclude measures at the same time"
        assert all(
            measure in ALL_MEASURES for measure in config["included_measures"]
        ), "Some measures in included_measures are not valid"
    if "excluded_measures" not in config and "included_measures" not in config:
        logger.info(
            "Not specifying which measures to compute. Defaulting to all."
            " Specify through 'included_measures' or 'excluded_measures'."
        )
    # Make sure either one or the other is in the config
    assert not (
        ("excluded_measures" in config) and ("included_measures" in config)
    ), "Cannot include and exclude measures at the same time"

    assert "experiments" in config, "No experiments in config."
    # for exp in config["experiments"]:
    #     filter_key_vals = config.get("filter_key_vals", None)
    #     if filter_key_vals:
    #         for key, val in filter_key_vals.items():
    #             assert isinstance(key, str), "Key should be a string"
    #             assert isinstance(val, str) or isinstance(val, list), "Value should be a string or a list of strings"

    #     differentiation_keys = config.get("differentiation_keys", None)
    #     if differentiation_keys:
    #         for key in differentiation_keys:
    #             assert isinstance(key, str), "Differentiation key should be a string"

    if "raw_results_filename" in config:
        assert config["raw_results_filename"].endswith(
            ".parquet"
        ), "The 'raw_results_filename' must end with '.parquet'"


def create_pivot_table(config: dict):
    table_cfg = config.get("table_creation", None)
    if table_cfg is not None and table_cfg.get("save_aggregated_df", True):
        return True
    return False


def create_full_table(config: dict):
    table_cfg = config.get("table_creation", None)
    if table_cfg is not None and table_cfg.get("save_full_df", False):
        return True
    return False


def run(config_path: str):
    config = read_yaml_config(config_path)
    verify_config(config)
    logger.debug("Config is valid")
    measures = get_measures(config)
    threads = config.get("threads", 1)
    cache_to_disk = config.get("cache_to_disk", False)
    cache_to_mem = config.get("cache_to_mem", False)
    rerun_nans = config.get("rerun_nans", False)
    raw_results_filename = config.get("raw_results_filename", None)
    storage_path = (
        os.path.join(EXPERIMENT_RESULTS_PATH, raw_results_filename) if raw_results_filename is not None else None
    )
    only_extract_reps = config.get("only_extract_reps", False)
    only_eval = config.get("only_eval", False)

    logger.info(f"Running with {threads} Threads! Reduce if running OOM or set to 1 for single threaded execution.")

    all_experiments = []
    for experiment in config["experiments"]:
        logger.debug(f"Creating experiment for {experiment}")
        if experiment["type"] == "GroupSeparationExperiment":
            filter_key_vals = experiment.get("filter_key_vals", None)
            grouping_keys = experiment.get("grouping_keys", None)
            separation_keys = experiment.get("separation_keys", None)

            # Not quite sure how to best make this work with the entire setup.
            #   Some form of hierarchy would need to be added, but that doesn't exist atm.
            #   Maybe create a hierarchical nested structure and make this a recursive function? to auto-aggregate?
            grouped_models: list[list[Sequence[TrainedModel]]] = get_grouped_models(
                models=ALL_TRAINED_MODELS,
                filter_key_vals=filter_key_vals,
                separation_keys=separation_keys,
                grouping_keys=grouping_keys,
            )
            for group in grouped_models:
                exp = GroupSeparationExperiment(
                    grouped_models=group,
                    measures=measures,
                    representation_dataset=experiment["representation_dataset"],
                    storage_path=storage_path,
                    threads=threads,
                    cache_to_disk=cache_to_disk,
                    cache_to_mem=cache_to_mem,
                    only_extract_reps=only_extract_reps,
                    rerun_nans=rerun_nans,
                )
                all_experiments.append(exp)

        if experiment["type"] == "OutputCorrelationExperiment":
            filter_key_vals = experiment.get("filter_key_vals", None)
            separation_keys = experiment.get("separation_keys", None)
            compare_accs = experiment.get("use_acc_comparison", False)

            models = _filter_models(ALL_TRAINED_MODELS, filter_key_vals)
            model_sets = _separate_models_by_keys(models, separation_keys)
            for models in model_sets:
                exp = OutputCorrelationExperiment(
                    models=models,
                    repsim_measures=measures,
                    functional_measures=list(FUNCTIONAL_SIMILARITY_MEASURES.values()),
                    representation_dataset=experiment["representation_dataset"],
                    storage_path=storage_path,
                    threads=threads,
                    cache_to_disk=cache_to_disk,
                    cache_to_mem=cache_to_mem,
                    only_extract_reps=only_extract_reps,
                    rerun_nans=rerun_nans,
                    use_acc_comparison=compare_accs,
                )
                all_experiments.append(exp)

        if experiment["type"] == "MonotonicityExperiment":
            filter_key_vals = experiment.get("filter_key_vals", None)
            separation_keys = experiment.get("separation_keys", None)

            models = _filter_models(ALL_TRAINED_MODELS, filter_key_vals)
            model_sets = _separate_models_by_keys(models, separation_keys)
            for models in model_sets:
                exp = MonotonicityExperiment(
                    models=models,
                    measures=measures,
                    representation_dataset=experiment["representation_dataset"],
                    storage_path=storage_path,
                    cache_to_disk=cache_to_disk,
                    cache_to_mem=cache_to_mem,
                    only_extract_reps=only_extract_reps,
                    rerun_nans=rerun_nans,
                )
                all_experiments.append(exp)
    # -------------------- Now compare/eval the grouped models ------------------- #
    exp_results = []
    for ex in all_experiments:
        if not only_eval:
            ex.run()
        if not only_extract_reps:
            exp_results.extend(ex.eval())

    if create_full_table(config) and (not only_extract_reps):
        save_full_table(exp_results, config["table_creation"]["full_df_filename"])

    if create_pivot_table(config) and (not only_extract_reps):
        table_cfg = config["table_creation"]
        create_pivot_excel_table(
            exp_results,
            row_index=table_cfg["row_index"],
            columns=table_cfg["columns"],
            value_key=table_cfg["value_key"],
            filename=table_cfg["filename"],
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Domain selection to run experiments on.",
    )
    args = parser.parse_args()
    logger.add(LOG_PATH / "{time}.log")
    logger.debug("Parsing config")
    config_path = args.config
    # config_path = os.path.join(os.path.dirname(__file__), "configs", "hierarchical_vision_shortcuts.yaml")
    run(config_path)
