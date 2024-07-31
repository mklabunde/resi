from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from repsim.benchmark.paths import EXPERIMENT_RESULTS_PATH
from repsim.benchmark.paths import LOG_PATH
from tqdm import tqdm


def read_yaml_config(config_path: str) -> dict:
    """
    Read a yaml file and return the dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # ------------ If no measures provided we default to all measures ------------ #
    return config


def get_to_be_merged_parquets(config: dict) -> list[Path]:
    """
    Reads the parquet names that are to be merged and verifies that they exist.
    """
    parquets = []
    for pq in config["parquets_to_merge"]:
        parquets.append(EXPERIMENT_RESULTS_PATH / pq)

    for pq in parquets:
        assert pq.exists(), f"Parquet {pq} does not exist."

    return parquets


def remove_duplicates(joint_df: pd.DataFrame) -> pd.DataFrame:
    clean_values = []
    unique_indices = len(np.unique(joint_df["id"]))
    if unique_indices == len(joint_df):
        logger.info("No duplicates found.")
        return joint_df

    grouped_df = joint_df.groupby("id")
    logger.info("Deduplicating dataframes...")
    for group_id, group_df in tqdm(grouped_df):
        if len(group_df) > 1:
            non_nan_vals = group_df[~group_df["metric_value"].isna()]
            if non_nan_vals.empty:
                res_dict = group_df.iloc[0].to_dict()
                res_dict["setting"] = res_dict["id"].split("_")[0]
                clean_values.append(res_dict)
            else:
                res_dict = group_df.iloc[-1].to_dict()
                res_dict["setting"] = res_dict["id"].split("_")[0]
                clean_values.append(res_dict)
        else:
            res_dict = group_df.iloc[0].to_dict()
            res_dict["setting"] = res_dict["id"].split("_")[0]
            clean_values.append(res_dict)
            # clean_values.append(group_df.iloc[0].to_dict())
    logger.info("Creating joint new df.")
    index = [v["id"] for v in clean_values]
    joint_clean_df = pd.DataFrame(clean_values, index=index)
    logger.info("Done.")
    return joint_clean_df


def get_output_parquet(config: dict, overwrite: bool = False) -> list[Path]:
    """
    Reads the parquet names that are to be merged and verifies that they exist.
    """
    out_parquet = config.get("output_parquet")
    out_parquet = EXPERIMENT_RESULTS_PATH / out_parquet
    if not overwrite:
        assert not out_parquet.exists(), f"Output parquet {out_parquet} does exist and overwrite is False."
    return out_parquet


def verify_merge_config(config: dict) -> None:
    """
    Raise error if config is not valid
    """

    pq_to_merge = config.get("parquets_to_merge", None)
    if pq_to_merge is None:
        raise ValueError("No parquets to merge in config.")
    assert isinstance(pq_to_merge, list), "Parquets to merge should be a list of strings."
    for pq in pq_to_merge:
        assert isinstance(pq, str), "Parquets to merge should be a list of strings."
    for pq in pq_to_merge:
        assert pq.endswith("parquet"), "Parquet files should end with '.parquet'."

    out_parquet = config.get("output_parquet", None)
    if out_parquet is None:
        raise ValueError("No output parquet name specified in config.")
    assert isinstance(out_parquet, str), "Output parquet should be a string."
    return


def create_pivot_table(config: dict):
    table_cfg = config.get("table_creation", None)
    save_aggregated_df = table_cfg.get("save_aggregated_df", True)
    if table_cfg is not None and save_aggregated_df:
        return True
    return False


def create_full_table(config: dict):
    table_cfg = config.get("table_creation", None)
    if table_cfg is not None and table_cfg.get("save_full_df", False):
        return True
    return False


def merge_parquets(config_path: str):
    config = read_yaml_config(config_path)
    verify_merge_config(config)
    logger.debug("Merge config is valid")
    to_be_merged_parquets: list[Path] = get_to_be_merged_parquets(config)
    out_parquet: Path = get_output_parquet(config)

    all_dataframes = [pd.read_parquet(pq) for pq in to_be_merged_parquets]

    joint_df = pd.concat(all_dataframes)
    # ------------- Remove duplicate indices from the joint dataframe ------------ #
    joint_df_without_dups = remove_duplicates(joint_df)
    # ------------- Save the joint dataframe to a new parquet ------------ #
    joint_df_without_dups.to_parquet(out_parquet)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to merge Config containing which parquets to combine.",
    )
    args = parser.parse_args()
    logger.add(LOG_PATH / "{time}.log")
    logger.debug("Parsing config")
    config_path = args.config
    # config_path = os.path.join(os.path.dirname(__file__), "configs", "hierarchical_vision_shortcuts.yaml")
    merge_parquets(config_path)
