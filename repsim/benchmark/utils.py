import os
from pathlib import Path
import warnings
from collections.abc import Sequence
from dataclasses import asdict
from itertools import chain
from itertools import combinations
from itertools import product

import numpy as np
import pandas as pd
from loguru import logger
from repsim.benchmark.paths import EXPERIMENT_RESULTS_PATH
from repsim.benchmark.registry import TrainedModel
from repsim.measures.utils import BaseSimilarityMeasure
from repsim.measures.utils import RepresentationalSimilarityMeasure
from repsim.utils import BaseModelOutput
from repsim.utils import SingleLayerRepresentation


class ExperimentStorer:

    def __init__(self, path_to_store: str | None = None) -> None:
        if path_to_store is None:
            path_to_store = os.path.join(EXPERIMENT_RESULTS_PATH, "experiments.parquet")
        self.path_to_store = path_to_store

        Path(path_to_store).parent.mkdir(parents=True, exist_ok=True)
        # Create cache dir if it's not there already.
        (Path(path_to_store).parent.parent / "cache").mkdir(parents=True, exist_ok=True)
        self._old_experiments = (
            pd.read_parquet(self.path_to_store) if os.path.exists(self.path_to_store) else pd.DataFrame()
        )
        self._overwrite_indices: list[str] = []
        self._sanity_check_parquett()
        self._new_experiments = pd.DataFrame()

    def _sanity_check_parquett(self) -> None:
        """
        Check if the parquet file is corrupted.
        """
        indices = self._old_experiments.index
        if len(indices) != len(set(indices)):
            logger.error("The indices are not unique.")

    def add_results(
        self,
        src_single_rep: BaseModelOutput,
        tgt_single_rep: BaseModelOutput,
        metric: BaseSimilarityMeasure,
        metric_value: float,
        runtime: float | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Add a comparison result of the experiments to disk.
        Serializes all the information into a unique identifier and stores the results in a pandas dataframe.
        """

        if metric.is_symmetric:
            reps = [(src_single_rep, tgt_single_rep), (tgt_single_rep, src_single_rep)]
        else:
            reps = [(src_single_rep, tgt_single_rep)]

        for source_rep, target_rep in reps:
            comp_id = self._get_comparison_id(
                src_single_rep=source_rep,
                tgt_single_rep=target_rep,
                metric_name=metric.name,
            )
            comp_exists = False
            if self.comparison_exists(src_single_rep, tgt_single_rep, metric, ignore_symmetry=True) and not overwrite:
                comp_exists = True
                logger.info("Comparison already exists and Overwrite is False. Skipping.")
                continue

            try:
                import git

                repo = git.Repo(search_parent_directories=True)
                sha = repo.head.object.hexsha
            except ImportError:
                GIT_SHA_PATH = ".git/refs/heads/main"
                with open(GIT_SHA_PATH, "r") as f:
                    sha = f.readline().strip()

            ids_of_interest = [
                "layer_id",
                "_architecture_name",
                "_train_dataset",
                "_representation_dataset",
                "_seed",
                "_setting_identifier",
            ]
            content = {"source_" + k: v for k, v in asdict(source_rep).items() if k in ids_of_interest}
            content.update({"target_" + k: v for k, v in asdict(target_rep).items() if k in ids_of_interest})
            content.update(
                {
                    "metric": metric.name,
                    "metric_value": metric_value,
                    "runtime": runtime,
                    "id": comp_id,
                    "is_symmetric": metric.is_symmetric,
                    "larger_is_more_similar": metric.larger_is_more_similar,
                    "is_metric": metric.is_metric,
                    "invariant_to_affine": metric.invariant_to_affine,
                    "invariant_to_invertible_linear": metric.invariant_to_invertible_linear,
                    "invariant_to_ortho": metric.invariant_to_ortho,
                    "invariant_to_permutation": metric.invariant_to_permutation,
                    "invariant_to_isotropic_scaling": metric.invariant_to_isotropic_scaling,
                    "invariant_to_translation": metric.invariant_to_translation,
                }
            )
            content.update({"hash": sha, "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")})
            content_df = pd.DataFrame(content, index=[comp_id])
            if overwrite and comp_exists:
                self._overwrite_indices.append(comp_id)
            self._new_experiments = self._new_experiments._append(content_df, ignore_index=False)

    def _sort_models(
        self,
        single_rep_a: BaseModelOutput,
        single_rep_b: BaseModelOutput,
    ) -> tuple[BaseModelOutput, BaseModelOutput]:
        """Return the SingeLayerRepresentations in a sorted order, to avoid permutation issues."""
        id_a, id_b = single_rep_a.unique_identifier(), single_rep_b.unique_identifier()
        if id_a < id_b:
            return single_rep_a, single_rep_b
        return single_rep_b, single_rep_a

    def _get_comparison_id(
        self,
        src_single_rep: BaseModelOutput,
        tgt_single_rep: BaseModelOutput,
        metric_name: str,
    ) -> str:
        """
        Serialize the experiment setting into a unique identifier that can be used to index if it already exists in the dataframe.

        Args:
            single_rep_a (BaseModelOutput): The representation of a single layer in model A.
            single_rep_b (BaseModelOutput): The representation of a single layer in model B.
            metric_name (str): The name of the metric used for comparison.

        Returns:
            str: A unique identifier that represents the experiment setting.
        """
        src_id_reps = src_single_rep.unique_identifier()
        tgt_id_reps = tgt_single_rep.unique_identifier()

        first_id, second_id = [src_id_reps, tgt_id_reps]
        joint_id = "___".join([first_id, second_id, metric_name])
        return joint_id

    def get_comp_result(
        self,
        src_single_rep: BaseModelOutput,
        tgt_single_rep: BaseModelOutput,
        metric: BaseSimilarityMeasure,
    ) -> float | None:
        """
        Return the result of the comparison
        Arsg:
            src_single_rep: BaseModelOutput
            tgt_single_rep: BaseModelOutput
            metric: SimilarityMeasure
        Returns:
            float: The result of the comparison
        Raises:
            ValueError: If the comparison does not exist in the dataframe.
        """
        comp_id = self._get_comparison_id(src_single_rep, tgt_single_rep, metric.name)
        symm_comp_id = self._get_comparison_id(tgt_single_rep, src_single_rep, metric.name)
        if len(self._new_experiments) > 0:
            experiment = pd.concat([self._old_experiments, self._new_experiments], ignore_index=False)
        else:
            experiment = (
                self._old_experiments
            )  # pd.concat([self._old_experiments, self._new_experiments], ignore_index=False)
        normal_exists = comp_id in experiment.index
        symm_exists = symm_comp_id in experiment.index

        # ----------------- Read or add symmetric result to dataframe ---------------- #
        if normal_exists and symm_exists:
            res = experiment.loc[comp_id]
        elif normal_exists and (not symm_exists):
            res = experiment.loc[comp_id]
            self.add_results(
                src_single_rep=tgt_single_rep,
                tgt_single_rep=src_single_rep,
                metric=metric,
                metric_value=res["metric_value"],
                runtime=res["runtime"],
            )
            # self.save_to_file()
        elif (not normal_exists) and symm_exists and metric.is_symmetric:
            res = experiment.loc[symm_comp_id]
            self.add_results(
                src_single_rep=src_single_rep,
                tgt_single_rep=tgt_single_rep,
                metric=metric,
                metric_value=res["metric_value"],
                runtime=res["runtime"],
            )
            # self.save_to_file()
        else:
            logger.warning(f"Comparison {comp_id} does not exist in the dataframe. -- Skipping")
            return None
            # raise ValueError("Comparison does not exist in the dataframe.")

        metric_values = res["metric_value"]
        if isinstance(metric_values, (np.float32, np.float64, np.float16, float)):
            sim_value = metric_values
        elif isinstance(metric_values, pd.Series):
            logger.warning(f"Multiple entries found for {comp_id}.")
            metric_values = res["metric_value"]
            metric_values = [v for v in metric_values if not np.isnan(v)]
            if len(metric_values) > 0 and all(
                np.isclose(value, metric_values[0], atol=1e-6) for value in metric_values if value
            ):
                sim_value = metric_values[0]
            else:
                logger.error("Multiple different values found for the same comparison. Returning nan.")
                sim_value = np.nan
        else:
            sim_value = None

        return sim_value

    def comparison_exists(
        self,
        src_single_rep: BaseModelOutput,
        tgt_single_rep: BaseModelOutput,
        metric: BaseSimilarityMeasure,
        ignore_symmetry: bool = False,
    ) -> bool:
        """
        Check if the comparison (or if symmetrict the inverse) already exists in the dataframe.
        Args:
            src_single_rep: BaseModelOutput  # Represents the source model that in non-symmetric cases
            tgt_single_rep: BaseModelOutput  # represents the target in non-symmetric cases
            metric: SimilarityMeasure
        Returns:
            bool: True if the comparison exists, False otherwise.
        """
        comp_id = self._get_comparison_id(src_single_rep, tgt_single_rep, metric.name)
        comp_in_old = comp_id in self._old_experiments.index
        comp_in_new = comp_id in self._new_experiments.index
        if (not metric.is_symmetric) or ignore_symmetry:
            return comp_in_old or comp_in_new

        comp_id_symm = self._get_comparison_id(tgt_single_rep, src_single_rep, metric.name)
        comp_in_old_symm = comp_id_symm in self._old_experiments.index
        comp_in_new_symm = comp_id_symm in self._new_experiments.index

        return comp_in_old or comp_in_new or comp_in_old_symm or comp_in_new_symm

    def save_to_file(self) -> None:
        """Save the results of the experiment to disk"""
        # Re-read to make sure that other processes have not written to the file in the meantime.
        latest_experiment_results = (
            pd.read_parquet(self.path_to_store) if os.path.exists(self.path_to_store) else pd.DataFrame()
        )
        if len(self._overwrite_indices) > 0:
            # Need to ignore errors in case multiple processes try to overwrite the same index.
            latest_experiment_results.drop(self._overwrite_indices, inplace=True, errors="ignore")
            self._overwrite_indices = []
        # Read all the experiments and make sure that we do not overwrite any existing ones.
        all_experiments = pd.concat([latest_experiment_results, self._new_experiments], ignore_index=False)
        all_experiments.to_parquet(self.path_to_store)
        self._new_experiments = pd.DataFrame()  # Empty the new experiments or duplicates will be written.
        self._old_experiments = pd.read_parquet(self.path_to_store)  # Refresh the available data

    def __enter__(self):
        """When entering a context, load the experiments from disk"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """When exiting a context, save the experiments to disk"""
        if len(self._new_experiments) > 0:
            self.save_to_file()
        self.experiments = None
        return False


def get_in_group_cross_group_sims(
    in_group_slrs, out_group_slrs, measure: BaseSimilarityMeasure, storer: ExperimentStorer
):
    """
    Get the in-group and cross-group similarities for a given measure.
    Args:
        in_group_slrs: List of BaseModelOutputs of the in-group models.
        out_group_slrs: List of BaseModelOutputs of the out-group models.
        measure_name: Name of the measure to be used.
        storer: ExperimentStorer object to store and retrieve the results.
        Returns:
        in_group_sims: List of in-group similarities.
        cross_group_sims: List of cross-group similarities.
    """
    in_group_comps = list(combinations(in_group_slrs, 2))
    in_group_comps_existing = [
        (slr1, slr2) for slr1, slr2 in in_group_comps if storer.comparison_exists(slr1, slr2, measure)
    ]

    cross_group_comps = list(product(in_group_slrs, out_group_slrs))
    cross_group_comps_existing = [
        (slr1, slr2) for slr1, slr2 in cross_group_comps if storer.comparison_exists(slr1, slr2, measure)
    ]

    if len(in_group_comps) != len(in_group_comps_existing):
        logger.warning(f"Only {len(cross_group_comps)} out of {len(in_group_comps)} in-group comparisons exist.")

    if len(cross_group_comps) != len(cross_group_comps_existing):
        logger.warning(
            f"Only {len(cross_group_comps_existing)} out of {len(cross_group_comps)} cross-group comparisons exist."
        )

    # Redo to not have empty iterable
    # in_group_comps = combinations(in_group_slrs, 2)
    # cross_group_comps = product(in_group_slrs, out_group_slrs)
    in_group_sims = [storer.get_comp_result(slr1, slr2, measure) for slr1, slr2 in in_group_comps_existing]
    cross_group_sims = [storer.get_comp_result(slr1, slr2, measure) for slr1, slr2 in cross_group_comps_existing]

    # ToDo: Make sure that the None values are handled correctly.
    not_none_in_group_sims = [sim for sim in in_group_sims if sim is not None]
    not_none_cross_group_sims = [sim for sim in cross_group_sims if sim is not None]

    return not_none_in_group_sims, not_none_cross_group_sims


def get_ingroup_outgroup_SLRs(
    groups_of_models: tuple[list[TrainedModel]], in_group_id: int, rep_layer_id: int, representation_dataset: str
) -> tuple[list[SingleLayerRepresentation], list[SingleLayerRepresentation]]:
    n_groups = set(range(len(groups_of_models)))

    out_group_ids = n_groups - {in_group_id}
    in_group_slrs = [
        m.get_representation(representation_dataset).representations[rep_layer_id]
        for m in groups_of_models[in_group_id]
    ]
    out_group_slrs = [
        m.get_representation(representation_dataset).representations[rep_layer_id]
        for m in chain(*[groups_of_models[out_id] for out_id in out_group_ids])
    ]
    return in_group_slrs, out_group_slrs


def create_pivot_excel_table(
    eval_result: list[dict],
    row_index: str | Sequence[str],
    columns: str | Sequence[str],
    value_key: str,
    filename: str,
    sheet_name: str = "Sheet1",
) -> None:
    """
    Convert the evaluation result to a pandas dataframe
    Args:
        eval_result: Dictionary of evaluation results.
    Returns:
        None, but writes out a table to disk.
    """
    result_df = pd.DataFrame(eval_result)
    pivoted_result = result_df.pivot_table(index=row_index, columns=columns, values=value_key)
    file_path = os.path.join(EXPERIMENT_RESULTS_PATH, filename)
    if filename.endswith(".xlsx"):
        with pd.ExcelWriter(file_path) as writer:
            pivoted_result.to_excel(writer, sheet_name=sheet_name)
    elif filename.endswith(".csv"):
        pivoted_result.to_csv(file_path)
    elif filename.endswith(".tex"):
        pivoted_result.to_latex(file_path)
    else:
        raise ValueError(f"Unsupported file format: {filename}")


def save_full_table(eval_result: list[dict], full_df_filename: str):
    df = pd.DataFrame.from_records(eval_result)
    df.to_csv(os.path.join(EXPERIMENT_RESULTS_PATH, full_df_filename))
