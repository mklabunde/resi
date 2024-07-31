import multiprocessing
import time
import warnings
from collections.abc import Sequence
from itertools import product
from multiprocessing.managers import BaseManager
from multiprocessing.synchronize import Lock as LockBase

import numpy as np
import repsim.benchmark.registry
import repsim.utils
from loguru import logger
from repsim.benchmark.abstract_experiment import AbstractExperiment
from repsim.benchmark.measure_quality_metrics import auprc
from repsim.benchmark.measure_quality_metrics import violation_rate
from repsim.benchmark.utils import ExperimentStorer
from repsim.benchmark.utils import get_in_group_cross_group_sims
from repsim.benchmark.utils import get_ingroup_outgroup_SLRs
from repsim.measures.utils import RepresentationalSimilarityMeasure
from repsim.utils import SingleLayerRepresentation
from repsim.utils import suppress
from tqdm import tqdm


def flatten_nested_list(xss):
    return [x for xs in xss for x in xs]


def compare_single_measure(rep_a, rep_b, measure: RepresentationalSimilarityMeasure, shape):
    """Compare a single measure between two representations."""
    logger.info(f"Starting {measure.name}.")
    try:
        start_time = time.perf_counter()
        with suppress():  # Mute printouts of the measures
            sim = measure(rep_a, rep_b, shape)
            runtime = time.perf_counter() - start_time
    except Exception as e:
        sim = np.nan
        runtime = np.nan
        logger.error(f"'{measure.name}' comparison failed.")
        logger.error(e)

    return {
        "metric": measure,
        "metric_value": sim,
        "runtime": runtime,
    }


def gather_representations(sngl_rep_src, sngl_rep_tgt, lock):
    """Get the representations without trying to access the GPU simultaneously."""
    try:

        lock.acquire()
        # logger.debug("Acquired Lock, starting Rep extraction ...")
        with suppress():
            rep_a = sngl_rep_src.representation
            rep_b = sngl_rep_tgt.representation
            shape = sngl_rep_src.shape
    finally:
        lock.release()
    return rep_a, rep_b, shape


def compare(
    comps: list[tuple[SingleLayerRepresentation, SingleLayerRepresentation, RepresentationalSimilarityMeasure]],
    rep_lock: LockBase,
    storage_lock: LockBase,
    storer: ExperimentStorer,
) -> list[dict]:
    """
    Multithreaded comparison function with GPU blocking support.
    Does all comparisons for a single model in series, to minimize redundant representation loading.
    """
    # --------------------------- Start extracting reps -------------------------- #
    sngl_rep_src, sngl_rep_tgt, _ = comps[0]
    measures = [c[2] for c in comps]

    rep_a, rep_b, shape = gather_representations(sngl_rep_src, sngl_rep_tgt, rep_lock)
    # ----------------------------- Start metric calculation ----------------------------- #
    results: list[dict] = []
    for measure in measures:
        res = compare_single_measure(rep_a, rep_b, measure, shape)
        res["sngl_rep_src"] = sngl_rep_src
        res["sngl_rep_tgt"] = sngl_rep_tgt
        try:
            storage_lock.acquire()
            logger.info("Saved result to file")
            storer.add_results(**res)
            storer.save_to_file()
        finally:
            storage_lock.release()
    return results


class GroupSeparationExperiment(AbstractExperiment):
    def __init__(
        self,
        grouped_models: list[Sequence[repsim.benchmark.registry.TrainedModel]],
        measures: list[RepresentationalSimilarityMeasure],
        representation_dataset: str,
        storage_path: str | None = None,
        meta_data: dict | None = None,
        threads: int = 1,
        cache_to_disk: bool = False,
        cache_to_mem: bool = False,
        only_extract_reps: bool = False,
        rerun_nans: bool = False,
        **kwargs,
    ) -> None:
        """Collect all the models and datasets to be used in the experiment"""
        super().__init__(
            measures,
            representation_dataset,
            storage_path,
            threads,
            cache_to_disk,
            cache_to_mem,
            only_extract_reps,
            rerun_nans=rerun_nans,
        )
        self.groups_of_models = grouped_models
        self.meta_data = meta_data
        self.kwargs = kwargs
        self.rep_cache = {}  # lookup table for representations, so we can reuse computed representations

    def measure_violation_rate(self, measure: RepresentationalSimilarityMeasure) -> float:
        n_groups = len(self.groups_of_models)
        group_violations = []
        with ExperimentStorer(self.storage_path) as storer:
            for i in range(n_groups):
                in_group_slrs, out_group_slrs = get_ingroup_outgroup_SLRs(
                    self.groups_of_models,
                    i,
                    rep_layer_id=-1,
                    representation_dataset=self.representation_dataset,
                )

                in_group_sims, cross_group_sims = get_in_group_cross_group_sims(
                    in_group_slrs,
                    out_group_slrs,
                    measure,
                    storer,
                )

                in_group_sims = [sim for sim in in_group_sims if not np.isnan(sim)]  # <--- This changed
                cross_group_sims = [sim for sim in cross_group_sims if not np.isnan(sim)]  # <--- This changed
                # We remove NaNs and return Nones if stuff failed, so if the metric has these, we skip it! the similarity lists
                if len(in_group_sims) == 0 or len(cross_group_sims) == 0:
                    logger.info(
                        "One group was all NaN! Metric is assigned NaN as one cluster can't be compared to others making the task easier or harder."
                    )
                    return float(np.nan)

                # Calculate the violations, i.e. the number of times the in-group similarity is lower than the cross-group similarity
                group_violations.append(
                    violation_rate(
                        in_group_sims, cross_group_sims, larger_is_more_similar=measure.larger_is_more_similar
                    )
                )
        # If one quality measure returned NaN, we set all to NaN, as it might make stuff easier or harder for the test.
        return float(np.mean(group_violations))  # <--- This changed

    def auprc(self, measure: RepresentationalSimilarityMeasure) -> float:
        """Calculate the mean auprc for the in-group and cross-group similarities"""
        n_groups = len(self.groups_of_models)
        group_auprcs = []
        with ExperimentStorer(self.storage_path) as storer:
            for i in range(n_groups):
                in_group_slrs, out_group_slrs = get_ingroup_outgroup_SLRs(
                    self.groups_of_models,
                    i,
                    rep_layer_id=-1,
                    representation_dataset=self.representation_dataset,
                )
                in_group_sims, cross_group_sims = get_in_group_cross_group_sims(
                    in_group_slrs,
                    out_group_slrs,
                    measure,
                    storer,
                )
                in_group_sims = [sim for sim in in_group_sims if not np.isnan(sim)]
                cross_group_sims = [sim for sim in cross_group_sims if not np.isnan(sim)]
                # We remove NaNs and return Nones if stuff failed, so if the metric has these, we skip it! the similarity lists
                if len(in_group_sims) == 0 or len(cross_group_sims) == 0:
                    logger.info(
                        "One group was all NaN! Metric is assigned NaN as one cluster can't be compared to others making the task easier or harder."
                    )
                    return float(np.nan)
                # Calculate the area under the precision-recall curve for the in-group and cross-group similarities
                group_auprcs.append(auprc(in_group_sims, cross_group_sims, measure.larger_is_more_similar))
        # If one quality measure returned NaN, we set all to NaN, as it might make stuff easier or harder for the test.
        return float(np.mean(group_auprcs))

    def eval(self) -> list[dict]:
        """Evaluate the results of the experiment"""
        measure_wise_results: list[dict] = []
        examplary_model = self.groups_of_models[0][0]
        # This here currently assumes that both models are of the same architecture (which may not always remain true)
        meta_data = {
            "domain": examplary_model.domain,
            "architecture": examplary_model.architecture,
            "representation_dataset": self.representation_dataset,
            "identifier": examplary_model.identifier,
        }
        if self.meta_data is not None:
            meta_data.update(self.meta_data)
        logger.info(
            f"Starting to evaluate quality measures of {examplary_model.architecture} on {self.representation_dataset}"
        )
        for measure in tqdm(self.measures, desc="Evaluating quality of measures"):
            violation_rate = self.measure_violation_rate(measure)
            measure_wise_results.append(
                {
                    "similarity_measure": measure.name,
                    "quality_measure": "violation_rate",
                    "value": violation_rate,
                    **meta_data,
                }
            )
            auprc = self.auprc(measure)
            measure_wise_results.append(
                {
                    "similarity_measure": measure.name,
                    "quality_measure": "AUPRC",
                    "value": auprc,
                    **meta_data,
                }
            )
        return measure_wise_results

    def run(self) -> None:
        """Run the experiment. Results can be accessed afterwards."""
        if self.threads == 1:
            self._run_single_threaded()
        else:
            raise NotImplementedError("Multithreading was removed as compute intensive measures do it themselves.")

    def _get_todo_combos(self, combos, storer: ExperimentStorer) -> tuple[
        list[
            tuple[
                SingleLayerRepresentation,
                SingleLayerRepresentation,
                list[RepresentationalSimilarityMeasure],
            ]
        ],
        int,
    ]:

        def get_final_layer_representation(
            model: repsim.benchmark.registry.TrainedModel,
            cache_to_mem: bool = False,
        ) -> SingleLayerRepresentation:
            final_layer_rep = self.rep_cache.get(model.id, None)
            if final_layer_rep is None:
                final_layer_rep = model.get_representation(
                    self.representation_dataset, **self.kwargs
                ).representations[-1]
                if cache_to_mem:
                    self.rep_cache[model.id] = final_layer_rep
            return final_layer_rep

        comparisons_todo = []
        n_total = 0
        for model_src, model_tgt in tqdm(combos, desc="Identifying comparisons that are to do."):
            if model_src == model_tgt:
                continue  # Skip self-comparisons

            single_layer_rep_source: SingleLayerRepresentation = get_final_layer_representation(
                model_src, self.cache_to_mem
            )
            single_layer_rep_target: SingleLayerRepresentation = get_final_layer_representation(
                model_tgt, self.cache_to_mem
            )

            todo_by_measure = []

            for measure in self.measures:
                if not storer.comparison_exists(single_layer_rep_source, single_layer_rep_target, measure):
                    todo_by_measure.append(measure)
                    n_total += 1
                elif self.rerun_nans:
                    if np.isnan(storer.get_comp_result(single_layer_rep_source, single_layer_rep_target, measure)):
                        todo_by_measure.append(measure)
                        n_total += 1
            if len(todo_by_measure) > 0:
                comparisons_todo.append((single_layer_rep_source, single_layer_rep_target, todo_by_measure))
        logger.info(f"Found {n_total} comparisons to do -- Commencing.")
        return comparisons_todo, n_total

    def _run_single_threaded(self) -> None:
        """Run the experiment. Results can be accessed afterwards via the .results attribute"""
        flat_models = flatten_nested_list(self.groups_of_models)
        logger.debug(f"Using models: {[m.id for m in flat_models]}")
        combos = list(product(flat_models, flat_models))  # Necessary for non-symmetric values

        logger.info("")
        with ExperimentStorer(self.storage_path) as storer:
            todo_combos, n_total = self._get_todo_combos(combos, storer)
            self.compare_combos(todo_combos, n_total, storer, tqdm_descr="Comparing representations")

        return
