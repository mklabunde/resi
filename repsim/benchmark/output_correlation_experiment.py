from itertools import combinations
from itertools import product
from typing import Literal

import numpy as np
import repsim.benchmark.registry
import repsim.utils
import scipy.stats
from loguru import logger
from repsim.benchmark.abstract_experiment import AbstractExperiment
from repsim.benchmark.utils import ExperimentStorer
from repsim.measures import AbsoluteAccDiff
from repsim.measures.utils import FunctionalSimilarityMeasure
from repsim.measures.utils import RepresentationalSimilarityMeasure
from repsim.utils import Accuracy
from repsim.utils import Prediction
from repsim.utils import SingleLayerRepresentation
from tqdm import tqdm


class OutputCorrelationExperiment(AbstractExperiment):

    def __init__(
        self,
        models: list[repsim.benchmark.registry.TrainedModel],
        repsim_measures: list[RepresentationalSimilarityMeasure],
        functional_measures: list[FunctionalSimilarityMeasure],
        representation_dataset: str,
        storage_path: str | None = None,
        meta_data: dict | None = None,
        threads: int = 1,
        cache_to_disk: bool = False,
        cache_to_mem: bool = False,
        only_extract_reps: bool = False,
        rerun_nans: bool = False,
        use_acc_comparison: bool = False,
        **kwargs,
    ) -> None:
        """Collect all the models and datasets to be used in the experiment"""
        super().__init__(
            repsim_measures,
            representation_dataset,
            storage_path,
            threads,
            cache_to_disk,
            cache_to_mem,
            only_extract_reps,
            functional_measures=functional_measures,
            rerun_nans=rerun_nans,
        )
        self.accuracy_based_measures = [AbsoluteAccDiff()] if use_acc_comparison else []
        self.models = models
        self.meta_data = meta_data
        self.kwargs = kwargs
        self.rep_cache = {}  # lookup table for representations, so we can reuse computed representations
        self.output_cache = {}
        self.accuracy_cache = {}

    def eval(self) -> list[dict]:
        """Evaluate the results of the experiment.

        Collect all pairwise representational and functional similarities in an array, respectively, then compute
        correlation.
        """
        measure_wise_results: list[dict] = []
        examplary_model = self.models[0]
        # This here currently assumes that both models are of the same architecture (which may not always remain true)
        meta_data = {
            "domain": examplary_model.domain,
            "architecture": examplary_model.architecture,
            "representation_dataset": self.representation_dataset,
            "identifier": examplary_model.identifier,
        }
        if self.meta_data is not None:
            meta_data.update(self.meta_data)

        all_funcsim_measures = self.functional_measures + self.accuracy_based_measures
        with ExperimentStorer(self.storage_path) as storer:
            for repsim_measure, funcsim_measure in tqdm(
                product(self.measures, all_funcsim_measures),
                desc="Evaluating quality of measures",
                total=len(self.measures) * len(all_funcsim_measures),
            ):
                repsims, funcsims = [], []
                if repsim_measure.is_symmetric and funcsim_measure.is_symmetric:
                    # Using product for symmetric measures inflates the pvalue, so we use combinations in that case
                    model_pairs = combinations(self.models, r=2)
                else:
                    model_pairs = product(self.models, self.models)

                for model_pair in model_pairs:
                    if model_pair[0] == model_pair[1]:
                        continue
                    slr_a, slr_b = [self._get_final_layer_representation(m, self.cache_to_mem) for m in model_pair]
                    if isinstance(funcsim_measure, AbsoluteAccDiff):
                        pred_a, pred_b = [self._get_model_accuracy(m, self.cache_to_mem) for m in model_pair]
                    else:
                        pred_a, pred_b = [self._get_model_output(m, self.cache_to_mem) for m in model_pair]
                    repsims.append(storer.get_comp_result(slr_a, slr_b, repsim_measure))
                    funcsims.append(storer.get_comp_result(pred_a, pred_b, funcsim_measure))

                repsims = np.array(list(map(lambda x: np.nan if x is None else x, repsims)))
                funcsims = np.array(list(map(lambda x: np.nan if x is None else x, funcsims)))
                comparisons_with_at_least_one_nan = np.isnan(repsims) | np.isnan(funcsims)
                repsims = repsims[~comparisons_with_at_least_one_nan]
                funcsims = funcsims[~comparisons_with_at_least_one_nan]

                if repsim_measure.larger_is_more_similar:
                    # make similarities into distances
                    repsims = -1 * repsims

                for corr_func in [scipy.stats.pearsonr, scipy.stats.spearmanr, scipy.stats.kendalltau]:
                    if len(repsims) == 0:
                        measure_wise_results.append(
                            {
                                "similarity_measure": repsim_measure.name,
                                "functional_similarity_measure": funcsim_measure.name,
                                "quality_measure": corr_func.__name__,
                                "corr": float("nan"),
                                "pval": float("nan"),
                                **meta_data,
                            }
                        )
                    else:
                        result = corr_func(repsims, funcsims)
                        measure_wise_results.append(
                            {
                                "similarity_measure": repsim_measure.name,
                                "functional_similarity_measure": funcsim_measure.name,
                                "quality_measure": corr_func.__name__,
                                "corr": result.statistic,
                                "pval": result.pvalue,
                                **meta_data,
                            }
                        )

        return measure_wise_results

    def _get_final_layer_representation(
        self, model: repsim.benchmark.registry.TrainedModel, cache_to_mem: bool = False
    ) -> SingleLayerRepresentation:
        final_layer_rep = self.rep_cache.get(model.id, None)
        if final_layer_rep is None:
            final_layer_rep = model.get_representation(self.representation_dataset, **self.kwargs).representations[-1]
            if cache_to_mem:
                self.rep_cache[model.id] = final_layer_rep
        return final_layer_rep

    def _get_model_output(
        self, model: repsim.benchmark.registry.TrainedModel, cache_to_mem: bool = False
    ) -> Prediction:
        output = self.output_cache.get(model.id, None)
        if output is None:
            output = model.get_output(self.representation_dataset, **self.kwargs)
            if cache_to_mem:
                self.output_cache[model.id] = output
        return output

    def _get_model_accuracy(
        self, model: repsim.benchmark.registry.TrainedModel, cache_to_mem: bool = False
    ) -> Accuracy:
        acc = self.accuracy_cache.get(model.id, None)
        if acc is None:
            acc = model.get_accuracy(self.representation_dataset, **self.kwargs)
            if cache_to_mem:
                self.accuracy_cache[model.id] = acc
        return acc

    def _get_todo_combos(
        self, combos, storer: ExperimentStorer, type: Literal["representations", "predictions", "accuracy"]
    ) -> tuple[
        list[
            tuple[
                Prediction,
                Prediction,
                list[FunctionalSimilarityMeasure],
            ]
            | tuple[
                SingleLayerRepresentation,
                SingleLayerRepresentation,
                list[RepresentationalSimilarityMeasure],
            ]
            | tuple[
                Accuracy,
                Accuracy,
                list[FunctionalSimilarityMeasure],
            ]
        ],
        int,
    ]:
        if type == "representations":
            logger.debug("Checking for representation TODOs.")
            get_obj_func = self._get_final_layer_representation
            measures = self.measures
        elif type == "predictions":
            logger.debug("Checking for prediction TODOs.")
            get_obj_func = self._get_model_output
            measures = self.functional_measures
        elif type == "accuracy":
            logger.debug("Checking for accuracy TODOs.")
            get_obj_func = self._get_model_accuracy
            measures = self.accuracy_based_measures
        else:
            raise ValueError(f"Unexpected {type=}. Must be one of ['representations', 'predictions'].")

        comparisons_todo = []
        n_total = 0
        for model_src, model_tgt in combos:
            if model_src == model_tgt:
                continue  # Skip self-comparisons

            obj_source = get_obj_func(model_src, self.cache_to_mem)
            obj_target = get_obj_func(model_tgt, self.cache_to_mem)

            todo_by_measure = []

            for measure in measures:
                if not storer.comparison_exists(obj_source, obj_target, measure):
                    todo_by_measure.append(measure)
                    n_total += 1
                elif self.rerun_nans:
                    if np.isnan(storer.get_comp_result(obj_source, obj_target, measure)):  # type:ignore
                        todo_by_measure.append(measure)
                        n_total += 1
            if len(todo_by_measure) > 0:
                comparisons_todo.append((obj_source, obj_target, todo_by_measure))
        return comparisons_todo, n_total

    def run(self) -> None:
        """Run the experiment. Results can be accessed afterwards via the .results attribute"""
        logger.debug(f"Using models: {[m.id for m in self.models]}")

        logger.info("")
        with ExperimentStorer(self.storage_path) as storer:
            combos = product(self.models, self.models)  # Necessary for non-symmetric values
            todo_combos, n_total = self._get_todo_combos(combos, storer, type="representations")
            self.compare_combos(todo_combos, n_total, storer, tqdm_descr="Comparing representations")

            combos = product(self.models, self.models)
            todo_combos, n_total = self._get_todo_combos(combos, storer, type="predictions")
            self.compare_combos(todo_combos, n_total, storer, tqdm_descr="Comparing predictions")

            combos = product(self.models, self.models)
            todo_combos, n_total = self._get_todo_combos(combos, storer, type="accuracy")
            self.compare_combos(todo_combos, n_total, storer, tqdm_descr="Comparing accuracy")
        return
