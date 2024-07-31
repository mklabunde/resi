import itertools

import numpy as np
import repsim.benchmark.registry
import repsim.utils
import scipy.stats
from loguru import logger
from repsim.benchmark.abstract_experiment import AbstractExperiment
from repsim.benchmark.utils import ExperimentStorer
from repsim.measures.utils import RepresentationalSimilarityMeasure
from repsim.utils import SingleLayerRepresentation
from tqdm import tqdm


class MonotonicityExperiment(AbstractExperiment):
    def __init__(
        self,
        models: list[repsim.benchmark.registry.TrainedModel],
        measures: list[RepresentationalSimilarityMeasure],
        representation_dataset: str,
        storage_path: str | None = None,
        meta_data: dict | None = None,
        cache_to_disk: bool = False,
        cache_to_mem: bool = True,
        only_extract_reps: bool = False,
        rerun_nans: bool = False,
        **kwargs,
    ) -> None:
        """Collect all the models and datasets to be used in the experiment"""
        super().__init__(
            measures=measures,
            representation_dataset=representation_dataset,
            storage_path=storage_path,
            cache_to_disk=cache_to_disk,
            cache_to_mem=cache_to_mem,
            only_extract_reps=only_extract_reps,
            threads=1,
            rerun_nans=rerun_nans,
        )
        self.models = models
        self.meta_data = meta_data
        self.kwargs = kwargs
        self.storage_path = storage_path
        self.rep_cache: dict[str, repsim.utils.ModelRepresentations] = (
            {}
        )  # lookup table for representations, so we can reuse computed representations

    def _all_have_same_attr_val(self, obj_list, attr_name):
        # Code generated via ChatGPT-3.5
        if not obj_list:  # If the list is empty, return False
            return False
        first_obj = obj_list[0]  # Get the first object in the list
        first_attr = getattr(first_obj, attr_name)  # Get the attribute value of the first object

        # Check if the attribute value of all objects matches the attribute value of the first object
        for obj in obj_list[1:]:
            if getattr(obj, attr_name) != first_attr:
                return False
        return True

    def corr_btw_sim_and_layer_distance(
        self, measure: RepresentationalSimilarityMeasure, model: repsim.utils.TrainedModel
    ) -> float:
        with ExperimentStorer(self.storage_path) as storer:
            reps = self._get_all_layer_representations(
                model, self.cache_to_mem, do_forward_pass=False
            ).representations
            n_reps = len(reps)

            forward_corrs = []
            for src_rep_layer in range(n_reps):
                forward_sims = []
                forward_dists = []  # layer distance
                for tgt_rep_layer in range(src_rep_layer + 1, n_reps):
                    src_to_target_sim = storer.get_comp_result(reps[src_rep_layer], reps[tgt_rep_layer], measure)
                    if src_to_target_sim is None:
                        logger.warning(f"Similarity score is missing ({reps[src_rep_layer]}, {reps[tgt_rep_layer]})")
                        continue
                    assert (
                        src_to_target_sim is not None
                    ), f"Similarity score is missing ({reps[src_rep_layer]}, {reps[tgt_rep_layer]})"
                    forward_sims.append(src_to_target_sim)
                    forward_dists.append(tgt_rep_layer - src_rep_layer)
                if len(forward_sims) == 0:
                    corr = float("nan")
                else:
                    corr, _ = scipy.stats.spearmanr(forward_sims, forward_dists)
                if measure.larger_is_more_similar:
                    # Similarity decreases with increasing distance.
                    # To be uniform with distance measures, reverse the correlation.
                    corr = -1 * corr
                forward_corrs.append(corr)

            backward_corrs = []
            for src_rep_layer in range(n_reps):
                backward_sims = []
                backward_dists = []  # layer distance
                for tgt_rep_layer in range(0, src_rep_layer):
                    src_to_target_sim = storer.get_comp_result(reps[src_rep_layer], reps[tgt_rep_layer], measure)
                    if src_to_target_sim is None:
                        logger.warning(f"Similarity score is missing ({reps[src_rep_layer]}, {reps[tgt_rep_layer]})")
                        continue
                    assert (
                        src_to_target_sim is not None
                    ), f"Similarity score is missing ({reps[src_rep_layer]}, {reps[tgt_rep_layer]})"
                    backward_sims.append(src_to_target_sim)
                    backward_dists.append(src_rep_layer - tgt_rep_layer)
                if len(backward_sims) == 0:
                    corr = float("nan")
                else:
                    corr, _ = scipy.stats.spearmanr(backward_sims, backward_dists)
                if measure.larger_is_more_similar:
                    corr = -1 * corr
                backward_corrs.append(corr)

            return float(np.nanmean(forward_corrs + backward_corrs))

    def violation_rate(self, measure: RepresentationalSimilarityMeasure, model: repsim.utils.TrainedModel) -> float:
        with ExperimentStorer(self.storage_path) as storer:
            reps = self._get_all_layer_representations(
                model, self.cache_to_mem, do_forward_pass=False
            ).representations
            n_reps = len(reps)

            n_violations = 0
            n_total = 0
            # Comparing a layer to all higher layers
            for src_rep_layer in range(n_reps):
                for tgt_rep_layer in range(src_rep_layer + 1, n_reps):
                    sim_src_to_target = storer.get_comp_result(reps[src_rep_layer], reps[tgt_rep_layer], measure)
                    if sim_src_to_target is None:
                        logger.warning(f"Similarity score is missing ({reps[src_rep_layer]}, {reps[tgt_rep_layer]})")
                        continue
                    for further_away_tgt_rep_layer in range(tgt_rep_layer + 1, n_reps):
                        sim_src_to_further_away_target = storer.get_comp_result(
                            reps[src_rep_layer], reps[further_away_tgt_rep_layer], measure
                        )
                        if sim_src_to_further_away_target is None:
                            logger.warning(
                                f"Similarity score is missing ({reps[src_rep_layer]}, {reps[tgt_rep_layer]})"
                            )
                            continue
                        assert (
                            sim_src_to_target is not None
                        ), f"Similarity score is missing ({reps[src_rep_layer]}, {reps[tgt_rep_layer]})"
                        assert (
                            sim_src_to_further_away_target is not None
                        ), f"Similarity score is missing ({reps[src_rep_layer]}, {reps[further_away_tgt_rep_layer]})"
                        n_total += 1
                        if measure.larger_is_more_similar and sim_src_to_target < sim_src_to_further_away_target:
                            n_violations += 1
                        elif (
                            not measure.larger_is_more_similar and sim_src_to_target > sim_src_to_further_away_target
                        ):
                            n_violations += 1

            # Comparing a layer to all lower layers
            for src_rep_layer in range(n_reps):
                for tgt_rep_layer in range(0, src_rep_layer):
                    sim_src_to_target = storer.get_comp_result(reps[src_rep_layer], reps[tgt_rep_layer], measure)
                    if sim_src_to_target is None:
                        logger.warning(f"Similarity score is missing ({reps[src_rep_layer]}, {reps[tgt_rep_layer]})")
                        continue
                    for further_away_tgt_rep_layer in range(0, tgt_rep_layer):
                        sim_src_to_further_away_target = storer.get_comp_result(
                            reps[src_rep_layer], reps[further_away_tgt_rep_layer], measure
                        )
                        if sim_src_to_further_away_target is None:
                            logger.warning(
                                f"Similarity score is missing ({reps[src_rep_layer]}, {reps[tgt_rep_layer]})"
                            )
                            continue
                        assert (
                            sim_src_to_target is not None
                        ), f"Similarity score is missing ({reps[src_rep_layer]}, {reps[tgt_rep_layer]})"
                        assert (
                            sim_src_to_further_away_target is not None
                        ), f"Similarity score is missing ({reps[src_rep_layer]}, {reps[further_away_tgt_rep_layer]})"
                        n_total += 1
                        if measure.larger_is_more_similar and sim_src_to_target < sim_src_to_further_away_target:
                            n_violations += 1
                        elif (
                            not measure.larger_is_more_similar  # distances instead of similarities
                            and sim_src_to_target > sim_src_to_further_away_target
                        ):
                            n_violations += 1
            if n_total == 0:
                return float("nan")
            else:
                return n_violations / n_total

    def eval(self) -> list[dict]:
        """Evaluate the results of the experiment"""
        results: list[dict] = []
        examplary_model = self.models[0]
        assert self._all_have_same_attr_val(self.models, "domain")
        assert self._all_have_same_attr_val(self.models, "architecture")
        assert self._all_have_same_attr_val(self.models, "identifier")
        # This here currently assumes that both models are of the same architecture (which may not always remain true)
        meta_data = {
            "domain": examplary_model.domain,
            "architecture": examplary_model.architecture,
            "representation_dataset": self.representation_dataset,
            "identifier": examplary_model.identifier,
        }

        if self.meta_data is not None:
            meta_data.update(self.meta_data)

        for measure in tqdm(self.measures, desc="Evaluating quality of measures"):
            for model in self.models:
                violation_rate = self.violation_rate(measure, model)
                results.append(
                    {
                        "similarity_measure": measure.name,
                        "quality_measure": "violation_rate",
                        "model": model.id,
                        "value": violation_rate,
                        **meta_data,
                    }
                )

                corr = self.corr_btw_sim_and_layer_distance(measure, model)
                results.append(
                    {
                        "similarity_measure": measure.name,
                        "quality_measure": "correlation",
                        "model": model.id,
                        "value": corr,
                        **meta_data,
                    }
                )
        return results

    def _get_all_layer_representations(
        self,
        model: repsim.benchmark.registry.TrainedModel,
        cache_to_mem: bool = False,
        do_forward_pass: bool = True,
    ) -> repsim.utils.ModelRepresentations:
        if do_forward_pass:
            # Do the forward pass to fill the .representations in ModelRepresentations all at once with a single forward pass
            # The argument compute_on_demand is exclusive to NLPModel, but others take kwargs that are not passed further
            reps = model.get_representation(self.representation_dataset, compute_on_demand=False, **self.kwargs)
        elif self.rep_cache.get(model.id, None) is None:
            # Leave it up to the implementation whether a forward pass is done or not.
            reps = model.get_representation(self.representation_dataset, **self.kwargs)
        else:
            reps = self.rep_cache[model.id]

        if model.domain == "VISION":
            # Only use that last 5 layers for monotonicity to keep spatial extent low.
            reps.representations = self._only_lowres_vision_layers(model, reps)

        if cache_to_mem:
            self.rep_cache[model.id] = reps

        return reps

    def _find_todos(self, representations: tuple[SingleLayerRepresentation, ...], storer: ExperimentStorer) -> tuple[
        list[
            tuple[
                SingleLayerRepresentation,
                SingleLayerRepresentation,
                list[RepresentationalSimilarityMeasure],
            ]
        ],
        int,
    ]:
        comparisons_todo = []
        n_total = 0
        for rep_src, rep_tgt in itertools.product(representations, repeat=2):
            if rep_src == rep_tgt:
                continue  # Skip self-comparisons

            todo_by_measure = []
            for measure in self.measures:
                if not storer.comparison_exists(rep_src, rep_tgt, measure):
                    todo_by_measure.append(measure)
                    n_total += 1
                elif self.rerun_nans:
                    if np.isnan(storer.get_comp_result(rep_src, rep_tgt, measure)):  # type:ignore
                        todo_by_measure.append(measure)
                        n_total += 1
            if len(todo_by_measure) > 0:
                comparisons_todo.append((rep_src, rep_tgt, todo_by_measure))
        return comparisons_todo, n_total

    def _only_lowres_vision_layers(
        self, model: repsim.utils.TrainedModel, modelreps: repsim.utils.ModelRepresentations
    ) -> tuple[repsim.utils.SingleLayerVisionRepresentation]:
        if model.architecture == "VGG11":
            n_last = 5
        # elif model.architecture == "VGG19":
        #     n_last = 8
        elif model.architecture == "ResNet18":
            n_last = 5
        else:  # We don't use anymore than the 10 last layers.
            n_last = 6
        # n_last = 5

        return modelreps.representations[-n_last:]

    def _get_todo_combos(self, model: repsim.utils.TrainedModel, storer: ExperimentStorer) -> tuple[
        list[
            tuple[
                SingleLayerRepresentation,
                SingleLayerRepresentation,
                list[RepresentationalSimilarityMeasure],
            ]
        ],
        int,
    ]:
        # First check todos with potentially not-yet-computed representations
        modelreps = self._get_all_layer_representations(model, self.cache_to_mem, do_forward_pass=False)
        comparisons_todo, n_total = self._find_todos(modelreps.representations, storer)
        if n_total > 0:
            # Populate the reps for all SingleLayerRepresentations. Before might have been None to skip the forward pass.
            modelreps = self._get_all_layer_representations(model, self.cache_to_mem, do_forward_pass=True)
            comparisons_todo, n_total = self._find_todos(modelreps.representations, storer)

        return comparisons_todo, n_total

    def run(self) -> None:
        """Run the experiment. Results can be accessed afterwards via the .results attribute"""
        logger.debug(f"Using models: {[m.id for m in self.models]}")

        with ExperimentStorer(self.storage_path) as storer:
            for model in self.models:
                logger.debug(f"Comparing layers of {model}")
                todo_combos, n_total = self._get_todo_combos(model, storer)
                self.compare_combos(todo_combos, n_total, storer, tqdm_descr="Comparing representations")
        return
