import time
from abc import abstractmethod
from collections.abc import Sequence

import numpy as np
from loguru import logger
from repsim.benchmark.utils import ExperimentStorer
from repsim.measures.utils import FunctionalSimilarityMeasure
from repsim.measures.utils import RepresentationalSimilarityMeasure
from repsim.utils import Accuracy
from repsim.utils import Prediction
from repsim.utils import SingleLayerRepresentation
from repsim.utils import suppress
from tqdm import tqdm


class AbstractExperiment:
    def __init__(
        self,
        measures: list[RepresentationalSimilarityMeasure],
        representation_dataset: str,
        storage_path: str | None = None,
        threads: int = 1,
        cache_to_disk: bool = False,
        cache_to_mem: bool = False,
        only_extract_reps: bool = False,
        functional_measures: list[FunctionalSimilarityMeasure] | None = None,
        rerun_nans: bool = False,
    ) -> None:
        """
        Needs the measures to be employed, the dataset to be used and the path where to store the results.
        """
        self.measures: list[RepresentationalSimilarityMeasure] = measures
        self.representation_dataset: str = representation_dataset
        self.storage_path = storage_path
        self.cache_to_disk = cache_to_disk
        self.cache_to_mem = cache_to_mem
        self.rerun_nans: bool = rerun_nans
        self.threads = threads
        self.only_extract_reps: bool = only_extract_reps
        self.functional_measures = [] if functional_measures is None else functional_measures

    @abstractmethod
    def eval(self) -> None:
        """Evaluate the results of the experiment"""
        pass

    @abstractmethod
    def run(self) -> None:
        """Run the experiment. Results can be accessed afterwards via the experiment storer."""

    def compare_combos(
        self,
        todo_combos: Sequence[
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
        n_total: int,
        storer: ExperimentStorer,
        tqdm_descr: str = "",
    ):
        if len(todo_combos) == 0:
            return
        obj_src, _, _ = todo_combos[0]
        value_attr_name = obj_src.value_attr_name()

        with tqdm(total=n_total, desc=tqdm_descr) as pbar:
            for obj_src, obj_tgt, measures in todo_combos:

                obj_src.cache = self.cache_to_disk  # Optional persistent cache to disk
                obj_tgt.cache = self.cache_to_disk  # Optional persistent cache to disk
                for measure in measures:
                    overwrite = False
                    if storer.comparison_exists(obj_src, obj_tgt, measure):
                        if self.rerun_nans and np.isnan(storer.get_comp_result(obj_src, obj_tgt, measure)):
                            logger.info(f"Rerunning NaN comparison for '{measure.name}'.")
                            overwrite = True
                        else:
                            pbar.update(1)
                            continue
                    try:
                        vals_a = getattr(obj_src, value_attr_name)
                        vals_b = getattr(obj_tgt, value_attr_name)
                        shape = getattr(obj_src, "shape", None)
                        if obj_src.origin_model.domain == "VISION":
                            if isinstance(obj_src, Accuracy):  # Don't do anything if accuracy measure
                                pass
                            elif len(vals_a.shape) != len(vals_b.shape):
                                max_shape = max(len(vals_a.shape), len(vals_b.shape))
                                if max_shape == 4:
                                    if len(vals_a.shape) == 2:
                                        vals_a = vals_a[..., None, None]
                                        shape = "nchw"  # we only need to set this here, as this shape is taken as reference.
                                    if len(vals_b.shape) == 2:
                                        vals_b = vals_b[..., None, None]
                                elif max_shape == 3:
                                    if len(vals_a.shape) == 2:
                                        vals_a = vals_a[:, None, :]
                                        shape = "ntd"  # we only need to set this here, as this shape is taken as reference.
                                    if len(vals_b.shape) == 2:
                                        vals_b = vals_b[:, None, :]
                        if self.only_extract_reps:
                            logger.info("Only extracting representations. Skipping comparison.")
                            # Break as all measures use the same rep.
                            pbar.update(len(measures))
                            break  # Skip the actual comparison and prepare all reps for e.g. a CPU only machine.

                        logger.debug(f"Starting: {measure.name}")
                        start_time = time.perf_counter()
                        with suppress():  # Mute printouts of the measures
                            if shape is not None:
                                assert isinstance(measure, RepresentationalSimilarityMeasure)
                                sim = measure(vals_a, vals_b, shape)
                            else:
                                assert isinstance(measure, FunctionalSimilarityMeasure)
                                sim = measure(vals_a, vals_b)
                        runtime = time.perf_counter() - start_time
                        storer.add_results(obj_src, obj_tgt, measure, sim, runtime, overwrite)
                        logger.debug(
                            f"{measure.name}: Similarity '{sim:.04f}' in {time.perf_counter() - start_time:.1f}s."
                        )

                    except Exception as e:
                        storer.add_results(obj_src, obj_tgt, measure, metric_value=np.nan, runtime=np.nan)

                        logger.error(f"'{measure.name}' comparison failed.")
                        logger.error(e)

                    if measure.is_symmetric:
                        pbar.update(1)
                    pbar.update(1)

                # TODO: should be able to be removed without OOM, because self.rep_cache keeps reps more efficiently
                # than before
                storer.save_to_file()
                if (
                    self.cache_to_disk
                    and isinstance(obj_src, SingleLayerRepresentation)
                    and isinstance(obj_tgt, SingleLayerRepresentation)
                ):
                    obj_src.representation = None  # Clear memory
                    obj_tgt.representation = None  # Clear memory
