import os

from repsim.benchmark.group_separation_experiment import GroupSeparationExperiment
from repsim.benchmark.paths import EXPERIMENT_RESULTS_PATH
from repsim.benchmark.registry import ALL_TRAINED_MODELS
from repsim.benchmark.registry import TrainedModel
from repsim.measures import distance_correlation
from repsim.measures import gulp
from repsim.measures.cca import pwcca
from repsim.measures.cca import svcca
from repsim.measures.cka import centered_kernel_alignment
from repsim.measures.correlation_match import hard_correlation_match
from repsim.measures.correlation_match import soft_correlation_match
from repsim.measures.eigenspace_overlap import eigenspace_overlap_score
from repsim.measures.geometry_score import geometry_score
from repsim.measures.linear_regression import linear_reg
from repsim.measures.multiscale_intrinsic_distance import imd_score
from repsim.measures.nearest_neighbor import jaccard_similarity
from repsim.measures.nearest_neighbor import joint_rank_jaccard_similarity
from repsim.measures.nearest_neighbor import rank_similarity
from repsim.measures.nearest_neighbor import second_order_cosine_similarity
from repsim.measures.procrustes import aligned_cossim
from repsim.measures.procrustes import orthogonal_angular_shape_metric
from repsim.measures.procrustes import orthogonal_angular_shape_metric_centered
from repsim.measures.procrustes import orthogonal_procrustes
from repsim.measures.procrustes import orthogonal_procrustes_centered_and_normalized
from repsim.measures.procrustes import permutation_aligned_cossim
from repsim.measures.procrustes import permutation_angular_shape_metric
from repsim.measures.procrustes import permutation_procrustes
from repsim.measures.procrustes import procrustes_size_and_shape_distance
from repsim.measures.rsa import representational_similarity_analysis
from repsim.measures.rsm_norm_difference import rsm_norm_diff
from repsim.measures.statistics import concentricity_difference
from repsim.measures.statistics import concentricity_nrmse
from repsim.measures.statistics import magnitude_difference
from repsim.measures.statistics import magnitude_nrmse
from repsim.measures.statistics import uniformity_difference


if __name__ == "__main__":
    subset_of_vision_models = [
        m
        for m in ALL_TRAINED_MODELS
        if (m.domain == "VISION")
        and (m.architecture == "ResNet18")
        and (
            m.train_dataset
            in [
                "Gauss_Max_CIFAR10DataModule",
                "Gauss_S_CIFAR10DataModule",
            ]
        )
        and (m.additional_kwargs["seed_id"] <= 1)
    ]

    # experiment = SameLayerExperiment(subset_of_vision_models, [centered_kernel_alignment], "CIFAR10")
    def pseudo_split(models: list[TrainedModel]) -> tuple[list[TrainedModel]]:
        """Split the models into groups based on the dataset they were trained on."""
        model_train_ds = sorted(list(set([m.train_dataset for m in models])))
        model_groups = []
        for tr_ds in model_train_ds:
            group = [m for m in models if m.train_dataset == tr_ds]
            model_groups.append(group)
        return tuple(model_groups)

    experiment = GroupSeparationExperiment(
        experiment_identifier="runtime_benchmark_vision_models",
        models=subset_of_vision_models,
        group_splitting_func=pseudo_split,
        storage_path=os.path.join(EXPERIMENT_RESULTS_PATH, "runtime_experiments.parquet"),
        measures=[
            # cca
            svcca,
            pwcca,
            # cka
            centered_kernel_alignment,
            # correlation_match
            hard_correlation_match,
            soft_correlation_match,
            # distance_correlation
            distance_correlation,
            # eigenspace_overlap
            eigenspace_overlap_score,
            # geometry_score
            geometry_score,
            # gulp
            gulp,
            # linear regression
            linear_reg,
            # multiscale_intrinsic_distance
            imd_score,
            # nearest_neighbor
            jaccard_similarity,
            second_order_cosine_similarity,
            rank_similarity,
            joint_rank_jaccard_similarity,
            # Procrustes
            orthogonal_procrustes,
            procrustes_size_and_shape_distance,
            orthogonal_procrustes_centered_and_normalized,
            permutation_procrustes,
            permutation_angular_shape_metric,
            orthogonal_angular_shape_metric,
            orthogonal_angular_shape_metric_centered,
            aligned_cossim,
            permutation_aligned_cossim,
            # rsa
            representational_similarity_analysis,
            # rsm_norm_diff
            rsm_norm_diff,
            # statistics
            magnitude_difference,
            magnitude_nrmse,
            uniformity_difference,
            concentricity_difference,
            concentricity_nrmse,
        ],
        representation_dataset="CIFAR10",
    )
    result = experiment.run()
    print(result)
    print(0)
