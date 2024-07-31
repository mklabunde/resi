from __future__ import annotations

import numpy as np
import torch
import torch as t
import vision.util.data_structs as ds
from vision.metrics.utils import mean_upper_triangular


def error_ratios(y_hats_1: torch.Tensor, y_hats_2: torch.Tensor, groundtruth: torch.Tensor) -> float:
    """Calculates the ratio between two models making the same mistake vs them making different mistakes
    :param y_hats_1: Most probable class of model 1  # N_SAMPLES
    :param y_hats_2: Most probable class of model 2  # N_SAMPLES
    :param groundtruth: Actual true class
    """

    errors_1 = y_hats_1 != groundtruth
    errors_2 = y_hats_2 != groundtruth

    both_wrong = errors_1 & errors_2
    n_both_wrong = torch.sum(both_wrong, dtype=torch.float32)
    n_both_wrong_same_way = torch.sum((y_hats_1 == y_hats_2)[both_wrong], dtype=torch.float32)
    n_both_wrong_different = n_both_wrong - n_both_wrong_same_way

    error_ratio = float(n_both_wrong_different / n_both_wrong_same_way)

    return error_ratio


def calculate_error_ratios(all_y_hats: list[t.Tensor], groundtruth: t.Tensor) -> ds.GroupMetrics:
    """
    Calculates the error ratios between all models. This leads to a NxN matrix.
    Additionally provides error ratios for various combinations of models.

    """
    err_ratios: np.ndarray = np.eye(N=len(all_y_hats), M=len(all_y_hats))
    for i in range(len(all_y_hats)):
        for j in range(len(all_y_hats)):
            if i == j:
                continue
            if i > j:
                err_ratios[i][j] = err_ratios[j][i]
            else:
                err_ratios[i][j] = error_ratios(all_y_hats[i], all_y_hats[j], groundtruth)
    ensemble_mean_ck = mean_upper_triangular(err_ratios)
    ck_of_new_model_to_existing: np.ndarray = err_ratios[-1, :-1]  # CK of new model to all other models
    mean_ck_new = float(np.mean(ck_of_new_model_to_existing))
    ck_of_new_model_to_first_model: float = float(err_ratios[0, -1])

    return ds.GroupMetrics(
        all_to_all=err_ratios.tolist(),
        all_to_all_mean=ensemble_mean_ck,
        last_to_others=ck_of_new_model_to_existing.tolist(),
        last_to_others_mean=mean_ck_new,
        last_to_first=ck_of_new_model_to_first_model,
    )
