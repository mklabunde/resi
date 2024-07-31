import numpy as np
import torch
from vision.metrics.utils import mean_upper_triangular
from vision.util import data_structs as ds


def jensen_shannon_distance(probs_1, probs_2, aggregated=True):
    """Calcualtes bounded JSD between two predictions."""
    total_m = 0.5 * (probs_1 + probs_2)
    if aggregated:
        kl_p1_m = torch.mean(torch.sum(probs_1 * torch.log2(probs_1 / total_m), dim=-1), dim=0)
        kl_p2_m = torch.mean(torch.sum(probs_2 * torch.log2(probs_2 / total_m), dim=-1), dim=0)
    else:
        kl_p1_m = torch.sum(probs_1 * torch.log2(probs_1 / total_m), dim=-1)
        kl_p2_m = torch.sum(probs_2 * torch.log2(probs_2 / total_m), dim=-1)
    jsd = (kl_p1_m + kl_p2_m) / 2
    return jsd


def jensen_shannon_divergences(all_pred_probs: list[torch.Tensor]) -> ds.GroupMetrics:
    """
    Calculates the error ratios between all models. This leads to a NxN matrix.
    Additionally provides error ratios for various combinations of models.

    """
    jsds: np.ndarray = np.zeros((len(all_pred_probs), len(all_pred_probs)))
    for i in range(len(all_pred_probs)):
        for j in range(len(all_pred_probs)):
            if i == j:
                continue
            if i > j:
                jsds[i][j] = jsds[j][i]
            else:
                jsds[i][j] = jensen_shannon_distance(all_pred_probs[i], all_pred_probs[j]).cpu().numpy()
    ensemble_mean_jsd = mean_upper_triangular(jsds)
    jsd_of_new_model_to_existing: np.ndarray = jsds[-1, :-1]  # CK of new model to all other models
    mean_jsd_new = float(np.mean(jsd_of_new_model_to_existing))
    jsd_of_new_model_to_first_model: float = float(jsds[0, -1])

    return ds.GroupMetrics(
        all_to_all=jsds.tolist(),
        all_to_all_mean=ensemble_mean_jsd,
        last_to_others=jsd_of_new_model_to_existing.tolist(),
        last_to_others_mean=mean_jsd_new,
        last_to_first=jsd_of_new_model_to_first_model,
    )
