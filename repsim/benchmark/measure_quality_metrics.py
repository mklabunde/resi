from itertools import product

import numpy as np
from loguru import logger
from sklearn.metrics import average_precision_score


def violation_rate(intra_group: list[float], cross_group: list[float], larger_is_more_similar: bool) -> float:
    """
    Calculate the rate of violations, that the intra-group similarity is less than the cross-group similarity.
    Args:
        intra_group: List of intra-group similarities.
        cross_group: List of cross-group similarities.
    """
    intra_group = intra_group.copy()
    cross_group = cross_group.copy()
    if not larger_is_more_similar:
        max_val = max(intra_group + cross_group)
        intra_group = [float(-sim + max_val) for sim in intra_group]
        cross_group = [float(-sim + max_val) for sim in cross_group]
    violations = sum([in_sim <= cross_sim for in_sim, cross_sim in product(intra_group, cross_group)])
    adherence = sum([in_sim > cross_sim for in_sim, cross_sim in product(intra_group, cross_group)])
    if (violations + adherence) == 0:
        return np.nan
    else:
        violation_rate = violations / (violations + adherence)
    return violation_rate


def auprc(intra_group: list[float], cross_group: list[float], larger_is_more_similar: bool) -> float:
    """
    Calculate the area under the precision-recall curve.
    Args:
        intra_group: List of intra-group similarities.
        cross_group: List of cross-group similarities.
    """
    intra_group = intra_group.copy()
    cross_group = cross_group.copy()
    if not larger_is_more_similar:
        max_val = max(np.max(intra_group), np.max(cross_group))
        if max_val == 0:  # avoid division by zero
            max_val = 5e-5
        intra_group = [float(-sim + max_val) / max_val for sim in intra_group]
        cross_group = [float(-sim + max_val) / max_val for sim in cross_group]

    in_group_sims = np.array(intra_group)
    cross_group_sims = np.array(cross_group)
    y_true = np.concatenate([np.ones_like(in_group_sims), np.zeros_like(cross_group_sims)])
    y_score = np.concatenate([in_group_sims, cross_group_sims])

    # Use partial data if some of the comparisons failed and gave nans instead of failing the whole AUPRC computation
    use_comparison = (~np.isnan(y_score)) & (~np.isnan(y_true))
    if (~use_comparison).sum() > 0:
        logger.warning(
            f"{(~use_comparison).sum()} comparisons between models resulted in NaNs. Using non-NaN {use_comparison.sum()} comparisons."
        )
    if use_comparison.sum() == 0:
        return np.nan
    else:
        auprc = average_precision_score(
            y_true[use_comparison], y_score[use_comparison]
        )  # 1 for perfect separation, 0.5 for random, 0 for inverse separation (inverted metric)

    # if any(np.isnan(y_score)) or any(np.isnan(y_true)):
    #     return np.nan
    # else:
    #     auprc = average_precision_score(
    #         y_true, y_score
    #     )  # 1 for perfect separation, 0.5 for random, 0 for inverse separation (inverted metric)
    return float(auprc)
