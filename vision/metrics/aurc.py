from __future__ import annotations

import torch as t


def avg_adjacent_elements(t: t.Tensor) -> t.Tensor:
    t1 = t.narrow(0, 0, t.shape[0] - 1)
    t2 = t.narrow(0, 1, t.shape[0] - 1)
    return (t1 + t2) / 2


def calculate_area_under_selective_risk(selective_risks: t.Tensor) -> t.Tensor:
    return t.mean(avg_adjacent_elements(selective_risks))


def parallel_aurc(residuals: t.Tensor, confidence: t.Tensor) -> float:
    sorted_residuals = residuals[t.argsort(confidence, descending=True)]
    selective_risk = t.cumsum(sorted_residuals, dim=0) / t.arange(1, len(residuals) + 1, device=residuals.device)
    p_aurc = calculate_area_under_selective_risk(selective_risk)
    return float(p_aurc)


def aurc(residuals: t.Tensor, confidence: t.Tensor) -> float:
    """
    Calculate the Area-under-the-risc-coverage curve.

    :param residuals: Represents the residuals (For classification this would be
     1 for wrong predictions and 0 for correct)
    :param confidence: The confidence of that prediction
    """
    risks = []
    n = float(residuals.shape[0])
    idx_sorted = t.argsort(confidence)
    cov = n
    error_sum = t.sum(residuals[idx_sorted])
    risks.append(error_sum / n)
    weights = []
    tmp_weight = 0

    for i in range(0, len(idx_sorted) - 1):
        cov = cov - 1
        error_sum = error_sum - residuals[idx_sorted[i]]
        selective_risk = error_sum / (n - 1 - i)
        tmp_weight += 1

        if i == 0 or confidence[idx_sorted[i]] != confidence[idx_sorted[i - 1]]:
            risks.append(selective_risk)
            weights.append(tmp_weight / n)
            tmp_weight = 0

    if tmp_weight > 0:
        risks.append(risks[-1])
        weights.append(tmp_weight / n)

    aurc = float(
        t.sum(t.stack([(risks[i] + risks[i + 1]) * 0.5 * weights[i] for i in range(len(weights))])).detach().cpu()
    )

    return aurc
