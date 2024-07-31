import torch


def relative_ensemble_performance(
    all_predictions: torch.Tensor, ensemble_predictions: torch.Tensor, groundtruth: torch.Tensor
):
    """
    Relative rate of ensemble performance gain w.r.t the average ensemble performance
    :param all_predictions: Predictions of classes shape of [NModels x NSamples]
    :param groundtruth: Groundtruths in shape (not one-hot) [NSamples]
    :param ensemble_predictions: Predictions of classes shape of [NSamples]
    """

    mean_single_acc = torch.mean(all_predictions == groundtruth[None, :], dtype=torch.float)
    ensemble_acc = torch.mean(ensemble_predictions == groundtruth[None, :], dtype=torch.float)
    rel_ens_perf = ensemble_acc / mean_single_acc
    return rel_ens_perf
