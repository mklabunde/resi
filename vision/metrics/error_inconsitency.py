"""
Gontijo-Lopes, R., Dauphin, Y., & Cubuk, E. D. (n.d.).
 NO ONE REPRESENTATION TO RULE THEM ALL: OVERLAPPING FEATURES OF TRAINING METHODS.

In order to understand whether model similarity reported in literature varies
 as a function of training methodology, we evaluate error correlation by
 measuring the number of test-set examples where one model predicts the correct
 class, and the other predicts an incorrect class.
"""
import torch


def error_inconsistency(preds_a: torch.Tensor, preds_b: torch.Tensor, groundtruth: torch.Tensor) -> torch.Tensor:
    """
    Calculates observed error inconsistency

    'Observed error inconsistency' is the fraction of [the whole test set]
     examples where only one model in the pair makes a correct prediction'
    :param preds_a: Value of the predicted class e.g. CIFAR10: [0 ... 9] Size: [Batch]
    :param preds_b: Value of the predicted class e.g. CIFAR10: [0 ... 9] Size: [Batch]
    :param groundtruth: Value of the groundtruth class e.g. CIFAR10: [0 ... 9] Size: [Batch]
    """

    total_sample = float(preds_a.shape[0])

    predictions_a_correct = preds_a == groundtruth
    predictions_b_correct = preds_b == groundtruth

    a_xor_b_incorrect = torch.logical_xor(predictions_a_correct, predictions_b_correct)
    observed_error_inconsitency = torch.sum(a_xor_b_incorrect) / total_sample

    return observed_error_inconsitency


def cc_like_error_inconsitency(
    preds_a: torch.Tensor, preds_b: torch.Tensor, groundtruth: torch.Tensor
) -> torch.Tensor:
    """
    Calculates observed error inconsistency w.r.t. the expected error inconsistency
    More similar to cohens kappa as the estimated

    'Observed error inconsistency' is the fraction of [the whole test set]
     examples where only one model in the pair makes a correct prediction'
    :param preds_a: Value of the predicted class e.g. CIFAR10: [0 ... 9] Size: [Batch]
    :param preds_b: Value of the predicted class e.g. CIFAR10: [0 ... 9] Size: [Batch]
    :param groundtruth: Value of the groundtruth class e.g. CIFAR10: [0 ... 9] Size: [Batch]
    """

    total_sample = float(preds_a.shape[0])

    predictions_a_correct = preds_a == groundtruth
    predictions_b_correct = preds_b == groundtruth

    acc_a = predictions_a_correct / total_sample
    acc_b = predictions_b_correct / total_sample

    a_xor_b_incorrect = torch.logical_xor(predictions_a_correct, predictions_b_correct)
    observed_error_inconsitency = torch.sum(a_xor_b_incorrect) / total_sample
    expected_inconsistency = ((1 - acc_a) * acc_b) + ((1 - acc_b) * acc_a)

    inconsistency = (observed_error_inconsitency - expected_inconsistency) / (1.0 + 1e-7 - expected_inconsistency)

    return inconsistency
