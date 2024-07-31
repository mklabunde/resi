from __future__ import annotations

from collections.abc import Iterable

import torch
import torch as t
from torch.nn import functional as F


def pseudo_inversion(probs: torch.Tensor) -> torch.Tensor:
    fill_value = 1.0 / (probs.shape[1] - 1)
    reshuffle_matrix = fill_value * (
        1 - torch.eye(probs.shape[1], probs.shape[1], device=probs.device, dtype=probs.dtype)
    )
    reshuffled_probs = probs @ reshuffle_matrix
    return reshuffled_probs


def assert_shapes_match(values_a: list[t.Tensor], values_b: list[t.Tensor]):
    for cnt, (val_a, val_b) in enumerate(zip(values_a, values_b)):
        same_n_models = val_a.shape[0] == val_b.shape[0]
        is_broadcastable = (val_a.shape[0] == 1) | (val_b.shape[0] == 1)
        assert same_n_models or is_broadcastable, (
            f"Number of model shapes do not match in id {cnt} "
            f"of Hooks given!\nGot: {val_a.shape[0]} {val_b.shape[0]}"
        )
    return


def celu_explained_variance(true: list[t.Tensor], approximations: list[t.Tensor]) -> list[t.Tensor]:
    """Can't be used when LinCKA is the metric, as no alignment is created and Channel dimensions do not match!"""
    all_cevs: list[t.Tensor] = []
    assert_shapes_match(true, approximations)
    for tr, apxs in zip(true, approximations):
        if tr.shape[2] != apxs.shape[2]:
            all_cevs.append(torch.tensor([torch.nan], device=tr.device, dtype=tr.dtype))
        else:
            mean_error = t.sum((tr - t.mean(tr, dim=(1, 3, 4), keepdim=True)) ** 2, dim=(1, 3, 4))
            cev = F.celu(1.0 - ((t.sum((tr - apxs) ** 2, dim=(1, 3, 4))) / (mean_error + 1e-4)), inplace=True)
            all_cevs.append(cev)
    t.cuda.empty_cache()
    return all_cevs


def cosine_similarity(apxs: list[t.Tensor]) -> list[list[t.Tensor]]:
    """Calculates the cosine similarity for approximated representations."""
    cs = []
    for apx in apxs:
        apxs_cos = []
        for a in apx:
            batch_vectorized_a = t.reshape(a, [a.shape[0], -1])
            normed_a = batch_vectorized_a / t.norm(batch_vectorized_a, p="fro", dim=1, keepdim=True)
            apx_cos_sim = normed_a @ normed_a.T
            # normed_a = batch_vectorized_a / t.norm(batch_vectorized_a, p="fro", dim=1, keepdim=True)
            apxs_cos.append(apx_cos_sim)
        cs.append(apxs_cos)
    return cs


def euclidean_distance_csim(cosine_values: Iterable[tuple[list[t.Tensor], list[t.Tensor]]]) -> torch.Tensor:
    """
    Calculates the euclidean distance between the relative representations of the true and the approximated models.
    High values means different
    Low values mean similar
    """
    all_layer_results = []
    for cvs in cosine_values:
        layerwise_results = []
        for cos_sim_a, cos_sim_b in zip(cvs[0], cvs[1]):
            layerwise_results.append(torch.mean((cos_sim_a - cos_sim_b) ** 2))
        layerwise_disagreement = torch.mean(torch.stack(layerwise_results))
        all_layer_results.append(layerwise_disagreement)
    all_layer_results = torch.mean(torch.stack(all_layer_results))
    return all_layer_results


def dot_product_csim(cosine_values: list[tuple[t.Tensor, list[t.Tensor]]]):
    """
    Calculates the cosine similarity between the relative representations of the true and the approximated models.
    When matrix values are high the
    """
    all_layer_results = []
    for cvs in cosine_values:
        true_cos_sim = cvs[0]
        layerwise_results = []
        for apx_cos_sim in cvs[1]:
            layerwise_results.append(torch.mean((true_cos_sim * apx_cos_sim) ** 2))
        layerwise_disagreement = torch.mean(torch.stack(layerwise_results))
        all_layer_results.append(layerwise_disagreement)
    all_layer_results = torch.mean(torch.stack(all_layer_results))
    return all_layer_results


def topk_celu_explained_variance(
    true: list[t.Tensor], approximations: list[t.Tensor], k: int = 1000
) -> list[list[t.Tensor]]:
    r2s: list[list[t.Tensor]] = []
    assert_shapes_match(true, approximations)
    for trs, apxs in zip(true, approximations):
        if trs.shape[0] == apxs.shape[0]:
            pass
        elif trs.shape[0] < apxs.shape[0]:
            trs = torch.repeat_interleave(trs, repeats=apxs.shape[0], dim=0)
        else:
            apxs = torch.repeat_interleave(apxs, repeats=trs.shape[0], dim=0)

        modelwise_topk_cev: list[t.Tensor] = []
        for tr, a in zip(trs, apxs):
            # Get TOPK true values
            centered_tr = tr - torch.mean(tr, dim=(0, 2, 3), keepdim=True)
            centered_tr = torch.reshape(centered_tr, (centered_tr.shape[0], -1))
            flat_tr = torch.reshape(centered_tr, (-1,))

            _, tr_topk_indices = torch.topk(torch.abs(centered_tr), k=k, dim=-1)
            mask_tr = torch.zeros_like(centered_tr)
            mask_tr[torch.unsqueeze(torch.arange(mask_tr.shape[0]), -1), tr_topk_indices] = 1
            flat_mask_tr = torch.reshape(mask_tr, (-1,))

            centered_a = a - torch.mean(a, dim=(0, 2, 3), keepdim=True)  # Average for channels
            centered_a = torch.reshape(centered_a, (centered_a.shape[0], -1))
            flat_apx = torch.reshape(centered_a, (-1,))
            # Use centered values

            # Find TOPK indices
            _, a_topk_indices = torch.topk(torch.abs(centered_a), k=k, dim=-1)
            # Create empty mask
            mask_a = torch.zeros_like(centered_a)
            # Insert values where approx value is highest for each **sample** k values!
            mask_a[torch.unsqueeze(torch.arange(mask_a.shape[0]), -1), a_topk_indices] = 1
            flat_mask_a = torch.reshape(mask_a, (-1,))
            # Flatten to make it per sample!
            # Determine the joint mask of important foreground values
            joint_mask = torch.logical_or(flat_mask_tr, flat_mask_a)
            flat_joint_mask = torch.reshape(joint_mask, shape=(-1,))

            # Extract the values which will be used to determine the explained variance
            #   between [k, 2k] values maximum depending on overlap!
            fltr_tr = flat_tr[flat_joint_mask]
            fltr_apx = flat_apx[flat_joint_mask]

            # Calculate the approx error and try to minimize exp variance for them.
            approx_error = torch.sum((fltr_apx - fltr_tr) ** 2)
            mean_error = torch.sum(fltr_tr**2)  # Is zero-centered so mean = 0!
            r2 = torch.celu(1 - (approx_error / (mean_error + 1e-4)))
            modelwise_topk_cev.append(r2)
        r2s.append(modelwise_topk_cev)
    return r2s


def correlation(values_a: list[t.Tensor], values_b: list[t.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Expects a list of tensors where each list entry has
    true and approximation with (N_Models x Samples x Channels x w x h ) shape.
    The Number of models between true and approximated models can differ. Should they differ
    true or approx can be broadcast accordingly.

    """
    tmp_corrs = []
    tmp_stds = []
    assert_shapes_match(values_a, values_b)
    for tr, apps in zip(values_a, values_b):
        # Flat and normalize true values
        tt = t.transpose(tr, dim0=2, dim1=1)
        tt_flat = t.reshape(tt, (tt.shape[0], tt.shape[1], -1))  # Flatten the rest
        tt_means = t.mean(tt_flat, dim=(2,), keepdim=True)
        tt_flat_cent = tt_flat - tt_means
        tt_std = t.std(tt_flat_cent, dim=2, keepdim=True)

        ta = t.transpose(apps, dim0=2, dim1=1)
        ta_flat = t.reshape(ta, (ta.shape[0], ta.shape[1], -1))  # N_Approx x Channels x Values
        ta_means = t.mean(ta_flat, dim=(2,), keepdim=True)
        ta_std = t.std(ta_flat, dim=(2,), keepdim=True)
        ta_flat_cent = ta_flat - ta_means

        corrs = torch.sum((tt_flat_cent * ta_flat_cent) / ((ta_std * tt_std) + 1e-7), dim=-1) / (
            tt_flat_cent.shape[-1] - 1
        )

        tmp_corrs.append(corrs)
        tmp_stds.append(tt_std)
    t.cuda.empty_cache()
    return tmp_corrs, tmp_stds


def topk_correlation(true: list[t.Tensor], approximations: list[t.Tensor], k: int = 1000) -> list[list[t.Tensor]]:
    """
    Takes the top k values of the true and approximated samples across all channels and correlates them.
    1. Get Indices of the top k true values
    2. Get Indices of top k approx. values
    3. Logical or the location of these values
    4. Filter values for true and approximated values
    5. Calculate correlation between these values only
    - Does ignore channels; Normalization across samples & channels,
    """
    corrs: list[list[t.Tensor]] = []
    assert_shapes_match(true, approximations)
    for trs, apxs in zip(true, approximations):
        # Focus is still on the topk values with the highest amplitude but the normalized values
        #   are still needed later for correlation calculation!
        if trs.shape[0] < apxs.shape[0]:
            trs = torch.repeat_interleave(trs, repeats=apxs.shape[0], dim=0)
        else:
            apxs = torch.repeat_interleave(apxs, repeats=trs.shape[0], dim=0)
        hooks_modelwise_topk_corr = []
        for tr, a in zip(trs, apxs):
            # Get the training mask
            sample_wise_tr = torch.reshape(tr, (tr.shape[0], -1))
            flat_tr = torch.reshape(sample_wise_tr, (-1,))
            _, tr_topk_indices = torch.topk(torch.abs(sample_wise_tr), k=k, dim=-1)
            mask_tr = torch.zeros_like(sample_wise_tr)
            mask_tr[torch.unsqueeze(torch.arange(mask_tr.shape[0]), -1), tr_topk_indices] = 1
            flat_mask_tr = torch.reshape(mask_tr, (-1,))
            flat_mask_tr = flat_mask_tr.detach()

            # Get the approximation mask
            sample_wise_a = torch.reshape(a, (a.shape[0], -1))
            flat_apx = torch.reshape(a, (-1,))
            # Find TOPK indices
            _, a_topk_indices = torch.topk(torch.abs(sample_wise_a), k=k, dim=-1)
            # Create empty mask
            mask_a = torch.zeros_like(sample_wise_a)
            # Insert values where approx value is highest for each **sample** k values!
            mask_a[torch.unsqueeze(torch.arange(mask_a.shape[0]), -1), a_topk_indices] = 1
            flat_mask_a = torch.reshape(mask_a, (-1,))
            # Flatten to make it per sample!
            # Determine the joint mask of important foreground values
            joint_mask = torch.logical_or(flat_mask_tr, flat_mask_a)
            joint_mask = joint_mask.detach()

            # Extract the values which will be used to determine the explained variance
            #   between [k, 2k] values maximum depending on overlap!
            fltr_tr = flat_tr[joint_mask]
            fltr_ap = flat_apx[joint_mask]

            nrm_tr = fltr_tr - torch.mean(fltr_tr)
            nrm_ap = fltr_ap - torch.mean(fltr_ap)

            cor = torch.sum(
                (nrm_tr * nrm_ap) / ((torch.std(fltr_tr) * torch.std(fltr_ap) * (fltr_tr.shape[0] - 1)) + 1e-9)
            )

            # Calculate the approx error and try to minimize exp variance for them.
            hooks_modelwise_topk_corr.append(torch.abs(cor))
        corrs.append(hooks_modelwise_topk_corr)
    return corrs


def centered_kernel_alignment(true: list[t.Tensor], approximations: list[t.Tensor]) -> list[list[t.Tensor]]:
    """
    Calculates an unbiased HSIC between K and L with t.
    Paper:
    Similarity of Neural Network Representations Revisited - https://arxiv.org/abs/1905.00414
    (Previous Code inspired from):
    https://towardsdatascience.com/do-different-neural-networks-learn-the-same-things-ac215f2103c3
    """
    ckas = []
    assert_shapes_match(true, approximations)
    for trs, apxs in zip(true, approximations):
        # Focus is still on the topk values with the highest amplitude but the normalized values
        #   are still needed later for correlation calculation!
        if trs.shape[0] == apxs.shape[0]:
            pass
        elif trs.shape[0] < apxs.shape[0]:
            trs = torch.repeat_interleave(trs, repeats=apxs.shape[0], dim=0)
        else:
            apxs = torch.repeat_interleave(apxs, repeats=trs.shape[0], dim=0)
        models_ckas = []
        for tr, a in zip(trs, apxs):
            a_size = a.size()
            t_size = tr.size()

            # Since its about making the model learn something different w.r.t. the already trained model
            #   I decided to consider each channel a Neuron and take the spatial dimensions as new samples!
            #   In other papers each pixel in ch x w x h dimension is considered a neuron. Here this is not the case
            #   as for models with same architecture correspondence at spatial locations can be assumed.

            #  Batch x Ch x Width x Height --> (Batch * width * height) x Ch == Samples x Neuron
            a_trans = t.transpose(a, dim0=1, dim1=0)
            a_flat = t.reshape(a_trans, shape=(a_size[1], -1))
            ax = t.transpose(a_flat, dim0=1, dim1=0)

            t_trans = t.transpose(tr, dim0=1, dim1=0)
            t_flat = t.reshape(t_trans, shape=(t_size[1], -1))
            tr = t.transpose(t_flat, dim0=1, dim1=0)

            with t.autocast(enabled=False, device_type="cuda"):
                a_full = ax.type(t.float32)
                t_full = tr.type(t.float32)
                models_ckas.append(cka(a_full, t_full))
                # cka_batch_size = 2048
                # if a_flat.size()[0] >= cka_batch_size:  # OOM protection mechanism
                #     tmp_ckas = []
                #     remainder = bool(a_flat.size()[0] % cka_batch_size)
                #     iters = a_flat.size()[0] // cka_batch_size
                #     for i in range(iters):
                #         partial_a_flat = a_full[i * cka_batch_size : (i + 1) * cka_batch_size]
                #         partial_t_flat = t_full[i * cka_batch_size : (i + 1) * cka_batch_size]
                #         tmp_ckas.append(cka(partial_a_flat, partial_t_flat))
                #     if remainder:
                #         partial_a_flat = a_full[iters * cka_batch_size : -1]
                #         partial_t_flat = t_full[iters * cka_batch_size : -1]
                #         tmp_ckas.append(cka(partial_a_flat, partial_t_flat))
        ckas.append(models_ckas)

    return ckas


def cka(actis_a: t.Tensor, actis_b: t.Tensor):
    """
    Expects samples x Neurons.
     Can either be a batch_size or the whole (batch_size*width*height) first!
    """
    actis_a = actis_a - t.mean(actis_a, dim=0, keepdim=True)  # Mean along the neuron dimension
    actis_b = actis_b - t.mean(actis_b, dim=0, keepdim=True)

    nominator = t.norm(t.matmul(actis_b.T, actis_a), p="fro") ** 2
    denominator = t.norm(t.matmul(actis_a.T, actis_a), p="fro") * t.norm(t.matmul(actis_b.T, actis_b), p="fro")

    cka = nominator / (denominator + 1e-9)

    return cka


def unbiased_hsic(K: t.Tensor, L: t.Tensor):
    """
    Calculates an biased HSIC between K and L with t.
    Paper:
    Similarity of Neural Network Representations Revisited - https://arxiv.org/abs/1905.00414
    Code ported to t from:
    https://towardsdatascience.com/do-different-neural-networks-learn-the-same-things-ac215f2103c3
    """
    k_size = K.size()[0]
    l_size = L.size()[0]
    ones = t.ones(size=(k_size, 1), device=K.device, dtype=K.dtype)

    new_K = K * (1 - torch.eye(n=k_size, m=k_size, device=K.device, dtype=K.dtype))
    new_L = L * (1 - torch.eye(n=l_size, m=l_size, device=K.device, dtype=L.dtype))

    trace = t.trace(t.matmul(new_K, new_L))

    nominator1 = t.matmul(t.matmul(ones.T, new_K), ones).float()
    nominator2 = t.matmul(t.matmul(ones.T, new_L), ones).float()
    denominator = (k_size - 1) * (k_size - 2)
    middle = nominator1 * nominator2 / denominator

    multiplier1 = 2 / (k_size - 2)
    multiplier2 = t.matmul(t.matmul(ones.T, new_K), t.matmul(new_L, ones))
    last = multiplier1 * multiplier2

    ub_hsic = 1 / ((k_size * (k_size - 3)) * (trace + middle - last) + 1e-4)

    return ub_hsic
