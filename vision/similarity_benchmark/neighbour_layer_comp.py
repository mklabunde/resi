import numpy as np
from scipy.stats import spearmanr
from torch import nn
from tqdm import tqdm
from vision.arch.abstract_acti_extr import AbsActiExtrArch
from vision.data.base_datamodule import BaseDataModule
from vision.util.vision_rep_extraction import extract_representations


# ToDo: Wrap the metric result into a more general result dataclass object
def layerwise_forward_sim(sim: np.ndarray) -> float:
    """Calculate the spearman rank correlation of the similarity to the layers"""
    aranged_1 = np.arange(sim.shape[0])[:, None]
    aranged_2 = np.arange(sim.shape[0])[None, :]
    dist = np.abs(aranged_1 - aranged_2)

    forward_corrs = []
    backward_corrs = []
    for i in range(sim.shape[0]):
        current_line_sim = sim[i]
        current_line_dist = dist[i]
        forward_sims = current_line_sim[i:]
        backward_sims = current_line_sim[:i]
        forward_dists = current_line_dist[i:]
        backward_dists = current_line_dist[:i]
        if len(forward_sims) > 1:
            corr, _ = spearmanr(forward_sims, forward_dists)
            forward_corrs.append(corr)
        if len(backward_sims) > 1:
            corr, _ = spearmanr(backward_sims, backward_dists)
            backward_corrs.append(corr)

    return np.nanmean(forward_corrs + backward_corrs)


def compare_models_layer_to_neighbours(
    model: AbsActiExtrArch, datamodule: BaseDataModule, metrics: dict[str, callable]
):
    model.eval()

    test_dataloader = datamodule.test_dataloader(batch_size=100)
    reps = extract_representations(model, test_dataloader, rel_reps=None, meta_info=True, remain_spatial=True)
    extracted_reps = reps["reps"]

    # Compare the representations
    all_results = {}
    for metric_name, metric in metrics.items():
        sim = np.zeros((len(extracted_reps), len(extracted_reps)))
        for cnt_a, rep_a in tqdm(enumerate(extracted_reps.keys()), total=len(extracted_reps)):
            for cnt_b, rep_b in tqdm(enumerate(extracted_reps.keys()), total=len(extracted_reps), leave=False):
                if cnt_a <= cnt_b:
                    # Compute the distance
                    sim[cnt_a, cnt_b] = metric(extracted_reps[rep_a], extracted_reps[rep_b])
                    sim[cnt_b, cnt_a] = sim[cnt_a, cnt_b]  # is symmetric (at least it should be)
        corr = layerwise_forward_sim(sim)
        all_results[metric_name] = corr
    return all_results


"""
Problems:
   - Similarity after pooling increases (generally)
   - PWCCA and SVCCA do not work trivially for spatial data (when increasing in size)

"""
