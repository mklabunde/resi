import random

import numpy as np
import torch


def subsample_torch_mask(idx, size, seed):
    # sample from numpy random state for more safe reproducibility
    rng = np.random.RandomState(seed)
    numeric_idx = np.arange(len(idx))[idx.numpy()]

    sub_idx = torch.from_numpy(rng.choice(numeric_idx, size, replace=False))
    res_idx = torch.zeros(size=(len(idx),), dtype=torch.bool)
    res_idx[sub_idx] = True

    return res_idx


def subsample_torch_index(idx, size, seed):
    # sample from numpy random state for more safe reproducibility
    rng = np.random.RandomState(seed)

    return torch.from_numpy(rng.choice(idx.numpy(), size, replace=False))


def shuffle_labels(y, frac=0.5, seed=None):

    if seed is not None:
        random.seed(seed)

    is_tensor = torch.is_tensor(y)

    if is_tensor:
        y = y.numpy().flatten()

    n_instances = len(y)
    Y = list(np.unique(y))
    shuffle_idx = random.sample(list(range(n_instances)), k=int(frac * n_instances))
    for i in shuffle_idx:
        old_label = y[i]
        new_label = random.sample([label for label in Y if label != old_label], k=1)[0]
        y[i] = new_label

    if is_tensor:
        return torch.from_numpy(np.reshape(y, newshape=(n_instances, 1)))

    return y
