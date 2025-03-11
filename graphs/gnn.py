import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from graphs.config import OPTIMIZER_PARAMS_DECAY_KEY
from graphs.config import OPTIMIZER_PARAMS_EPOCHS_KEY
from graphs.config import OPTIMIZER_PARAMS_LR_KEY
from graphs.config import SPLIT_IDX_TEST_KEY
from graphs.config import SPLIT_IDX_TRAIN_KEY
from graphs.config import SPLIT_IDX_VAL_KEY
from torch.nn import functional as func
from torch_geometric.utils import dropout_edge
from torcheval.metrics.functional import multiclass_accuracy
from tqdm import tqdm


def train_model(
    model,
    data,
    edge_index,
    split_idx,
    device,
    seed: int,
    optimizer_params: Dict,
    p_drop_edge: float,
    save_path: Path,
    b_test: bool = False,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = model.to(device)
    data = data.to(device)

    # TODO: maybe rearrange this, data passing is currently quite messy
    train_idx = split_idx[SPLIT_IDX_TRAIN_KEY].to(device)
    val_idx = split_idx[SPLIT_IDX_VAL_KEY]

    edge_index = edge_index.to(device)
    model.reset_parameters()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_params[OPTIMIZER_PARAMS_LR_KEY],
        weight_decay=optimizer_params[OPTIMIZER_PARAMS_DECAY_KEY],
    )

    results = []
    for epoch in tqdm(range(1, 1 + optimizer_params[OPTIMIZER_PARAMS_EPOCHS_KEY])):

        if p_drop_edge > 0:
            loss = train_epoch_dropout(
                model=model,
                x=data.x,
                edge_index=edge_index,
                y=data.y,
                train_idx=train_idx,
                p_drop_edge=p_drop_edge,
                optimizer=optimizer,
            )
        else:
            loss = train_epoch(
                model=model, x=data.x, edge_index=edge_index, y=data.y, train_idx=train_idx, optimizer=optimizer
            )

        train_acc, val_acc = validate(model, data, train_idx, val_idx)

        epoch_res = [epoch, loss, train_acc, val_acc]

        if b_test:
            test_acc = test(model, data, test_idx=split_idx[SPLIT_IDX_TEST_KEY])
            epoch_res.append(test_acc)

        results.append(epoch_res)

    torch.save(model.state_dict(), save_path)

    if b_test:
        return results, test(model, data, split_idx[SPLIT_IDX_TEST_KEY])

    return results


def train_epoch(model, x, edge_index, y, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)[train_idx]
    loss = func.cross_entropy(out, y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


def train_epoch_dropout(model, x, edge_index, y, train_idx, p_drop_edge, optimizer):
    model.train()
    optimizer.zero_grad()

    if p_drop_edge > 0:
        curr_adj, _ = dropout_edge(edge_index, p=p_drop_edge)
        out = model(x, curr_adj)[train_idx]
    else:
        out = model(x, edge_index)[train_idx]
    # loss = func.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss = func.cross_entropy(out, y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def validate(model, data, train_idx, val_idx):
    model.eval()

    out_train = model(data.x, data.adj_t)[train_idx]
    train_pred = out_train.argmax(dim=-1, keepdim=True)

    out_val = model(data.x, data.adj_t)[val_idx]
    val_pred = out_val.argmax(dim=-1, keepdim=True)

    train_acc, val_acc = (
        multiclass_accuracy(train_pred.squeeze(1), data.y[train_idx].squeeze(1)).detach().cpu().numpy(),
        multiclass_accuracy(val_pred.squeeze(1), data.y[val_idx].squeeze(1)).detach().cpu().numpy(),
    )

    return train_acc, val_acc


@torch.no_grad()
def test(model, data, test_idx):
    model.eval()

    out = model(data.x, data.adj_t)[test_idx]
    pred = out.argmax(dim=-1, keepdim=True)

    return multiclass_accuracy(pred.squeeze(1), data.y.squeeze(1)[test_idx]).detach().cpu().numpy()


@torch.no_grad()
def get_representations(model, data, device, test_idx, layer_ids):
    model = model.to(device)
    data = data.to(device)
    test_idx = test_idx.to(device)

    model.eval()

    activations = {}

    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    hooks = dict()
    for i in layer_ids:
        hooks[i] = model.convs[i].register_forward_hook(getActivation(f"layer{i + 1}"))

    _ = model(data.x, data.adj_t)

    for i in layer_ids:
        hooks[i].remove()

    reps = dict()
    for i in layer_ids:
        reps[i] = activations[f"layer{i + 1}"].detach()[test_idx].cpu().numpy()

    return reps


@torch.no_grad()
def get_test_output(model, data, device, test_idx, return_accuracy=False):
    model = model.to(device)
    data = data.to(device)
    test_idx = test_idx.to(device)

    model.eval()
    out = model(data.x, data.adj_t)[test_idx]

    if return_accuracy:
        pred = out.argmax(dim=-1, keepdim=True)
        acc = multiclass_accuracy(pred.squeeze(1), data.y.squeeze(1)[test_idx]).detach().cpu().numpy()
        return out, float(acc)

    return out
