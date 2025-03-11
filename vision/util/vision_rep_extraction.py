from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from repsim.utils import ModelRepresentations
else:
    ModelRepresentations = None

from vision.util import data_structs as ds
from vision.arch.abstract_acti_extr import AbsActiExtrArch
from vision.arch.arch_loading import load_model_from_info_file

from vision.util.file_io import get_vision_model_info
from vision.util import find_datamodules as fd


def get_single_layer_vision_representation_on_demand(
    architecture_name: str,
    train_dataset: str,
    seed: int,
    setting_identifier: str | None,
    representation_dataset: str,
    layer_id: int,
) -> ModelRepresentations:
    """Creates Model Representations with representations that can be extracted only when needed)"""
    model_info: ds.ModelInfo = get_vision_model_info(
        architecture_name=architecture_name,
        dataset=train_dataset,
        seed_id=seed,
        setting_identifier=setting_identifier,
    )
    # ---------- Create the on-demand-callable functions for each layer ---------- #
    """Function providing the representations for a single layer on demand."""
    loaded_model = load_model_from_info_file(model_info, load_ckpt=True)
    is_vit = True if architecture_name in ["ViT_B32", "ViT_L32"] else False
    datamodule = fd.get_datamodule(dataset=representation_dataset, is_vit=is_vit)
    test_dataloader = datamodule.test_dataloader(batch_size=100)
    res = extract_single_layer_representations(
        layer_id, loaded_model, test_dataloader, None, meta_info=True, remain_spatial=True
    )
    reps = res["reps"]
    return reps


def get_vision_output_on_demand(
    architecture_name: str,
    train_dataset: str,
    seed: int,
    setting_identifier: str | None,
    representation_dataset: str,
) -> np.ndarray:
    """Creates Model Representations with representations that can be extracted only when needed)"""
    model_info: ds.ModelInfo = get_vision_model_info(
        architecture_name=architecture_name,
        dataset=train_dataset,
        seed_id=seed,
        setting_identifier=setting_identifier,
    )
    # ---------- Create the on-demand-callable functions for each layer ---------- #
    """Function providing the representations for a single layer on demand."""
    loaded_model = load_model_from_info_file(model_info, load_ckpt=True)
    datamodule = fd.get_datamodule(dataset=representation_dataset)
    test_dataloader = datamodule.test_dataloader(batch_size=100)
    res = extract_single_layer_representations(
        -1, loaded_model, test_dataloader, None, meta_info=True, remain_spatial=True
    )  # We do extract reps, but discard them and only use the logits instead.
    if isinstance(res["logits"], torch.Tensor):
        logits = res["logits"].detach().cpu().numpy()
    else:
        logits = res["logits"]
    return logits


def extract_single_layer_representations(
    layer_id: int,
    model: AbsActiExtrArch,
    dataloader: torch.utils.data.DataLoader,
    rel_reps: dict[str, torch.Tensor] = None,
    meta_info: bool = True,
    remain_spatial: bool = False,
) -> dict[str, torch.Tensor]:
    """Extracts the anchor representations from the model."""
    reps: list[torch.Tensor] = []
    handle: torch.utils.hooks.RemovableHandle
    if rel_reps is None:
        handle = model.register_parallel_rep_hooks(model.hooks[layer_id], reps)
    else:
        raise NotImplementedError("Relative representations not used here.")

    logits = []
    probs = []
    labels = []
    pred_cls = []
    model.eval()
    model.cuda()
    with torch.no_grad():
        for cnt, batch in enumerate(dataloader):
            im, lbl = batch[0], batch[1]
            y_logit = model(im.cuda())
            y_probs = torch.softmax(y_logit, dim=1)
            y_hat = torch.argmax(y_probs, dim=1)
            logits.append(y_logit.cpu())
            probs.append(y_probs.cpu())
            labels.append(lbl.cpu())
            pred_cls.append(y_hat.cpu())

    handle.remove()

    reps = torch.cat(reps, dim=0)
    if not remain_spatial:
        reps = torch.reshape(reps, (reps.shape[0], -1))  # Flatten the into Samples x Features

    # If we use high dimensional representations, we might run out of memory.
    # Hence we downsample them to a maximum spatial extent of 8x8.

    if len(reps.shape) == 4 and reps.shape[2] > 7:
        reps = torch.nn.functional.adaptive_avg_pool2d(reps, (7, 7))

    if len(reps.shape) == 3:
        reps = reps[:, 0, :]  # Just take cls token, resulting in nd shape.

    out = {"reps": reps}
    if meta_info:
        out["logits"] = torch.cat(logits, dim=0)
        out["probs"] = torch.cat(probs, dim=0)
        out["y_hat"] = torch.cat(pred_cls, dim=0)
        out["gt"] = torch.cat(labels, dim=0)
    return out


# def extract_representations(
#     model: AbsActiExtrArch,
#     dataloader: torch.utils.data.DataLoader,
#     rel_reps: dict[str, torch.Tensor] = None,
#     meta_info: bool = True,
#     remain_spatial: bool = False,
# ) -> dict[str, torch.Tensor]:
#     """Extracts the anchor representations from the model."""
#     reps: dict[str, list[torch.Tensor]] = {}
#     handles: list[torch.utils.hooks.RemovableHandle] = []
#     if rel_reps is None:
#         for cnt, hook in enumerate(model.hooks):
#             reps[str(cnt)] = []
#             handles.append(model.register_parallel_rep_hooks(hook, reps[str(cnt)], remain_spatial))
#     else:
#         for cnt, hook in enumerate(model.hooks):
#             reps[str(cnt)] = []
#             handles.append(model.register_relative_rep_hooks(hook, rel_reps[str(cnt)], reps[str(cnt)]))

#     logits = []
#     probs = []
#     labels = []
#     pred_cls = []
#     model.eval()
#     model.cuda()
#     with torch.no_grad():
#         for cnt, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
#             if cnt > 50:
#                 continue
#             im, lbl = batch
#             y_logit = model(im.cuda())
#             y_probs = torch.softmax(y_logit, dim=1)
#             y_hat = torch.argmax(y_probs, dim=1)
#             logits.append(y_logit.cpu())
#             probs.append(y_probs.cpu())
#             labels.append(lbl.cpu())
#             pred_cls.append(y_hat.cpu())

#     for handle in handles:
#         handle.remove()

#     for cnt, rep in reps.items():
#         tmp_reps = torch.cat(rep, dim=0)
#         if remain_spatial:
#             reps[cnt] = tmp_reps
#         else:
#             reps[cnt] = torch.reshape(tmp_reps, (tmp_reps.shape[0], -1))  # Flatten the into Samples x Features
#     out_reps = {"reps": reps}
#     if meta_info:
#         out_reps["logits"] = torch.cat(logits, dim=0)
#         out_reps["probs"] = torch.cat(probs, dim=0)
#         out_reps["y_hat"] = torch.cat(pred_cls, dim=0)
#         out_reps["gt"] = torch.cat(labels, dim=0)
#     return out_reps
