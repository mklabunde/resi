from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytorch_lightning as pl
import torch
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # noqa
from torch.utils.tensorboard.writer import SummaryWriter
from vision.arch.abstract_acti_extr import AbsActiExtrArch
from vision.losses.dummy_loss import DummyLoss
from vision.metrics.ke_metrics import single_output_metrics
from vision.util import data_structs as ds
from vision.util.file_io import save_json


class BaseLightningModule(pl.LightningModule, ABC):

    def __init__(
        self,
        model_info: ds.ModelInfo,
        network: AbsActiExtrArch,
        save_checkpoints: bool,
        params: ds.Params,
        hparams: dict,
        loss: DummyLoss,
        log: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.net = network
        self.loss = loss
        self.mode_info: ds.ModelInfo = model_info
        self.params = params
        self.ke_hparams = hparams
        self.do_log = log

        if self.do_log:
            self.tb_logger_tr: SummaryWriter = SummaryWriter(log_dir=str(model_info.path_train_log / "train"))
            self.tb_logger_val: SummaryWriter = SummaryWriter(log_dir=str(model_info.path_train_log / "val"))

        self.checkpoint_path: Path = model_info.path_ckpt
        self.checkpoint_dir_path: Path = model_info.path_ckpt.parent
        self.save_checkpoints = save_checkpoints
        torch.backends.cudnn.benchmark = True  # noqa

        # For the final validation epoch we want to aggregate all activation maps and approximations
        # to calculate the metrics in a less noisy manner.
        self.y_hat: torch.Tensor | None = None
        self.y_out: torch.Tensor | None = None
        self.gts: torch.Tensor | None = None
        self.clear_outputs = True
        self.final_metrics: dict = {}

    def zero_saved_values(self):
        self.y_hat = None
        self.y_out = None
        self.gts = None

    def on_fit_end(self) -> None:
        """
        Writes the last validation metrics and closes summary writers.
        """
        serializable_metrics = deepcopy(self.final_metrics)
        for key, val in self.final_metrics.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    if isinstance(v, (dict, list)):
                        continue
                    else:
                        serializable_metrics[key + "/" + k] = float(v)
            else:
                serializable_metrics[key] = val

        # Create the final metrics instead here!
        if self.do_log:
            save_json(self.final_metrics, self.mode_info.path_last_metrics_json)
            self.tb_logger_tr.close()
            self.tb_logger_val.close()

    def forward(self, x):
        return self.net.forward(x)

    def log_message(self, tensorboard_dict: dict, is_train: bool):
        if self.do_log:
            if is_train:
                sm_wr = self.tb_logger_tr
            else:
                sm_wr = self.tb_logger_val
            for key, val in tensorboard_dict.items():
                if isinstance(val, dict):
                    for k, v in val.items():
                        if isinstance(v, (dict, list)):
                            continue
                        else:
                            sm_wr.add_scalar(key + "/" + k, scalar_value=v, global_step=self.global_step)
                else:
                    sm_wr.add_scalar(key, scalar_value=val, global_step=self.global_step)

    def save_checkpoint(self):
        """Save the checkpoint of the current model."""
        state_dict = self.net.state_dict()
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        if self.current_epoch <= (self.params.num_epochs - 1):
            torch.save(state_dict, self.checkpoint_path)
        return

    def load_latest_checkpoint(self):
        """Loads the latest checkpoint (only of the to be trained architecture)"""
        ckpt = torch.load(self.checkpoint_path)
        self.net.load_state_dict(ckpt)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        dy_fwd = self.loss.forward(label=y, y_out=y_hat)
        return dy_fwd

    def training_epoch_end(self, outputs: list[dict]):
        # Scheduler steps are done automatically! (No step is needed)
        with torch.no_grad():
            loss_values = self.loss.on_epoch_end(outputs)
            prog_bar_log = {"tr/loss": loss_values["loss/total"]}

            self.log_dict(prog_bar_log, prog_bar=True, logger=False)
            self.log_message(loss_values, is_train=True)

    def on_validation_start(self) -> None:
        """
        Empty potential remaining results from before.
        """
        self.zero_saved_values()

    def on_validation_end(self) -> None:
        """
        Empty potential remaining results from before.
        """
        if self.clear_outputs:
            self.zero_saved_values()

    def get_outputs(self) -> dict[str, torch.Tensor]:
        return {"outputs": self.y_out.detach().cpu().numpy(), "groundtruths": self.gts.detach().cpu().numpy()}

    def save_validation_values(
        self,
        groundtruths: torch.Tensor | None,
        y_hat: torch.Tensor | None,
        y_out: torch.Tensor | None,
    ):
        # Save Groundtruths:
        if groundtruths is not None:
            if self.gts is None:
                self.gts = groundtruths
            else:
                self.gts = torch.concatenate([self.gts, groundtruths], dim=0)

        # Aggregate new models outputs
        if y_hat is not None:
            detached_y_hat = y_hat.detach()
            if self.y_hat is None:
                self.y_hat = detached_y_hat
            else:
                self.y_hat = torch.cat([self.y_hat, detached_y_hat], dim=0)

        if y_out is not None:
            detached_y_out = y_out.detach()
            if self.y_out is None:
                self.y_out = detached_y_out
            else:
                self.y_out = torch.cat([self.y_out, detached_y_out], dim=0)

    def validation_step(self, batch, batch_idx):
        x, y = batch  # ["data"], batch["label"]
        with torch.no_grad():
            y_out = self(x)
            y_hat = torch.argmax(y_out, dim=1)  # Fix: Create the argmax of the probabilities
            self.save_validation_values(
                y_hat=y_hat,
                y_out=y_out,
                groundtruths=y,
            )
            return self.loss.forward(label=y, y_out=y_out)

    def validation_epoch_end(self, outputs):
        loss_dict = self.loss.on_epoch_end(outputs)
        single_metrics = single_output_metrics(self.y_out, self.gts, self.ke_hparams["n_cls"])
        self.final_metrics = asdict(single_metrics)

        loss_dict.update({f"metrics/{k}": v for k, v in self.final_metrics.items()})
        prog_bar_log = {"val/acc": single_metrics.accuracy}

        if self.current_epoch != 0:
            if self.save_checkpoints:
                self.save_checkpoint()

        self.log_dict(prog_bar_log, prog_bar=True, logger=False)
        self.log_message(loss_dict, is_train=False)
        return None

    def on_train_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def on_validation_epoch_start(self) -> None:
        self.zero_saved_values()
        self.final_metrics = {}  # Make sure this doesn't contain something from validation

    def on_test_start(self) -> None:
        """
        Empty potential remaining results form before.
        """
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        has_get_parameters = hasattr(self.net, "get_trainable_parameters")
        if has_get_parameters:
            parameters_to_train = self.net.get_trainable_parameters()
        else:
            parameters_to_train = self.net.parameters()

        if self.params.optimizer is not None:
            opti_name = self.params.optimizer["name"]
            if opti_name == "adamw":
                betas = self.params.optimizer["betas"]
                eps = self.params.optimizer["eps"]
                optim = torch.optim.AdamW(
                    params=parameters_to_train,
                    lr=self.params.learning_rate,
                    betas=betas,
                    eps=eps,
                    weight_decay=self.params.weight_decay,
                )
                scheduler = LinearWarmupCosineAnnealingLR(optim, warmup_epochs=35, max_epochs=self.params.num_epochs)
                return [optim], [scheduler]

        else:
            optim = torch.optim.SGD(
                params=parameters_to_train,
                lr=self.params.learning_rate,
                momentum=self.params.momentum,
                weight_decay=self.params.weight_decay,
                nesterov=self.params.nesterov,
            )
            if self.params.cosine_annealing:
                total_epochs = self.params.num_epochs
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_epochs, eta_min=0)
                return [optim], [scheduler]
            else:
                return [optim]
