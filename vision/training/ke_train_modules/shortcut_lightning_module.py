from __future__ import annotations

from dataclasses import asdict

import torch
from vision.arch.ke_architectures.single_model import SingleModel
from vision.losses.dummy_loss import DummyLoss
from vision.metrics.ke_metrics import single_output_metrics
from vision.data.shortcuts.shortcut_transforms import AbstractShortcut
from vision.training.ke_train_modules.base_training_module import BaseLightningModule
from vision.util import data_structs as ds


class ShortcutLightningModule(BaseLightningModule):
    def __init__(
        self,
        model_info: ds.ModelInfo,
        network: SingleModel,
        save_checkpoints: bool,
        params: ds.Params,
        hparams: dict,
        loss: DummyLoss,
        log: bool = True,
    ):
        super().__init__(
            model_info=model_info,
            network=network,
            loss=loss,
            save_checkpoints=save_checkpoints,
            params=params,
            hparams=hparams,
            log=log,
        )

        self.net: SingleModel = network
        self.loss = loss
        self.y_hat = None
        self.y_out = None
        self.gts_im = None
        self.gts_sc = None

    def training_step(self, batch, batch_idx):
        x, y_image, _ = batch

        y_out = self(x)  # Shortcut is only correlated but not predictive of class.
        dy_fwd = self.loss.forward(label=y_image, y_out=y_out)
        return dy_fwd

    def training_epoch_end(self, outputs: list[dict]):
        # Scheduler steps are done automatically! (No step is needed)
        with torch.no_grad():
            loss_values = self.loss.on_epoch_end(outputs)
            prog_bar_log = {"tr/loss": loss_values["loss/total"]}

            self.log_dict(prog_bar_log, prog_bar=True, logger=False)
            self.log_message(loss_values, is_train=True)

        return None

    def save_validation_values(
        self,
        groundtruths_image: torch.Tensor | None,
        groundtruth_short: torch.Tensor | None,
        y_hat: torch.Tensor | None,
        y_out: torch.Tensor | None,
    ):
        # Save Groundtruths:
        if groundtruths_image is not None:
            if self.gts_im is None:
                self.gts_im = groundtruths_image
            else:
                self.gts_im = torch.concatenate([self.gts_im, groundtruths_image], dim=0)

        # Save Groundtruths:
        if groundtruth_short is not None:
            if self.gts_sc is None:
                self.gts_sc = groundtruth_short
            else:
                self.gts_sc = torch.concatenate([self.gts_sc, groundtruth_short], dim=0)

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
        x, y_image, y_short = batch  # ["data"], batch["label"]
        with torch.no_grad():
            y_out = self(x)
            y_hat = torch.argmax(y_out, dim=1)
            self.save_validation_values(
                groundtruth_short=y_short,
                groundtruths_image=y_image,
                y_out=y_out,
                y_hat=y_hat,
            )
            loss_short = self.loss.forward(
                label=y_short,
                y_out=y_out,
            )
            loss_image = self.loss.forward(
                label=y_image,
                y_out=y_out,
            )
        return {"loss_image": loss_image, "loss_short": loss_short}

    def validation_epoch_end(self, outputs):
        loss_image = self.loss.on_epoch_end([o["loss_image"] for o in outputs])
        loss_short = self.loss.on_epoch_end([o["loss_short"] for o in outputs])
        single_metrics_sc = single_output_metrics(self.y_out, self.gts_sc, self.ke_hparams["n_cls"])
        single_metrics_no_sc = single_output_metrics(self.y_out, self.gts_im, self.ke_hparams["n_cls"])
        self.final_metrics = {"no_shortcut": asdict(single_metrics_no_sc), "shortcut": asdict(single_metrics_sc)}

        log_msg = {}
        log_msg.update({f"loss/image/{k}": float(v) for k, v in loss_image.items()})
        log_msg.update({f"loss/short/{k}": float(v) for k, v in loss_short.items()})
        log_msg.update({f"metrics/shortcut/{k}": v for k, v in asdict(single_metrics_sc).items()})
        log_msg.update({f"metrics/no_shortcut/{k}": v for k, v in asdict(single_metrics_no_sc).items()})

        prog_bar_log = {
            "loss/image": loss_image["loss/total"],
            "loss/short": loss_short["loss/total"],
            "val/acc_shortcut": single_metrics_sc.accuracy,
            "val/acc_no_shortcut": single_metrics_no_sc.accuracy,
        }

        if self.current_epoch != 0:
            if self.save_checkpoints:
                self.save_checkpoint()

        self.log_dict(prog_bar_log, prog_bar=True, logger=False)
        self.log_message(log_msg, is_train=False)
        return None

    def zero_saved_values(self):
        self.y_hat = None
        self.y_out = None
        self.gts_im = None
        self.gts_sc = None
