import torch
from torch import nn


class DummyLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 1.0,
    ):
        super(DummyLoss, self).__init__()
        self.ce_weight = ce_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, label: torch.Tensor, y_out: torch.Tensor) -> dict:
        l_ce = self.ce_loss(y_out, label) * self.ce_weight
        return {
            "loss": l_ce,
            "loss_info": {
                "total_loss": l_ce.detach(),
                "classification_raw": l_ce.detach(),
                "classification_weighted": l_ce.detach(),
            },
        }

    def on_epoch_end(self, ke_forward_dict: list[dict]) -> dict:
        class_raw = torch.stack([o["loss_info"]["classification_raw"] for o in ke_forward_dict]).mean()
        class_weighted = torch.stack([o["loss_info"]["classification_weighted"] for o in ke_forward_dict]).mean()
        loss_values = {
            "loss/total": class_weighted,
            "loss_raw/classification": class_raw,
            "loss/classification": class_weighted,
        }
        return loss_values
