from torch import nn
from vision.arch.ke_architectures.base_feature_arch import BaseFeatureArch


class SingleModel(BaseFeatureArch):
    def __init__(
        self,
        new_model: nn.Module,
    ):
        super(SingleModel, self).__init__()
        self.tbt_arch = new_model

    def get_new_model(self):
        return self.tbt_arch

    def forward(self, x):
        """
        Does forward through the different architecture blocks.
        :returns approximation/transfer, intermediate tbt features, output logits
        """
        out = self.tbt_arch(x)
        return out
