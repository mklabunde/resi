from __future__ import annotations

from abc import abstractmethod

from torch import nn
from vision.arch import abstract_acti_extr


class BaseFeatureArch(nn.Module):
    """
    This class is supposed to be a wrapper around old architectures,
        which provide features of an intermediate layer. Also
    """

    @abstractmethod
    def get_new_model(self) -> abstract_acti_extr.AbsActiExtrArch:
        return self.new_arch

    def get_new_model_state_dict(self):
        return self.get_new_model().state_dict()

    def load_new_model_state_dict(self, state_dict):
        self.get_new_model().load_state_dict(state_dict)

    def state_dict(self, *args, **kwargs):
        raise NotImplementedError(
            "This mustn't be implemented to make sure saving is called directly on the new architecture."
        )
