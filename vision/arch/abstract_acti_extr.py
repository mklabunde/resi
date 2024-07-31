from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import List
from typing import Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.hooks import RemovableHandle
from vision.util.data_structs import BaseArchitecture
from vision.util.data_structs import Hook


class AbsActiExtrArch(nn.Module):
    architecture_id: BaseArchitecture
    n_hooks: int

    def __init__(
        self,
        n_cls: int = 10,
        in_ch: int = 3,
        input_resolution: tuple[int, int] = (32, 32),
        early_downsampling: bool = False,
        global_average_pooling: int = 4,
    ):
        """Abstract Activation Extraction Architecture.
        This is the abstract base architecture that is trained to compare models.

        :param n_cls: Number of output classes
        :param in_ch: Number of input channels (Basically RGB/Black&White)
        """
        super().__init__()
        self.hooks: List[Hook] = []
        self.n_cls: int = n_cls
        self.in_ch: int = in_ch
        self.input_resolution: tuple[int, int] = input_resolution
        self.early_downsampling: bool = early_downsampling
        self.global_average_pooling = global_average_pooling
        self.activations: List[np.ndarray] = []
        self.hook_handle: Union[RemovableHandle, None] = None
        self.slice_shift = 0
        self.downsampling_ids: list[int] = []

    def register_rep_hook(self, hook: Hook, wanted_spatial: int = 0) -> None:
        """Registers the next hook in the list of all hooks.
        Should the index of the hooks overflow returns False,
         so interrupt can be handled.

        :param hook: Hook
        :param wanted_spatial: Number of spatial points for
         data samples (0 for whole spatial dimension)

        :return:
        """
        self.slice_shift = 0
        desired_module = self.get_wanted_module(hook)
        self.hook_handle = desired_module.register_forward_hook(
            self.get_layer_output(wanted_spatial=int(wanted_spatial))
        )
        self.activations = []
        return None

    def register_parallel_rep_hooks(self, hook: Hook, save_container: list):

        desired_module = self.get_wanted_module(hook)
        handle = desired_module.register_forward_hook(
            self.get_layer_output_parallel(
                save_container,
                at_input=hook.at_input,
            )
        )
        return handle

    @abstractmethod
    def get_wanted_module(self, hook: Hook | Sequence[str]) -> nn.Module:
        """Registers the next hook in the list of all hooks.
        Should the index of the hooks overflow returns False,
         so interrupt can be handled.

        :param hook: Hook specifying which layer to use

        :return:
        """
        pass

    @abstractmethod
    def get_predecessing_convs(self, hook) -> List[nn.Conv2d]:
        """Finds the convolutions that are predecessors of the hook.
        For VGG&Densenet this will be a single conv and for ResNet
        this returns both previous convs (skip and non-skip conv)."""

    @staticmethod
    @abstractmethod
    def get_channels(module) -> int:
        """Basic method that returns the output dimensions of a module.
        The module can be of varying type, depending on the architecture.

        :param module: nn.Module
        :raises NotImplemented Error should the given Module not be supported
        :return: Int corresponding to the channels (number of features)
        """

    @staticmethod
    @abstractmethod
    def get_partial_module(module: nn.Module, hook_keys: List[str], first_part: bool) -> nn.Module:
        """Abstract function that every activation extraction model has to impement.
        It will return a module, that outputs the features after/before the provided
        hook key. This differs for all the Architectures, since the positions where the
        splitting takes place is at different "depths". For VGG its super easy, but for
        DenseNet and ResNet the module is nested making it more difficult.
        That is also the reason why the hook_keys are multi-dimensional.

        :param module: Module to split
        :param hook_keys:
        :param first_part:
        :return:
        """

    @staticmethod
    @abstractmethod
    def get_intermediate_module(
        module: nn.Module,
        front_hook_keys: Union[List[str], None],
        end_hook_keys: Union[List[str], None],
    ) -> nn.Module:
        """Abstract method that returns a partial module.
        It starts at the front_hook_key position (including the key)
        and return everything until the end_hook_keys (excluding the key).
        If first_part is specified it returns everything up until the Hook point
         (including the hook point).
        If false, it takes everything from the Hook point (excluding the Hook point),
         which should be a ReLU
        :param module:
        :param front_hook_keys:
        :param end_hook_keys:
        :return:
        """

    @staticmethod
    @abstractmethod
    def get_linear_layer(module: nn.Module) -> nn.Module:
        """Abstract function that returns the linear layer of the architecture.
        This is identical to the last layer which outputs the logits.
        Return the corresponding nn.Module.
        """

    def remove_forward_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
        self.slice_shift = 0
        return

    def get_layer_output(self, wanted_spatial: int = 0, at_input=False):
        def hook(model, inp, output):
            """
            Attaches a forward hook that takes the output of a layer,
            checks how high the spatial extent is and only saves as many values
            of the representations as passed in wrapper `wanted_spatial`.

            ATTENTION: This procedure removes location information, making intra-layer comparisons
            based off pooling or something like it impossible!
            """
            # self.activations.append(output.detach().cpu().numpy())
            if at_input:
                output = inp
            output = output.detach().cpu().numpy()
            output_shape = output.shape  # Batch x Channel x Width x Height?
            wh_pixel = output_shape[2] * output_shape[3]
            flat_output = np.reshape(output, [output_shape[0], output_shape[1], -1])
            if wanted_spatial < wh_pixel and wanted_spatial != 0:  # Undersample
                ids = np.sort(
                    np.floor((np.linspace(0, wh_pixel, wanted_spatial, endpoint=False) + self.slice_shift) % wh_pixel)
                ).astype(int)
                flat_output = flat_output[:, :, ids]
                self.slice_shift = self.slice_shift + 1
                if self.slice_shift - 1 >= wanted_spatial:
                    self.slice_shift = 0
            self.activations.append(flat_output)

        return hook

    def get_layer_output_parallel(self, container: list, at_input=False):
        def hook(model, inp, output):
            """
            Attaches a forward hook that takes the output of a layer,
            checks how high the spatial extent is and only saves as many values
            of the representations as passed in wrapper `wanted_spatial`.

            ATTENTION: This procedure removes location information, making intra-layer comparisons
            based off pooling or something like it impossible!
            """
            # self.activations.append(output.detach().cpu().numpy())
            if at_input:
                output = inp
            if isinstance(output, tuple):
                output = output[0]  # VGG19 has a tuple as output of its linear layer...
            container.append(output.cpu())

        return hook

    def get_relative_rep_hook_parallel(
        self,
        anchor_reps: torch.Tensor,
        container: list,
    ):
        """
        Register a hook that can be used to calcualte relative representations.
        Uses the passed anchor_reps to calculate the cosine similarity between the anchors and the current batch.
        Saves the cosine similarity matrix in the container.
        :param anchor_reps: Tensor of shape (n_anchors, n_features). Has to be the same n_features as the layer

        """

        def hook(model, inp, output):
            """
            Attaches a forward hook that takes the output of a layer,
            checks how high the spatial extent is and only saves as many values
            of the representations as passed in wrapper `wanted_spatial`.

            ATTENTION: This procedure removes location information, making intra-layer comparisons
            based off pooling or something like it impossible!
            """
            # self.activations.append(output.detach().cpu().numpy())
            flat_output = torch.reshape(output, [output.shape[0], -1])
            cos_sim = F.cosine_similarity(flat_output[:, None, :], anchor_reps[None, ...], dim=2, eps=1e-8).cpu()
            container.append(cos_sim)

        return hook

    def get_layer_cka_dissim_matrix_parallel(self, container: list):
        def hook(model, inp, output):
            """
            Attaches a forward hook that takes the output of a layer,
            checks how high the spatial extent is and only saves as many values
            of the representations as passed in wrapper `wanted_spatial`.

            ATTENTION: This procedure removes location information, making intra-layer comparisons
            based off pooling or something like it impossible!
            """
            # self.activations.append(output.detach().cpu().numpy())
            output_shape = output.shape  # Batch x Channel x Width x Height?
            flat_output = torch.reshape(output, [output_shape[0], -1])  # Batch x (Channel x Width x Height)
            flat_output = flat_output.to(torch.float64)
            container[0] = flat_output @ flat_output.T

        return hook
