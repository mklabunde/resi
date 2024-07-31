from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import List
from typing import Union

import numpy as np
from torch import nn
from vision.arch.abstract_acti_extr import AbsActiExtrArch
from vision.util.data_structs import BaseArchitecture
from vision.util.data_structs import Hook

arch_cfg: dict[str, list[int | str]] = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class AbsVGG(AbsActiExtrArch, ABC):
    def __init__(
        self,
        n_cls=10,
        in_ch=3,
        input_resolution: tuple[int, int] = (32, 32),
        early_downsampling: bool = False,
        global_average_pooling: int = 4,
    ):
        super().__init__(n_cls, in_ch, input_resolution, early_downsampling, global_average_pooling)
        self.features: nn.Sequential
        self.classifier: nn.Linear

    @staticmethod
    def get_partial_module(module: nn.Module, hook_keys: List[str], first_part: bool):
        """Returns the nn.Module with loaded params until acertain point, depending
        on the module of the Architecture until a certain point.
        After that point removes all layers, therefore one gets as output of that
        nn.Module the intermediate feature-map.

        :param module:
        :param hook_keys:
        :param first_part:
        :return:
        """
        hook_key_val = int(hook_keys[-1])
        sequential_index = hook_key_val + 1

        if first_part:
            return module.features[:sequential_index]
        else:
            return module.features[sequential_index:]

    def create_hooks(self):

        self.hooks = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                self.hooks.append(
                    Hook(
                        architecture_index=len(self.hooks),
                        name=name,
                        keys=name.split("."),
                        n_channels=module.out_channels,
                        downsampling_steps=-1,
                        resolution=None,
                    )
                )
        self.hooks.append(
            Hook(
                architecture_index=len(self.hooks),
                name="classifier",
                keys=["classifier"],
                n_channels=0,
                at_input=True,
            )
        )
        return

    def get_predecessing_convs(self, hook) -> List[nn.Conv2d]:
        conv_hook_key_val = int(hook.keys[-1]) - 1
        return [self.features[conv_hook_key_val]]

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    @staticmethod
    def get_intermediate_module(
        module: nn.Module,
        front_hook_keys: Union[List[str], None],
        end_hook_keys: Union[List[str], None],
    ) -> nn.Module:
        """Returns a sequential ResNet that is split at the given Hook position.
        If first_part is specified it returns everything up until the Hook point (
        including the hook point).
        If false, it takes everything from the Hook point (excluding the Hook point),
        which should be a ReLU
        :param module:
        :param front_hook_keys:
        :param end_hook_keys:
        :return:
        """
        sequential = getattr(module, "features")  # Now is the sequential part of the DenseNets
        front_hook_keys = (
            [key for key in front_hook_keys[1:]] if isinstance(front_hook_keys, list) else None
        )  # Ignores the "features" hook key
        end_hook_keys = (
            [key for key in end_hook_keys[1:]] if isinstance(end_hook_keys, list) else None
        )  # Ignores the "features" hook key

        if front_hook_keys is None:
            front_first_index = 0
        else:
            front_first_index = int(front_hook_keys[0])

        if end_hook_keys is None:
            end_first_index = -1
        else:
            end_first_index = int(end_hook_keys[0])

        if front_hook_keys is None and end_hook_keys is None:
            return sequential  # Returns the whole model (everything in the features block)
        elif front_hook_keys is not None and end_hook_keys is None:
            return sequential[front_first_index + 1 :]
        elif front_hook_keys is None and end_hook_keys is not None:
            return sequential[: end_first_index + 1]
        else:
            return sequential[front_first_index + 1 : end_first_index + 1]

    @staticmethod
    def get_linear_layer(module: nn.Module) -> nn.Module:
        return nn.Sequential(nn.Flatten(), getattr(module, "classifier"))

    @staticmethod
    def get_channels(module) -> int:
        if isinstance(module, nn.Conv2d):
            return module.out_channels
        elif isinstance(module, nn.BatchNorm2d):
            return module.num_features
        else:
            raise NotImplementedError("Not supported layer selected for merging")

    def get_wanted_module(self, hook: Hook | Sequence[str]) -> nn.Module:
        cur_module = self
        if isinstance(hook, Hook):
            keys: Sequence[str] = hook.keys
        else:
            keys = hook
        for key in keys:
            cur_module = getattr(cur_module, key)
        return cur_module

    @staticmethod
    def _make_layers(cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=(3, 3), padding=(1, 1)),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d(output_size=1)]
        return nn.Sequential(*layers)


class VGG16(AbsVGG):
    architecture_id = BaseArchitecture.VGG16
    n_hooks = 14

    def __init__(
        self,
        n_cls: int = 10,
        in_ch: int = 3,
        input_resolution: tuple[int, int] = (32, 32),
        early_downsampling: bool = False,
        global_average_pooling: int = 4,
    ):
        super().__init__(n_cls, in_ch, input_resolution, early_downsampling)
        self.features = self._make_layers(arch_cfg["VGG16"])
        self.classifier = nn.Linear(512, n_cls)
        self.create_hooks()


class VGG11(AbsVGG):
    architecture_id = BaseArchitecture.VGG11
    n_hooks = 9

    def __init__(
        self,
        n_cls: int = 10,
        in_ch: int = 3,
        input_resolution: tuple[int, int] = (32, 32),
        early_downsampling: bool = False,
        global_average_pooling: int = 4,
    ):
        super().__init__(n_cls, in_ch, input_resolution, early_downsampling)
        self.features = self._make_layers(arch_cfg["VGG11"])
        self.classifier = nn.Linear(512, n_cls)
        self.create_hooks()


class VGG19(AbsVGG):
    architecture_id = BaseArchitecture.VGG19
    n_hooks = 17

    def __init__(
        self,
        n_cls=10,
        in_ch=3,
        input_resolution: tuple[int, int] = (32, 32),
        early_downsampling: bool = False,
        global_average_pooling: int = 4,
    ):
        super().__init__(n_cls, in_ch, input_resolution, early_downsampling)
        self.features = self._make_layers(arch_cfg["VGG19"])
        self.classifier = nn.Linear(512, n_cls)
        self.create_hooks()


if __name__ == "__main__":

    print("VGG11 ", len(VGG11().hooks))
    print("VGG16 ", len(VGG16().hooks))
