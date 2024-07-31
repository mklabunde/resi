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


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=(3, 3),
        stride=(stride, stride),
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=(dilation, dilation),
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=(1, 1),
        stride=(stride, stride),
        bias=False,
    )


class BasicBlockWOReLU(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride
        # != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.identity = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.identity(out)
        return out


class AbsResNet(AbsActiExtrArch, ABC):
    def set_hook_details(self, input_resolution: tuple[int, int], early_downsampling: bool):
        """Sets the detailled information of the feature maps at the hook location"""
        cur_downsampling = 0
        cur_resolution = input_resolution
        for cnt, hook in enumerate(self.hooks):
            if cnt == 0 and early_downsampling:
                cur_downsampling += 1
                cur_resolution = (cur_resolution[0] // 2, cur_resolution[1] // 2)
            if cnt in self.downsampling_ids:
                if cnt == 1:
                    if early_downsampling:
                        cur_downsampling += 1
                        cur_resolution = (cur_resolution[0] // 2, cur_resolution[1] // 2)
                elif cnt in self.downsampling_ids[1:]:
                    cur_downsampling += 1
                    cur_resolution = (cur_resolution[0] // 2, cur_resolution[1] // 2)
            hook.downsampling_steps = cur_downsampling
            hook.resolution = cur_resolution
        unique_resolutions = np.unique([h.resolution[0] for h in self.hooks])
        for res in unique_resolutions:
            remaining_hooks = [h for h in self.hooks if h.resolution[0] == res]
            n_hooks = len(remaining_hooks)
            rel_depth = np.linspace(0, 100, n_hooks)
            for h, depth in zip(remaining_hooks, rel_depth):
                h.resolution_relative_depth = depth

        # Resolutions relative depth

    @staticmethod
    def get_partial_module(module: nn.Module, hook_keys: List[str], first_part: bool) -> nn.Module:
        """Returns a sequential ResNet that is split at the given Hook position.
        If first_part is specified it returns everything up until the Hook point (
        including the hook point).
        If false, it takes everything from the Hook point (excluding the Hook point),
        which should be a ReLU

        :param module: Current instance. Can be ResNet34/50/101
        :param hook_keys: Hook key to split module at
        :param first_part: Bool. If True returns the front till hook_key
        else from directly after hook_key to end.
        :return: Respective module
        """
        sequential = getattr(getattr(module, "module"), "module")
        # Now is the sequential part of the ResNets
        hook_keys = [key for key in hook_keys[1:]]

        if first_part:
            if len(hook_keys) == 1:  # If backbone is split
                first_index = AbsResNet.get_index_by_name(sequential, hook_keys[0])
                return sequential[: first_index + 1]
            else:
                first_index = AbsResNet.get_index_by_name(sequential, hook_keys[0])
                second_index = AbsResNet.get_index_by_name(sequential[first_index], hook_keys[1])
                sequential = sequential[: first_index + 1]
                sequential[-1] = sequential[first_index][: second_index + 1]
                return sequential

        else:
            if len(hook_keys) == 1:
                return sequential[AbsResNet.get_index_by_name(sequential, hook_keys[0]) + 1 :]
            else:
                first_index = AbsResNet.get_index_by_name(sequential, hook_keys[0])
                second_index = AbsResNet.get_index_by_name(sequential[first_index], hook_keys[1])
                sequential = sequential[first_index:]  # Include the actual first layer, since a part of it
                # will be reused!
                sequential[0] = sequential[0][second_index + 1 :]  # Should continue from the ReLU
                return sequential

    @staticmethod
    def get_intermediate_module(
        module: nn.Module,
        front_hook_keys: Union[List[str], None],
        end_hook_keys: Union[List[str], None],
    ) -> nn.Module:
        """Returns a sequential ResNet that is split at the given Hook position.
        If first_part is specified it returns everything up until the Hook point
         (including the hook point).
        If false, it takes everything from the Hook point (excluding the Hook point),
         which should be a ReLU.

        :param module: Module to split
        :param front_hook_keys: Key or None. If None takes from the beginning
        Else starts from the hook_key (including the key)
        :param end_hook_keys: Key or None. If None takes till the very end.
        Else ends before the specified key (excluding the key)
        :return: Split up module.
        """
        sequential = getattr(getattr(module, "module"), "module")  # Now is the sequential part of the DenseNets
        front_hook_keys = (
            [key for key in front_hook_keys[1:]] if isinstance(front_hook_keys, list) else None
        )  # Ignores the "features" hook key
        end_hook_keys = (
            [key for key in end_hook_keys[1:]] if isinstance(end_hook_keys, list) else None
        )  # Ignores the "features" hook key

        if front_hook_keys is None:
            front_first_index = 0
            front_second_index = None
        elif len(front_hook_keys) == 1:
            front_first_index = (
                AbsResNet.get_index_by_name(sequential, front_hook_keys[0]) + 1
            )  # If we don't have an inner layer we skip the old hook.
            front_second_index = None
        else:
            front_first_index = AbsResNet.get_index_by_name(
                sequential, front_hook_keys[0]
            )  # If we have inner layers we keep old position
            front_second_index = (
                AbsResNet.get_index_by_name(sequential[front_first_index], front_hook_keys[1]) + 1
            )  # We find id of inner part & skip it

        if end_hook_keys is None:
            end_first_index = -1
            end_second_index = None
        elif len(end_hook_keys) == 1:
            end_first_index = (
                AbsResNet.get_index_by_name(sequential, end_hook_keys[0]) + 1
            )  # If we have no inner part we want to not go till that id
            end_second_index = None
        else:
            end_first_index = (
                AbsResNet.get_index_by_name(sequential, end_hook_keys[0]) + 1
            )  # If inner part exists we want to use inner part
            # Since we iterate till index + 1 we need to grab from the index before
            # for the actual last part we grab
            end_second_index = (
                AbsResNet.get_index_by_name(sequential[end_first_index - 1], end_hook_keys[1]) + 1
            )  # we go and take it

        if front_hook_keys is None and end_hook_keys is None:
            return sequential  # Returns the whole model (everything in the features block)
        else:
            intermediate_model = sequential[front_first_index:end_first_index]
            if end_second_index is not None:
                intermediate_model[-1] = nn.Sequential(*list(intermediate_model[-1])[:end_second_index])
            if front_second_index is not None:
                intermediate_model[0] = nn.Sequential(*list(intermediate_model[0])[front_second_index:])

            return intermediate_model

    @staticmethod
    def get_linear_layer(module: nn.Module) -> nn.Module:
        return getattr(getattr(module, "module"), "module")[-1]

    @staticmethod
    def get_index_by_name(module: nn.Sequential, name: str):
        return list(dict(module.named_children()).keys()).index(name)

    @staticmethod
    def get_channels(module) -> int:
        if isinstance(module, nn.Conv2d):
            return module.out_channels
        elif isinstance(module, nn.BatchNorm2d):
            return module.num_features
        elif isinstance(module, BottleneckWOReLU):
            return module.bn3.num_features
        elif isinstance(module, BasicBlockWOReLU):
            return module.bn2.num_features
        else:
            raise NotImplementedError("Not supported layer selected for merging")

    def get_wanted_module(self, hook: Hook | Sequence[str]) -> nn.Module:
        if isinstance(hook, Hook):
            cur_module = self.module
            keys: Sequence[str] = hook.keys
        else:
            cur_module = self
            keys = hook
        for key in keys:
            cur_module = getattr(cur_module, key)
        return cur_module

    def get_predecessing_convs(self, hook) -> List[nn.Conv2d]:
        cur_module = self.module
        if hook.name == "bn0":
            return [getattr(cur_module, "conv0")]
        else:
            layer_module: nn.Module = getattr(cur_module, hook.keys[1])
            submodule_names = [k[0] for k in layer_module.named_parameters()]
            if layer_module.__class__.__name__ == "BottleneckWOReLU":
                if "downsample.weight" in submodule_names:
                    return [layer_module.conv3, layer_module.downsample]
                else:
                    return [layer_module.conv3]
            elif layer_module.__class__.__name == "BasicBlockWOReLU":
                if "downsample.weight" in submodule_names:
                    return [layer_module.conv2, layer_module.downsample]
                else:
                    return [layer_module.conv2]
            else:
                raise ValueError("Unexpected positioning.")


class ResNet34(AbsResNet):
    architecture_id = BaseArchitecture.RESNET34
    n_hooks: int = 18

    def __init__(
        self,
        n_cls: int = 10,
        in_ch: int = 3,
        input_resolution: tuple[int, int] = (32, 32),
        early_downsampling: bool = False,
        global_average_pooling: int = 4,
    ):
        super().__init__(n_cls, in_ch, input_resolution, early_downsampling)
        n_channels = [
            [64, 64, 64],
            [128, 128, 128, 128],
            [256, 256, 256, 256, 256, 256],
            [512, 512, 512],
        ]
        layers = [3, 4, 6, 3]
        self.downsampling_ids = [1, 1 + layers[0], 1 + sum(layers[:2]), 1 + sum(layers[:3])]
        _tmp_counter = 1
        self.hooks = [(Hook(architecture_index=0, name="id0", keys=["module", "identity"], n_channels=64))]
        for j, layer in enumerate(layers):
            for i in range(layer):
                self.hooks.append(
                    Hook(
                        architecture_index=len(self.hooks),
                        name=f"id{_tmp_counter}",
                        keys=["module", f"layer{j + 1}", str(i * 2)],
                        n_channels=n_channels[j][i],
                    )
                )
                _tmp_counter += 1
        del _tmp_counter, j, layer, i
        self.hooks.append(
            Hook(
                architecture_index=len(self.hooks),
                name=f"id{len(self.hooks)}",
                keys=["module", "avgpool"],
                n_channels=self.hooks[-1].n_channels,
            )
        )
        self.set_hook_details(input_resolution, early_downsampling)

        self.module = ResNet(
            BasicBlockWOReLU,
            [3, 4, 6, 3],
            n_cls=self.n_cls,
            global_average_pooling=global_average_pooling,
            early_downsampling=early_downsampling,
            zero_init_residual=True,
        )

    def forward(self, x):
        return self.module(x)


class ResNet50(AbsResNet):
    architecture_id = BaseArchitecture.RESNET50
    n_hooks: int = 18

    def __init__(
        self,
        n_cls: int = 10,
        in_ch: int = 3,
        input_resolution: tuple[int, int] = (32, 32),
        early_downsampling: bool = False,
        global_average_pooling: int = 4,
    ):
        super().__init__(n_cls, in_ch, input_resolution, early_downsampling)
        n_channels = [
            [256, 256, 256],
            [512, 512, 512, 512],
            [1024, 1024, 1024, 1024, 1024, 1024],
            [2048, 2048, 2048],
        ]
        layers = [3, 4, 6, 3]
        self.downsampling_ids = [1, 1 + layers[0], 1 + sum(layers[:2]), 1 + sum(layers[:3])]
        _tmp_counter = 1
        self.hooks = [Hook(architecture_index=0, name="id0", keys=["module", "identity"], n_channels=64)]
        for j, layer in enumerate(layers):
            for i in range(layer):
                self.hooks.append(
                    Hook(
                        architecture_index=len(self.hooks),
                        name=f"id{_tmp_counter}",
                        keys=["module", f"layer{j + 1}", str(i * 2)],
                        n_channels=n_channels[j][i],
                    )
                )
                _tmp_counter += 1
        del _tmp_counter, j, layer, i
        self.hooks.append(
            Hook(
                architecture_index=len(self.hooks),
                name=f"id{len(self.hooks)}",
                keys=["module", "avgpool"],
                n_channels=self.hooks[-1].n_channels,
            )
        )
        self.set_hook_details(input_resolution, early_downsampling)
        self.module = ResNet(
            BottleneckWOReLU,
            [3, 4, 6, 3],
            n_cls=self.n_cls,
            global_average_pooling=global_average_pooling,
            early_downsampling=early_downsampling,
            zero_init_residual=True,
        )

    def forward(self, x):
        return self.module(x)


class DynResNet101(AbsResNet):
    architecture_id = BaseArchitecture.DYNRESNET101

    def __init__(
        self,
        n_cls: int = 10,
        in_ch: int = 3,
        input_resolution: tuple[int, int] = (32, 32),
        early_downsampling: bool = False,
        global_average_pooling: int = 4,
        downscale_factor: float = 1.0,
    ):
        super().__init__(n_cls, in_ch, input_resolution, early_downsampling)
        assert 0 <= downscale_factor <= 1.0, "Downscale factor needs to be between 0 and 1"
        n_channels = [
            [int(256 * downscale_factor) for _ in range(3)],
            [int(512 * downscale_factor) for _ in range(4)],
            [int(1024 * downscale_factor) for _ in range(23)],
            [int(2048 * downscale_factor) for _ in range(3)],
        ]
        layers = [3, 4, 23, 3]
        self.downsampling_ids = [1, 1 + layers[0], 1 + sum(layers[:2]), 1 + sum(layers[:3])]
        _tmp_counter = 1
        self.hooks = [Hook(architecture_index=0, name="id0", keys=["module", "identity"], n_channels=64)]
        for j, layer in enumerate(layers):
            for i in range(layer):
                self.hooks.append(
                    Hook(
                        architecture_index=len(self.hooks),
                        name=f"id{_tmp_counter}",
                        keys=["module", f"layer{j + 1}", str(i * 2)],
                        n_channels=n_channels[j][i],
                    )
                )
                _tmp_counter += 1
        del _tmp_counter, j, layer, i
        self.hooks.append(
            Hook(
                architecture_index=len(self.hooks),
                name=f"id{len(self.hooks)}",
                keys=["module", "avgpool"],
                n_channels=self.hooks[-1].n_channels,
            )
        )
        self.set_hook_details(input_resolution, early_downsampling)

        self.module = ResNet(
            BottleneckWOReLU,
            [3, 4, 23, 3],
            n_cls=n_cls,
            global_average_pooling=global_average_pooling,
            early_downsampling=early_downsampling,
            zero_init_residual=True,
        )


class ResNet101(AbsResNet):
    architecture_id = BaseArchitecture.RESNET101
    n_hooks: int = 35

    def __init__(
        self,
        n_cls: int = 10,
        in_ch: int = 3,
        input_resolution: tuple[int, int] = (32, 32),
        early_downsampling: bool = False,
        global_average_pooling: int = 4,
    ):
        super().__init__(n_cls, in_ch, input_resolution, early_downsampling)

        n_channels = [
            [256, 256, 256],
            [512, 512, 512, 512],
            [1024 for _ in range(23)],
            [2048, 2048, 2048],
        ]
        layers = [3, 4, 23, 3]
        self.downsampling_ids = [1, 1 + layers[0], 1 + sum(layers[:2]), 1 + sum(layers[:3])]
        _tmp_counter = 1
        self.hooks = [Hook(architecture_index=0, name="id0", keys=["module", "identity"], n_channels=64)]
        for j, layer in enumerate(layers):
            for i in range(layer):
                self.hooks.append(
                    Hook(
                        architecture_index=len(self.hooks),
                        name=f"id{_tmp_counter}",
                        keys=["module", f"layer{j + 1}", str(i * 2)],
                        n_channels=n_channels[j][i],
                    )
                )
                _tmp_counter += 1
        del _tmp_counter, j, layer, i

        self.hooks.append(
            Hook(
                architecture_index=len(self.hooks),
                name=f"id{len(self.hooks)}",
                keys=["module", "avgpool"],
                n_channels=self.hooks[-1].n_channels,
            )
        )
        self.set_hook_details(input_resolution, early_downsampling)

        self.module = ResNet(
            BottleneckWOReLU,
            [3, 4, 23, 3],
            n_cls=n_cls,
            global_average_pooling=global_average_pooling,
            early_downsampling=early_downsampling,
            zero_init_residual=True,
        )

    def forward(self, x):
        return self.module(x)


class ResNet18(AbsResNet):
    architecture_id = BaseArchitecture.RESNET18
    n_hooks = 10

    def __init__(
        self,
        n_cls: int = 10,
        in_ch: int = 3,
        input_resolution: tuple[int, int] = (32, 32),
        early_downsampling: bool = False,
        global_average_pooling: int = 4,
    ):
        super().__init__(n_cls, in_ch, input_resolution, early_downsampling)
        n_channels = [
            [64, 64],
            [128, 128],
            [256, 256],
            [512, 512],
        ]
        layers = [2, 2, 2, 2]
        self.downsampling_ids = [1, 1 + layers[0], 1 + sum(layers[:2]), 1 + sum(layers[:3])]
        _tmp_counter = 1
        self.hooks = [(Hook(architecture_index=0, name="id0", keys=["module", "identity"], n_channels=64))]
        for j, layer in enumerate(layers):
            for i in range(layer):
                self.hooks.append(
                    Hook(
                        architecture_index=len(self.hooks),
                        name=f"id{_tmp_counter}",
                        keys=["module", f"layer{j + 1}", str(i * 2)],
                        n_channels=n_channels[j][i],
                    )
                )
                _tmp_counter += 1
        del _tmp_counter, j, layer, i
        self.hooks.append(
            Hook(
                architecture_index=len(self.hooks),
                name=f"id{len(self.hooks)}",
                keys=["module", "avgpool"],
                n_channels=self.hooks[-1].n_channels,
            )
        )
        self.set_hook_details(input_resolution, early_downsampling)

        self.module = ResNet(
            BasicBlockWOReLU,
            [2, 2, 2, 2],
            n_cls=self.n_cls,
            global_average_pooling=global_average_pooling,
            early_downsampling=early_downsampling,
            zero_init_residual=True,
        )

    def forward(self, x):
        return self.module(x)


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        n_cls,
        global_average_pooling: int,
        early_downsampling: bool,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        norm_layer=nn.BatchNorm2d,
        downscale_factor=1.0,
    ):
        super().__init__()
        self._norm_layer = norm_layer
        self.inplanes = 64
        if downscale_factor != 1.0:
            self.inplanes = int(downscale_factor * self.inplanes)
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        self.module = nn.Sequential()

        if early_downsampling:
            stride = 2
            padding = 3
        else:
            stride = 1
            padding = 1

        self.module.add_module(
            "conv0",
            nn.Conv2d(
                3,
                self.inplanes,
                kernel_size=(3, 3),
                stride=(stride, stride),
                padding=padding,
                bias=False,
            ),
        )

        self.module.add_module("bn0", nn.BatchNorm2d(self.inplanes))
        self.module.add_module("identity", nn.Identity())
        self.module.add_module("relu", nn.ReLU())
        stride = 2 if early_downsampling else 1
        self.module.add_module(
            "layer1", self._make_sequential_layer(block, int(64 * downscale_factor), layers[0], stride=stride)
        )
        self.module.add_module(
            "layer2", self._make_sequential_layer(block, int(128 * downscale_factor), layers[1], stride=2)
        )
        self.module.add_module(
            "layer3", self._make_sequential_layer(block, int(256 * downscale_factor), layers[2], stride=2)
        )
        self.module.add_module(
            "layer4", self._make_sequential_layer(block, int(512 * downscale_factor), layers[3], stride=2)
        )
        self.module.add_module("avgpool", nn.AvgPool2d((global_average_pooling, global_average_pooling)))
        self.module.add_module("flatten", nn.Flatten())
        self.module.add_module("linear", nn.Linear(int(512 * downscale_factor) * block.expansion, n_cls))

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckWOReLU):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlockWOReLU):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_sequential_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = list()
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        layers.append(nn.ReLU())
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
            layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)


class BottleneckWOReLU(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3
    # convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(
    # self.conv1)
    # according to "Deep residual learning for image
    # recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride
        # != 1
        self.identity = nn.Identity()
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.identity(out)
        return out


if __name__ == "__main__":
    print("ResNet18 ", len(ResNet18().hooks))
    print("ResNet34 ", len(ResNet34().hooks))
    print("ResNet50 ", len(ResNet50().hooks))
    print("ResNet101 ", len(ResNet101().hooks))
