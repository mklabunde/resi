from typing import Type

import torchvision.models
from torch import nn
from vision.arch import abstract_acti_extr
from vision.arch import vit
from vision.util import data_structs as ds

# from ke.arch import o2o_average

# from ke.arch.ensembling import toworkon_loo_replacement

# Instead of automatically going through the registered subclasses
#   this is explicitly stated to keep the code more readable and traceable.


def get_base_arch(
    arch: ds.BaseArchitecture | str,
) -> Type[abstract_acti_extr.AbsActiExtrArch]:
    """Finds a Model network by its name.
    Should the class not be found it will raise an Error.
        :param arch: Name of the network class that should be used.
    :raises NotImplementedError: If not subclass is found for the given name.
    :return: Subclass with same name as network_name
    """
    from vision.arch import vgg, resnet

    if isinstance(arch, str):
        arch = ds.BaseArchitecture(arch)
    if arch == ds.BaseArchitecture.VGG16:
        return vgg.VGG16
    elif arch == ds.BaseArchitecture.VGG11:
        return vgg.VGG11
    elif arch == ds.BaseArchitecture.VGG19:
        return vgg.VGG19
    elif arch == ds.BaseArchitecture.RESNET18:
        return resnet.ResNet18
    elif arch == ds.BaseArchitecture.RESNET34:
        return resnet.ResNet34
    elif arch == ds.BaseArchitecture.RESNET50:
        return resnet.ResNet50
    elif arch == ds.BaseArchitecture.RESNET101:
        return resnet.ResNet101
    elif arch == ds.BaseArchitecture.VIT_B16:
        return vit.VIT_B16
    elif arch == ds.BaseArchitecture.VIT_B32:
        return vit.VIT_B32
    elif arch == ds.BaseArchitecture.VIT_L16:
        return vit.VIT_L16
    elif arch == ds.BaseArchitecture.VIT_L32:
        return vit.VIT_L32
    else:
        raise ValueError("Seems like the BaseArchitecture was not added here!")
