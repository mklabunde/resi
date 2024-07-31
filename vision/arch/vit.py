from collections.abc import Sequence

from repsim.benchmark.paths import VISION_MODEL_PATH
from torch import nn
from torchvision.models import VisionTransformer
from transformers import ViTForImageClassification
from transformers.models.vit.modeling_vit import ViTLayer
from vision.arch.abstract_acti_extr import AbsActiExtrArch
from vision.util.data_structs import BaseArchitecture
from vision.util.data_structs import Hook


def find_layers_of_interest(model: ViTForImageClassification):
    """Finds the layers of interest for the ViT architecture.

    :param model: The ViT model
    :return: A list of layers of interest
    """
    layers_of_interest = {}
    for name, module in model.named_modules():
        if isinstance(module, ViTLayer):
            layers_of_interest[name] = module
        elif name.endswith("blocks"):
            layers_of_interest[name] = module
        elif isinstance(module, nn.Linear) and name.endswith("classifier"):
            layers_of_interest[name] = module
    return layers_of_interest


class AbstractActiViT(AbsActiExtrArch):
    def __init__(self):
        super().__init__()
        self.image_size = 224

    def create_hooks(self):
        mods_of_interest = find_layers_of_interest(self)
        hooks = [
            Hook(
                cnt,
                name,
                name.split("."),
                n_channels=None,
                downsampling_steps=-1,
                resolution=None,
                resolution_relative_depth=None,
                at_input=True if name.endswith("classifier") else False,
            )
            for cnt, name in enumerate(mods_of_interest.keys())
        ]
        return hooks

    def get_wanted_module(self, hook: Hook | Sequence[str]) -> nn.Module:
        return self.get_submodule(hook.name)

    def forward(self, x):
        return self.model(x)["logits"]


class VIT_B16(AbstractActiViT):
    architecture_id = BaseArchitecture.VIT_B16
    n_hooks = 13

    def __init__(self, **kwargs):
        super().__init__()
        assert kwargs["input_resolution"][0] == 224, "Input resolution must be 224!"
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            cache_dir=VISION_MODEL_PATH,
        )
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, kwargs["n_cls"])
        self.hooks = self.create_hooks()


class VIT_B32(AbstractActiViT):
    architecture_id = BaseArchitecture.VIT_B32
    n_hooks = 13

    def __init__(self, **kwargs):
        super().__init__()
        assert kwargs["input_resolution"][0] == 224, "Input resolution must be 224!"
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch32-224-in21k",
            cache_dir=VISION_MODEL_PATH,
        )
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, kwargs["n_cls"])
        self.hooks = self.create_hooks()


class VIT_L16(AbstractActiViT):
    architecture_id = BaseArchitecture.VIT_L16
    n_hooks = 13

    def __init__(self, **kwargs):
        super().__init__()
        assert kwargs["input_resolution"][0] == 224, "Input resolution must be 224!"
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-large-patch16-224-in21k",
            cache_dir=VISION_MODEL_PATH,
        )
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, kwargs["n_cls"])
        self.hooks = self.create_hooks()


class VIT_L32(AbstractActiViT):
    architecture_id = BaseArchitecture.VIT_L32
    n_hooks = 13

    def __init__(self, **kwargs):
        super().__init__()
        assert kwargs["input_resolution"][0] == 224, "Input resolution must be 224!"
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-large-patch32-224-in21k",
            cache_dir=VISION_MODEL_PATH,
        )
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, kwargs["n_cls"])
        self.hooks = self.create_hooks()
