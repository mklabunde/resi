import numpy as np
from torchvision.datasets import CIFAR10


class RandomLabelCIFAR10(CIFAR10):

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        rng_seed=123,
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        n_labels = len(set(self.targets))
        rng = np.random.default_rng(rng_seed)
        self.targets = rng.integers(0, n_labels, size=len(self.targets))
