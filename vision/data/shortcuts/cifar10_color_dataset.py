from copy import deepcopy
from typing import Any, Tuple
from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR10

from vision.data.shortcuts.shortcut_transforms import ColorDotShortcut
from PIL import Image


class ColorDotCIFAR10(CIFAR10):

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        dot_correlation: int = 100,
        dot_diameter=5,
        rng_seed=123,
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self._color_sc_gen = ColorDotShortcut(
            n_classes=10,
            n_channels=3,
            image_size=(32, 32),
            dataset_mean=0,
            dataset_std=1,
            correlation_prob=dot_correlation / 100.0,
            dot_diameter=dot_diameter,
        )
        self._color_sc_gen.set_rng_seed(rng_seed)
        self.old_data = deepcopy(self.data)

        shortcut_images = []
        color_labels = []
        for i in range(len(self.data)):
            image, color_label = self._color_sc_gen.forward(self.data[i], self.targets[i])
            shortcut_images.append(image)
            color_labels.append(color_label)
        self.data = shortcut_images
        self.color_labels = color_labels

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, color_label = self.data[index], self.targets[index], self.color_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, color_label


if __name__ == "__main__":
    n_samples = 30
    dot_corr = 100
    dot_diam = 5
    from paths import VISION_DATA_PATH
    import os
    from pathlib import Path

    save_path = Path(__file__).parent / "example_imgs"
    save_path.mkdir(exist_ok=True)

    # ----------------------------- Actual execution ----------------------------- #
    cdotc10 = ColorDotCIFAR10(
        root=os.path.join(VISION_DATA_PATH, "CIFAR10"),
        train=True,
        transform=None,
        target_transform=None,
        download=True,
        dot_correlation=dot_corr,
        dot_diameter=dot_diam,
        rng_seed=1,
    )

    for i in range(n_samples):
        cdot_image = cdotc10.data[i]
        raw_image = cdotc10.old_data[i]
        target = cdotc10.targets[i]
        color_label = cdotc10.color_labels[i]

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(raw_image)
        axs[0].set_title(f"Orig. Image; lbl: {target}")
        axs[1].imshow(cdot_image)
        axs[1].set_title(f"ColorDot Image; color lbl: {color_label}")
        plt.savefig(save_path / f"color_dot_cifar10_{i}.png")
