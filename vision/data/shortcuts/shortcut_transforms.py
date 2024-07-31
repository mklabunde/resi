from abc import abstractmethod
from colorsys import hsv_to_rgb

import numpy as np
import torch
from skimage import draw


def create_unique_colors(n_classes: int) -> np.ndarray:
    """Create unique colors for each class."""
    colors = np.zeros((n_classes, 3), dtype=np.uint8)
    for i in range(n_classes):
        hue = i * (360 / n_classes)
        rgb = hsv_to_rgb(hue / 360, 1, 1)
        colors[i] = (np.array(rgb) * 255).astype(np.uint8)
    return colors


class AbstractShortcut(torch.nn.Module):
    @abstractmethod
    def apply_shortcut(self, img: torch.Tensor, lbl: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ColorDotShortcut(AbstractShortcut):
    def __init__(
        self,
        n_classes: int,
        n_channels: int,
        image_size: tuple[(int, int)],
        dataset_mean: float,
        dataset_std: float,
        correlation_prob: float = 1.0,
        dot_diameter: int = 5,
    ):
        """Create a ColorDotShortcut that can be used to add a shortcut to a dataset."""
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.image_size = image_size
        self.correlation_prob = correlation_prob
        self.dot_diameter = dot_diameter
        self.mean = dataset_mean
        self.std = dataset_std
        self.colors = create_unique_colors(n_classes)
        self._rng_gen = np.random.default_rng(seed=123)

    def _get_color(self, label: int) -> np.ndarray:
        """Get the color for a specific label."""
        return (self.colors[label] - self.mean) / self.std

    def set_rng_seed(self, seed: int) -> None:
        self._rng_gen = np.random.default_rng(seed)

    def _get_color_dot_coords(self, label: int) -> tuple[int, int, np.ndarray, int, int]:
        """Get the coordinates and color for a specific label."""
        # Maybe choose a different class label if the correlation is not 100%
        orig_label = label
        if self._rng_gen.random() > self.correlation_prob:
            label = self._rng_gen.integers(self.n_classes)
        color = self._get_color(label)
        boundaries = self.dot_diameter // 2
        x_center = self._rng_gen.integers(boundaries, self.image_size[0] - boundaries)
        y_center = self._rng_gen.integers(boundaries, self.image_size[1] - boundaries)

        return x_center, y_center, color, label, orig_label

    def _color_dot_from_coords(self, x_center: int, y_center: int, color: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """Draw a dot on a black image."""
        boundaries = self.dot_diameter // 2
        rr, cc = draw.disk((x_center, y_center), boundaries + 1.5, shape=(self.image_size[0], self.image_size[1]))
        dot_mask = np.zeros((self.image_size[0], self.image_size[1], self.n_channels), dtype=dtype)
        color_dot = np.zeros((self.image_size[0], self.image_size[1], self.n_channels), dtype=dtype)
        for c in range(self.n_channels):
            color_dot[rr, cc, c] = color[c]
            dot_mask[rr, cc, c] = 1

        return dot_mask, color_dot

    def _get_random_color_dot(self, label: int, dtype: np.dtype) -> np.ndarray:
        """Get a random color dot for a specific label."""
        x_center, y_center, color, color_label, cls_label = self._get_color_dot_coords(label)
        return self._color_dot_from_coords(x_center, y_center, color, dtype)

    def apply_shortcut(self, image: np.ndarray, dot_mask, color_dot) -> np.ndarray:
        dot_mask = dot_mask.astype(image.dtype)
        color_dot = color_dot.astype(image.dtype)

        sc_image = image * (1 - dot_mask) + color_dot
        return sc_image

    def forward(self, image: np.ndarray, label: np.ndarray) -> np.ndarray:
        """Takes a batch of images and labels and adds a colored dot to the image.
        The color is the same as the class if"""
        if self._rng_gen.random() > self.correlation_prob:
            label = self._rng_gen.integers(self.n_classes)  # Choose any random label instead of true one

        dot_mask, color_dot = self._get_random_color_dot(label, image.dtype)
        sc_image = image * (1 - dot_mask) + color_dot
        return sc_image, label
