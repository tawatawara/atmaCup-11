# -*- coding: utf-8 -*- #
"""Data Augmentation for Images"""
import typing as tp

import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform


class RandomErase(ImageOnlyTransform):
    """Class of RandomErase for Albumentations."""

    def __init__(
        self, s: tp.Tuple[float] = (0.02, 0.4), r: tp.Tuple[float] = (0.3, 2.7),
        mask_value_min: int = 0, mask_value_max: int = 255,
        always_apply: bool = False, p: float = 1.0
    ) -> None:
        """Initialize."""
        super().__init__(always_apply, p)
        self.s = s
        self.r = r
        self.mask_value_min = mask_value_min
        self.mask_value_max = mask_value_max

    def apply(self, image: np.ndarray, **params):
        """
        Apply transform.

        Note: Input image shape is (Height, Width, Channel).
        """
        image_copy = np.copy(image)

        # # decide mask value randomly
        mask_value = np.random.randint(self.mask_value_min, self.mask_value_max + 1)

        h, w, _ = image.shape
        # # decide num of pixcels for mask.
        mask_area_pixel = np.random.randint(h * w * self.s[0], h * w * self.s[1])

        # # decide aspect ratio for mask.
        mask_aspect_ratio = np.random.rand() * self.r[1] + self.r[0]

        # # decide mask hight and width
        mask_height = int(np.sqrt(mask_area_pixel / mask_aspect_ratio))
        if mask_height > h - 1:
            mask_height = h - 1
        mask_width = int(mask_aspect_ratio * mask_height)
        if mask_width > w - 1:
            mask_width = w - 1

        # # decide position of mask.
        top = np.random.randint(0, h - mask_height)
        left = np.random.randint(0, w - mask_width)
        bottom = top + mask_height
        right = left + mask_width
        image_copy[top:bottom, left:right, :].fill(mask_value)

        return image_copy
