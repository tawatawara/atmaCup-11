# -*- coding: utf-8 -*- #
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .dataset import PathLabelLazyDataset
from .dataset import ImagePathLabelLazyDataset, ImageLabelLazyDataset
from .transform import RandomErase

CONFIG_TYPES = {
    # # Dataset
    "ImagePathLabelLazyDataset": ImagePathLabelLazyDataset,
    "ImageLabelLazyDataset": ImageLabelLazyDataset,

    # # DataLoader
    "DataLoader": DataLoader,

    # # Data Augmentation
    "Compose": A.Compose,
    "OneOf": A.OneOf,
    "Resize": A.Resize,
    "PadIfNeeded": A.PadIfNeeded,
    "HorizontalFlip": A.HorizontalFlip,
    "VerticalFlip": A.VerticalFlip,
    "RandomRotate90": A.RandomRotate90,
    "ShiftScaleRotate": A.ShiftScaleRotate,
    "RandomResizedCrop": A.RandomResizedCrop,
    "Cutout": A.Cutout,
    "CoarseDropout": A.CoarseDropout,
    "RandomGridShuffle": A.RandomGridShuffle,
    "RandomBrightnessContrast": A.RandomBrightnessContrast,
    "RandomGamma": A.RandomGamma,
    "CLAHE": A.CLAHE,
    "GaussianBlur": A.GaussianBlur,
    "ColorJitter": A.ColorJitter,
    "HueSaturationValue": A.HueSaturationValue,
    "ToGray": A.ToGray,
    "ChannelShuffle": A.ChannelShuffle,
    "GaussNoise": A.GaussNoise,
    "Normalize": A.Normalize,
    "RandomErase": RandomErase,
    "ToTensorV2": ToTensorV2,
}