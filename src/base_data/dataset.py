# -*- coding: utf-8 -*- #
"""Defined classes and functions for Dataset"""

import typing as tp
from pathlib import Path
from abc import ABCMeta, abstractmethod

import numpy as np

import cv2
import albumentations as A
from torch.utils.data import Dataset

FilePath = tp.Union[str, Path]
Label = tp.Union[int, float, np.ndarray]


class PathLabelLazyDataset(Dataset, metaclass=ABCMeta):
    """
    Dataset which receives file paths and labels
    
    Attributes
    ----------
    paths : tp.Sequence[FilePath]
        Sequence of path to input file
    labels : tp.Sequence[Label]
        Sequence of label for input file
    """

    def __init__(
        self,
        paths: tp.Sequence[FilePath],
        labels: tp.Sequence[Label],
    ):
        """Initialize"""
        self.paths = paths
        self.labels = labels

    def __len__(self):
        """Return num of cadence snippets"""
        return len(self.paths)

    def __getitem__(self, index: int):
        """Return preprocessed input and label for given index."""
        path, label = self.paths[index], self.labels[index]
        data = self._read_input_file(path)
        data = self._apply_transform(data)

        return {"data": data, "target": label}

    @abstractmethod
    def _read_input_file(self):
        """Read and preprocess file"""
        pass

    @abstractmethod
    def _apply_transform(self):
        """apply transform to data"""
        pass

    def lazy_init(self, **kwargs):
        """Reset Members"""
        for k, v in kwargs.items():
            assert hasattr(self, k), f"not have a member `{k}`" 
            setattr(self, k, v)


class ImagePathLabelLazyDataset(PathLabelLazyDataset):
    """
    Dataset which receives image file paths and labels
    
    Attributes
    ----------
    paths : tp.Sequence[FilePath]
        Sequence of path to image file
    labels : tp.Sequence[Label]
        Sequence of label for image file
    transform: albumentations.Compose
        composed data augmentations for data
    """

    def __init__(
        self,
        paths: tp.Sequence[FilePath],
        labels: tp.Sequence[Label],
        transform: A.Compose,
    ):
        """Initialize"""
        super().__init__(paths, labels)
        self.transform = transform

    def _read_input_file(self, path: Path):
        """Read and preprocess file"""
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # shape: (H, W, C)
        return img

    def _apply_transform(self, img: np.ndarray):
        """apply transform to image""" 
        img = self.transform(image=img)["image"]
        return img


class ImageLabelLazyDataset(Dataset):
    """
    Dataset which receives iamges and labels
    
    Attributes
    ----------
    images : tp.Sequence[np.ndarray]
        Sequence of path to input file
    labels : tp.Sequence[Label]
        Sequence of label for input file
    transform: albumentations.Compose
        composed data augmentations for data
    """

    def __init__(
        self,
        images: tp.Sequence[np.ndarray],
        labels: tp.Sequence[Label],
        transform: A.Compose,
    ):
        """Initialize"""
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Return num of cadence snippets"""
        return len(self.images)

    def __getitem__(self, index: int):
        """Return preprocessed input and label for given index."""
        data, label = self.images[index], self.labels[index]
        data = self._apply_transform(data)

        return {"data": data, "target": label}

    def _apply_transform(self, img: np.ndarray):
        """apply transform to image""" 
        img = self.transform(image=img)["image"]
        return img

    def lazy_init(self, **kwargs):
        """Reset Members"""
        for k, v in kwargs.items():
            assert hasattr(self, k), f"not have a member `{k}`" 
            setattr(self, k, v)
