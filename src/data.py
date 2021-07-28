# -*- coding: utf-8 -*- #
"""Newly created dataset, dataloader, transoforms for competition"""
from base_data import ImagePathLabelLazyDataset


class ContrastiveImagePathLabelLazyDataset(ImagePathLabelLazyDataset):
    """
    Dataset which receives `two` iamges applied different augmentations
    
    Attributes
    ----------
    paths : tp.Sequence[FilePath]
        Sequence of path to image file
    labels : tp.Sequence[Label]
        Sequence of label for image file
    transform: albumentations.Compose
        composed data augmentations for data
    """

    def __getitem__(self, index: int):
        """Return preprocessed `two` input images for given index."""
        path, label = self.paths[index], self.labels[index]
        data = self._read_input_file(path)
        data0 = self._apply_transform(data)
        data1 = self._apply_transform(data)
        return {"data0": data0, "data1": data1, "target": label}
