# -*- coding: utf-8 -*- #
"""Utility Functions."""
import os
import gc
import time
import random
import logging
import typing as tp
from contextlib import contextmanager

import yaml
import numpy as np

from scipy.sparse import coo_matrix
from sklearn.utils import shuffle as skl_shuffle
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

import torch


def to_device(
    tensors: tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor]],
    device: torch.device, *args, **kwargs
):
    if isinstance(tensors, tuple):
        return (t.to(device, *args, **kwargs) for t in tensors)
    elif isinstance(tensors, dict):
        return {
            k: t.to(device, *args, **kwargs) for k, t in tensors.items()}
    else:
        return tensors.to(device, *args, **kwargs)


def get_timestamp():
    """Get timestamp now"""
    return time.strftime('%Y%m%d_%H%M%S')


def load_yaml_file(path: str):
    """Load YAML config file."""
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def set_random_seed(seed: int = 42, deterministic: bool = False):
    """Set seeds"""
    os.environ["PYTHONHASHSEED"] = str(seed)  # python
    random.seed(seed)  # python
    np.random.seed(seed)  # cpu
    torch.manual_seed(seed)  # cpu
    if torch.cuda.is_available():  # gpu
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic


@contextmanager
def timer(logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None):
    if prefix: format_str = str(prefix) + format_str
    if suffix: format_str = format_str + str(suffix)
    start = time.time()
    yield
    d = time.time() - start
    out_str = format_str.format(d)
    if logger:
        logger.info(out_str)
    else:
        print(out_str)


def get_logger(out_file=None):
    """Set logger"""
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handlers = []
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    if out_file is not None:
        fh = logging.FileHandler(out_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    logger.info("logger set up")
    return logger


def multi_label_stratified_group_k_fold(label_arr: np.array, gid_arr: np.array, n_fold: int, seed: int=42):
    """
    create multi-label stratified group kfold indexs.

    reference: https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    input:
        label_arr: numpy.ndarray, shape = (n_train, n_class)
            multi-label for each sample's index using multi-hot vectors
        gid_arr: numpy.array, shape = (n_train,)
            group id for each sample's index
        n_fold: int. number of fold.
        seed: random seed.
    output:
        yield indexs array list for each fold's train and validation.
    """
    np.random.seed(seed)
    random.seed(seed)
    start_time = time.time()
    n_train, n_class = label_arr.shape
    gid_unique = sorted(set(gid_arr))
    n_group = len(gid_unique)

    # # aid_arr: (n_train,), indicates alternative id for group id.
    # # generally, group ids are not 0-index and continuous or not integer.
    gid2aid = dict(zip(gid_unique, range(n_group)))
#     aid2gid = dict(zip(range(n_group), gid_unique))
    aid_arr = np.vectorize(lambda x: gid2aid[x])(gid_arr)

    # # count labels by class
    cnts_by_class = label_arr.sum(axis=0)  # (n_class, )

    # # count labels by group id.
    col, row = np.array(sorted(enumerate(aid_arr), key=lambda x: x[1])).T
    cnts_by_group = coo_matrix(
        (np.ones(len(label_arr)), (row, col))
    ).dot(coo_matrix(label_arr)).toarray().astype(int)
    del col
    del row
    cnts_by_fold = np.zeros((n_fold, n_class), int)

    groups_by_fold = [[] for fid in range(n_fold)]
    group_and_cnts = list(enumerate(cnts_by_group))  # pair of aid and cnt by group
    np.random.shuffle(group_and_cnts)
    print("finished preparation", time.time() - start_time)
    for aid, cnt_by_g in sorted(group_and_cnts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for fid in range(n_fold):
            # # eval assignment.
            cnts_by_fold[fid] += cnt_by_g
            fold_eval = (cnts_by_fold / cnts_by_class).std(axis=0).mean()
            cnts_by_fold[fid] -= cnt_by_g

            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = fid

        cnts_by_fold[best_fold] += cnt_by_g
        groups_by_fold[best_fold].append(aid)
    print("finished assignment.", time.time() - start_time)

    gc.collect()
    idx_arr = np.arange(n_train)
    for fid in range(n_fold):
        val_groups = groups_by_fold[fid]

        val_indexs_bool = np.isin(aid_arr, val_groups)
        train_indexs = idx_arr[~val_indexs_bool]
        val_indexs = idx_arr[val_indexs_bool]

        print("[fold {}]".format(fid), end=" ")
        print("n_group: (train, val) = ({}, {})".format(
            n_group - len(val_groups), len(val_groups)), end=" ")
        print("n_sample: (train, val) = ({}, {})".format(
            len(train_indexs), len(val_indexs)))

        yield train_indexs, val_indexs


class KFoldSpliter:
        
    def __init__(self, split_method="KFold", n_fold=5, seed=42):
        
        self.method = split_method
        self.n_fold = n_fold
        self.seed = seed
        
    def split(self, X, y=None, group=None):
        
        if self.method == "KFold":
            kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=self.seed)
            splits =  list(kf.split(X))

        elif self.method == "GroupKFold":
            X_sh, y_sh, group_sh = skl_shuffle(X, y, group, random_state=self.seed)
            gkf = GroupKFold(self.n_fold)
            splits = [
                (X_sh.index[tr_idx].values, X_sh.index[val_idx].values,)
                for tr_idx, val_idx in gkf.split(X_sh, y_sh, group_sh)]
        
        elif self.method == "StratifiedKFold":
            kf = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=self.seed)
            splits = list(kf.split(X, y))
        
        elif self.method == "StratifiedGroupKFold":
            y_onehot = np.zeros((len(y), len(set(y))), dtype=int)
            y_onehot[range(len(y)), y] = 1
            splits = list(multi_label_stratified_group_k_fold(y_onehot, group, self.n_fold, self.seed))
        else:
            raise NotImplementedError

        return splits

