# -*- coding: utf-8 -*- $
"""Preprocessing"""
from joblib import Parallel, delayed

import numpy as np
import pandas as pd

import cv2
from utils import KFoldSpliter
import global_config as CFG

def get_img_info(img_path):
    """画像サイズ, channel ごとの平均・標準偏差 などを抽出"""
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ch_mean = img.mean(axis=(0, 1)).tolist()
    ch_std = img.std(axis=(0, 1)).tolist()
    h, w = img.shape[:2]
    return [img_path.stem, h, w, *ch_mean, *ch_std]

def preprocess():
    """Run Preprocessing"""
    train = pd.read_csv(CFG.DATA / "train.csv")
    test = pd.read_csv(CFG.DATA / "test.csv")

    # # split fold
    sgkf = KFoldSpliter(
        split_method="StratifiedGroupKFold", n_fold=CFG.N_FOLDS, seed=CFG.RANDAM_SEED)
    train_val_indexs = list(
        sgkf.split(X=train.object_id.values, y=train.target.values, group=train.art_series_id))

    # # # カラム追加
    train["fold"] = -1
    for fold_id, (_, val_idx) in enumerate(train_val_indexs):
        train.loc[val_idx, "fold"] = fold_id

    # # # 確認
    print(pd.pivot_table(
        train, values=["object_id", "art_series_id"],
        index="fold", columns="target", aggfunc="nunique"))

    # # # 保存
    train.to_csv(CFG.DATA / "train_sgkf-5fold.csv", index=False)

    # # 画像サイズなどの抽出
    img_dir = CFG.DATA / "photos"
    img_paths = [
        img_dir / f"{o_id}.jpg" for o_id in train.object_id.values
    ] + [
        img_dir / f"{o_id}.jpg" for o_id in test.object_id.values
    ]
    info_list = Parallel(n_jobs=4, verbose=5)([
        delayed(get_img_info)(img_path) for img_path in img_paths
    ])
    info_df = pd.DataFrame(
        info_list, columns=[
            "object_id", "height", "width",
            "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std"
        ])
    info_df.insert(
        0, "data_type", ["train"] * len(train) + ["test"] * len(test))
    info_df.to_csv(CFG.DATA / "img_info.csv", index=False)

    # # 概算の統計値を算出
    ch_means = info_df[["r_mean", "g_mean", "b_mean"]].values
    ch_means_train = info_df.query("data_type == 'train'")[["r_mean", "g_mean", "b_mean"]].values
    ch_means_test = info_df.query("data_type == 'test'")[["r_mean", "g_mean", "b_mean"]].values

    ch_mean = ch_means.mean(axis=0)
    ch_std = np.sqrt(np.mean((ch_means - ch_mean) ** 2, axis=0))
    ch_mean_train = ch_means_train.mean(axis=0)
    ch_std_train = np.sqrt(np.mean((ch_means_train - ch_mean_train) ** 2, axis=0))
    ch_mean_test = ch_means_test.mean(axis=0)
    ch_std_test = np.sqrt(np.mean((ch_means_test - ch_mean_test) ** 2, axis=0))

    ch_stats_df = pd.DataFrame(
    [
        ch_mean.tolist(), ch_std.tolist(),
        (ch_mean / 255).tolist(), (ch_std / 255).tolist(),
        ch_mean_train.tolist(), ch_std_train.tolist(),
        (ch_mean_train / 255).tolist(), (ch_std_train / 255).tolist(),
        ch_mean_test.tolist(), ch_std_test.tolist(),
        (ch_mean_test / 255).tolist(), (ch_std_test / 255).tolist(),
    ], index=[
        "mean", "std", "mean_norm", "std_norm",
        "mean_train", "std_train", "mean_norm_train", "std_norm_train",
        "mean_test", "std_test", "mean_norm_test", "std_norm_test",
    ], columns=["Red", "Green", "Blue"])

    print(ch_stats_df)
    ch_stats_df.to_csv(CFG.DATA / "stats_by_data.csv")


if __name__ == "__main__":
    preprocess()
