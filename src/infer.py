# -*- coding: utf-8 -*- #
"""Inference Code"""
import os
import gc
import shutil
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

import torch
from pytorch_pfn_extras.config import Config
from utils import to_device, load_yaml_file

import global_config as CFG

def softmax(x):
    p = np.exp(x - x.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def get_path_label(cfg: Config):
    """Get file path and target info."""
    train_all = pd.read_csv(CFG.DATA / cfg["/globals/meta_file"])
    use_fold = cfg["/globals/val_fold"]

    train = train_all[train_all["fold"] != use_fold]
    val = train_all[train_all["fold"] == use_fold]
    train_idx = train.index.values
    val_idx = val.index.values

    train_path_label = {
        "paths": [CFG.IMAGES / f"{o_id}.jpg" for o_id in train["object_id"].values]}
    val_path_label = {
        "paths": [CFG.IMAGES / f"{o_id}.jpg" for o_id in val["object_id"].values]}
    
    # if cfg["/globals/task"] == "multi-class":
    #     train_path_label["labels"] = torch.from_numpy(train["target"].values)
    #     val_path_label["labels"] = torch.from_numpy(val["target"].values)
    # elif cfg["/globals/task"] == "regression":
    train_path_label["labels"] = torch.from_numpy(train[["target"]].values.astype("f"))
    val_path_label["labels"] = torch.from_numpy(val[["target"]].values.astype("f"))

    return train_path_label, val_path_label, train_idx, val_idx


def run_inference_loop(cfg, model, loader, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x = to_device(batch["data"], device)
            y = model(x)
            pred_list.append(y.detach().cpu().numpy())
        
        pred_arr = np.concatenate(pred_list)
        del pred_list
    return pred_arr


def infer(exp_dir_path, criteria, gpu_ids):
    """Main"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    test = pd.read_csv(CFG.DATA / "test.csv")
    test_path_label = {
        "paths": [CFG.IMAGES / f"{o_id}.jpg" for o_id in test["object_id"].values],
        "labels": [-1] * len(test)
    }
    labels_arr = pd.read_csv(CFG.DATA / "train.csv")["target"].values
    n_train = len(labels_arr)
    n_test = len(test)
    oof_pred_arr = np.zeros((n_train, 1))
    test_pred_arr = np.zeros((CFG.N_FOLDS, n_test, 1))
    oof_logit_arr = np.zeros((n_train, CFG.N_CLASSES))
    test_logit_arr = np.zeros((CFG.N_FOLDS, n_test, CFG.N_CLASSES))
    score_list = []
    
    pre_eval = load_yaml_file(exp_dir_path / "config.yml")
    pre_eval["model"]["pretrained"] = False

    for fold_id in range(CFG.N_FOLDS):
        print(f"[fold {fold_id} for {exp_dir_path.name}] ")
        fold_path = exp_dir_path / f"fold{fold_id}"
        model_path = exp_dir_path / f"best_{criteria}_model_fold{fold_id}.pth"
        
        # # extract best result
        log_df = pd.read_json(fold_path / "log")
        if criteria == "metric":
            # best_record = log_df.iloc[log_df["val/metric"].idxmax()]
            best_record = log_df.iloc[log_df["val/metric"].idxmin()]  # for RMSE
        else:
            best_record = log_df.iloc[log_df["val/loss"].idxmin()]
        best_epoch = int(best_record["epoch"])
        print(
            f"[fold{fold_id} - {criteria}] epoch: {best_epoch}, "
            f"loss: {best_record['val/loss']:.5f}, metric: {best_record['val/metric']:.4f}")
        if not model_path.exists():
            print("copy best model")
            shutil.copyfile(fold_path / f"snapshot_by_{criteria}_epoch_{best_epoch}.pth", model_path)
            for p in fold_path.glob(f"snapshot_by_{criteria}_epoch_*.pth"):
                p.unlink()
        else:
            print("best model was already copied")
        
        pre_eval["globals"]["val_fold"] = fold_id 
        cfg = Config(pre_eval, types=CFG.CONFIG_TYPES)

        _, val_path_label, _, val_index = get_path_label(cfg)
        # # # get data_loader
        cfg["/dataset/val"].lazy_init(**val_path_label)
        cfg["/dataset/test"].lazy_init(**test_path_label)
        val_loader = cfg["/loader/val"]
        test_loader = cfg["/loader/test"]
        print(f"val_loader: {len(val_loader)}, test_laoder: {len(test_loader)}")

        # # # get model
        model = cfg["/model"]
        model.load_state_dict(torch.load(model_path, map_location=device))

        val_pred = run_inference_loop(cfg, model, val_loader, device)
        test_pred = run_inference_loop(cfg, model, test_loader, device)

        if cfg["/globals/task"] == "multi-class":
            n_classes = val_pred.shape[1]
            oof_logit_arr[val_index] = val_pred
            val_prob = softmax(val_pred)
            val_pred = (val_prob * np.arange(n_classes)).sum(axis=1)[:, None]
            test_logit_arr[fold_id] = test_pred
            test_prob = softmax(test_pred)
            test_pred = (test_prob * np.arange(n_classes)).sum(axis=1)[:, None]
        
        val_pred = np.clip(val_pred, 0, 3)
        test_pred = np.clip(test_pred, 0, 3)

        oof_pred_arr[val_index] = val_pred
        test_pred_arr[fold_id] = test_pred
        
        val_labels = labels_arr[val_index]
        score = mean_squared_error(val_labels, val_pred, squared=False)
        loss = mean_squared_error(val_labels, val_pred)
        score_list.append([fold_id, best_epoch, loss, score])
        print(f"val loss: {loss:.5f}, val metric: {score:.5f}")

        del model, val_loader, test_loader
        pre_eval["globals"]["val_fold"] = None
        torch.cuda.empty_cache()
        gc.collect()

    oof_score = mean_squared_error(labels_arr, oof_pred_arr, squared=False)
    oof_loss = mean_squared_error(labels_arr, oof_pred_arr)
    print(f"oof loss: {oof_loss:.5f}, oof metric: {oof_score:.5f}")
    
    score_list.append(["oof", -1, oof_loss, oof_score])
    score_df = pd.DataFrame(
        score_list,
        columns=["fold", "epoch", "loss", "metric"])
    score_df.to_csv(exp_dir_path / f"score_by_best_{criteria}.csv", index=False)

    np.save(
        exp_dir_path / f"oof_prediction_by_best_{criteria}.npy", oof_pred_arr)

    if cfg["/globals/task"] == "multi-class":
        np.save(exp_dir_path / f"oof_logit_by_best_{criteria}.npy", oof_logit_arr)
        np.save(exp_dir_path / f"test_logit_by_best_{criteria}.npy", test_logit_arr)

    for fold_id in range(CFG.N_FOLDS):
        sub_df = pd.DataFrame({"target": [-1] * len(test)})
        sub_df["target"] = test_pred_arr[fold_id]
        sub_df.to_csv(exp_dir_path / f"submission_by_best_{criteria}_fold{fold_id}.csv", index=False)

    sub_df = pd.DataFrame({"target": [-1] * len(test)})
    if cfg["/globals/task"] == "multi-class":
        test_probs = np.zeros_like(test_logit_arr)
        for fold_id in range(CFG.N_FOLDS):
            test_probs[fold_id] = softmax(test_logit_arr[fold_id])
        sub_df["target"] = (test_probs.mean(axis=0) * np.arange(CFG.N_CLASSES)).sum(axis=1)
    else:
        sub_df["target"] = test_pred_arr.mean(axis=0)
    sub_df.to_csv(exp_dir_path / f"submission_by_best_{criteria}.csv", index=False)


def main():
    """Main."""
    # # parse comand
    usage_msg = """
\n  python {0} --exp_dir_path <str>\n
""".format(__file__,)
    parser = ArgumentParser(prog="train.py", usage=usage_msg)
    parser.add_argument("-e", "--exp_dir_path", dest="exp_dir_path", required=True)
    parser.add_argument("-cri", "--criteria", dest="criteria", default="metric")
    parser.add_argument("-g", dest="gpu_ids", default="0")
    argvs = parser.parse_args()

    assert argvs.criteria in ["loss", "metric"]

    exp_dir_path = Path(argvs.exp_dir_path).resolve()
    infer(exp_dir_path, argvs.criteria, argvs.gpu_ids)


if __name__ == "__main__":
    main()
