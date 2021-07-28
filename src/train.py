# -*- coding: utf-8 -*- #
"""Training Code"""
import os
import shutil
import warnings
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd

import torch
from torch.cuda import amp
import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.config import Config
from pytorch_pfn_extras.training import extensions as ppe_exts

from utils import to_device, set_random_seed, load_yaml_file

import global_config as CFG

warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)

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
    
    if cfg["/globals/task"] == "multi-class":
        train_path_label["labels"] = torch.from_numpy(train["target"].values)
        val_path_label["labels"] = torch.from_numpy(val["target"].values)
    
    elif cfg["/globals/task"] == "regression":
        train_path_label["labels"] = torch.from_numpy(train[["target"]].values.astype("f"))
        val_path_label["labels"] = torch.from_numpy(val[["target"]].values.astype("f"))
    
    else:
        raise NotImplementedError

    return train_path_label, val_path_label, train_idx, val_idx


def get_eval_func(cfg, model, device):
    
    def eval_func(**batch):
        """Run evaliation for val or test. This function is applied to each batch."""
        batch = to_device(batch, device)
        # model.eval()
        x = batch["data"]
        with amp.autocast(cfg["/globals/enable_amp"]): 
            y = model(x)
        # return batch, y  # input of metrics
        return y.detach().cpu().to(torch.float32)

    return eval_func


def mixup_data(use_mixup, x, t, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if not use_mixup:
        return x, t, None, None
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    t_a, t_b = t, t[index]
    return mixed_x, t_a, t_b, lam


def get_mixup_criterion(use_mixup, loss_func):

    def mixup_criterion(pred, t_a, t_b, lam):
        return lam * loss_func(pred, t_a) + (1 - lam) * loss_func(pred, t_b)

    def mono_criterion(pred, t_a, t_b, lam):
        return loss_func(pred, t_a)
    
    if use_mixup:
        return mixup_criterion
    else:
        return mono_criterion


def train_one_fold(cfg):
    """Run Training on one fold"""
    print(cfg["/globals"])
    print(cfg["!/model"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["/globals/cuda_visible_devices"])
    torch.backends.cudnn.benchmark = cfg["/globals/cudnn_benchmark"]
    set_random_seed(cfg["/globals/seed"], cfg["/globals/deterministic"])
    
    # # prepare train/valid image paths and labels
    train_path_label, val_path_label, _, _ = get_path_label(cfg)
    print("train: {}, val: {}".format(
        len(train_path_label["paths"]), len(val_path_label["paths"])))
   
    cfg["/dataset/train"].lazy_init(**train_path_label)
    cfg["/dataset/val"].lazy_init(**val_path_label)
    train_loader = cfg["/loader/train"]
    val_loader = cfg["/loader/val"]

    device = torch.device(cfg["/globals/device"])
    
    model = cfg["/model"]
    if cfg["/globals/freeze_backbone"]:
        for param in model.backbone.parameters():
            param.requires_grad = False
    model.to(device)

    optimizer = cfg["/optimizer"]
    loss_func = cfg["/loss"]
    loss_func.to(device)
    
    manager = cfg["/manager"]
    # # add extensions
    for ext in cfg["/extensions"]:
        if isinstance(ext, list) or isinstance(ext, tuple):
            manager.extend(*ext)
        elif isinstance(ext, dict):
            manager.extend(**ext)
        else:
            manager.extend(ext)

    evaluator = ppe_exts.Evaluator(
        val_loader, model, eval_func=get_eval_func(cfg, model, device),
        metrics=cfg["/eval"], progress_bar=True)
    manager.extend(evaluator, trigger=(1, "epoch"))

    mixup_enabled = cfg["/dataset/mixup/enabled"]
    mixup_start, mixup_end = cfg["/dataset/mixup/period"]
    mixup_alpha = cfg["/dataset/mixup/alpha"]

    scaler = amp.GradScaler(enabled=cfg["/globals/enable_amp"])
    grad_accum = cfg["/globals/grad_accum"]

    optimizer.zero_grad()
    while not manager.stop_trigger:
        use_mixup = mixup_enabled and mixup_start <= manager.epoch < mixup_end
        # model.train()
        for batch in train_loader:
            with manager.run_iteration():
                batch = to_device(batch, device)
                x, t = batch["data"], batch["target"]
                # # mixup
                mixed_x, t_a, t_b, lam = mixup_data(
                    use_mixup, x, t, device, mixup_alpha)
                criterion = get_mixup_criterion(use_mixup, loss_func)
                # # forward
                with amp.autocast(scaler.is_enabled()):
                    y = model(mixed_x)
                    loss = criterion(y, t_a, t_b, lam)
                    loss = torch.div(loss, grad_accum)
                # # gradient accumulation
                scaler.scale(loss).backward()
                if (manager.iteration + 1) % grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                ppe.reporting.report({'train/loss': loss.item() * grad_accum})


def main():
    """Main."""
    # # parse comand
    usage_msg = """
\n  python {0} --config_file_path <str>\n
""".format(__file__,)
    parser = ArgumentParser(prog="train.py", usage=usage_msg)
    parser.add_argument(
        "-cfg", "--config_file_path", dest="config_file_path", default="./config.yml")
    parser.add_argument("-v", dest="val_folds", default="0,1,2,3,4")
    parser.add_argument("-f", dest="force_mkdir", action="store_true")

    argvs = parser.parse_args()
    config_file_path = Path(argvs.config_file_path)
    pre_eval = load_yaml_file(config_file_path)

    output_root = Path(pre_eval["globals"]["output_root"]).resolve()
    output_root.mkdir(exist_ok=True)

    shutil.copyfile(CFG.SRC / __file__, output_root / "train.py")
    shutil.copyfile(CFG.SRC / "data.py", output_root / "data.py")
    shutil.copyfile(CFG.SRC / "model.py", output_root / "model.py")
    shutil.copyfile(config_file_path, output_root / "config.yml")
    shutil.copyfile(CFG.SRC / "global_config.py", output_root / "global_config.py")

    val_folds = list(map(int, argvs.val_folds.split(",")))
    for val_fold in val_folds:
        print(val_fold)
        pre_eval["globals"]["val_fold"] = val_fold
        cfg = Config(pre_eval, types=CFG.CONFIG_TYPES)

        output_path = Path(cfg["/globals/output_path"]).resolve()
        output_path.mkdir(exist_ok=argvs.force_mkdir)

        train_one_fold(cfg)
        pre_eval["globals"]["val_fold"] = None


if __name__ == "__main__":
    main()
