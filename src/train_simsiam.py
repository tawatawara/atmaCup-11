# -*- coding: utf-8 -*- #
"""Pre-Training"""
import os
import shutil
import warnings
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd

import torch
import torch.nn as nn
from torch.cuda import amp
import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.config import Config

from utils import to_device, set_random_seed, load_yaml_file

import global_config as CFG

warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)


def get_path_label(cfg: Config):
    """Get file path and target info."""
    train_all = pd.read_csv(CFG.DATA / "train.csv")
    test = pd.read_csv(CFG.DATA / "test.csv")
    test["target"] = -1
    all_data = pd.concat([train_all, test], axis=0, ignore_index=True)
    
    train_path_label = {
        "paths": [CFG.IMAGES / f"{o_id}.jpg" for o_id in all_data["object_id"].values],
        "labels": [-1] * len(all_data)  # dummy    
    }
    return train_path_label


def train_sim_siam(cfg):
    """Self Supervised Training by SimSiam"""
    print(cfg["/globals"])
    print(cfg["!/model"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["/globals/cuda_visible_devices"])
    torch.backends.cudnn.benchmark = cfg["/globals/cudnn_benchmark"]
    set_random_seed(cfg["/globals/seed"], cfg["/globals/deterministic"])

    train_path_label  = get_path_label(cfg)
    print("train: {}".format(len(train_path_label["paths"])))
    cfg["/dataset/train"].lazy_init(**train_path_label)
    train_loader = cfg["/loader/train"]

    device = torch.device(cfg["/globals/device"])
    model = cfg["/model"]
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

    scaler = amp.GradScaler(enabled=cfg["/globals/enable_amp"])
    grad_accum = cfg["/globals/grad_accum"]

    optimizer.zero_grad()
    while not manager.stop_trigger:
        # model.train()
        for batch in train_loader:
            with manager.run_iteration():
                x0 = to_device(batch["data0"], device)
                x1 = to_device(batch["data1"], device)

                # # forward
                with amp.autocast(scaler.is_enabled()):
                    out0, out1 = model(x0, x1)
                    loss = loss_func(out0, out1)
                    loss = torch.div(loss, grad_accum)
                
                # # gradient accumulation
                scaler.scale(loss).backward()
                if (manager.iteration + 1) % grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                ppe.reporting.report({'train/loss': loss.item() * grad_accum})

                # # check mode cllapse
                output = out0[0].detach()
                output = nn.functional.normalize(output, dim=1)
                output_std = torch.std(output, 0).mean()
                ppe.reporting.report({'train/std': output_std.item()})


def main():
    """Run SimSiam Training"""
    usage_msg = """
\n  python {0} --config_file_path <str>\n
""".format(__file__,)

    parser = ArgumentParser(prog="train_simsiam.py", usage=usage_msg)
    parser.add_argument("-cfg", dest="config_file_path", default="config.yml")
    parser.add_argument("-f", dest="force_mkdir", action="store_true")

    argvs = parser.parse_args()
    config_file_path = Path(argvs.config_file_path)
    pre_eval = load_yaml_file(config_file_path)
    cfg = Config(pre_eval, types=CFG.CONFIG_TYPES)
    output_path = Path(cfg["/globals/output_path"]).resolve()

    if output_path.parent.exists():
        output_path.mkdir(exist_ok=argvs.force_mkdir)
    else:
        output_path.mkdir(parents=True)
    
    shutil.copyfile(CFG.SRC / "data.py", output_path / "data.py")
    shutil.copyfile(CFG.SRC / "model.py", output_path / "model.py")
    shutil.copyfile(CFG.SRC / "global_config.py", output_path / "global_config.py")
    shutil.copyfile(CFG.SRC / __file__, output_path / "train.py")
    shutil.copyfile(config_file_path, output_path / "config.yml")
    
    train_sim_siam(cfg)


if __name__ == "__main__":
    main()
