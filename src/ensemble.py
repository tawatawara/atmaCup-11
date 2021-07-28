# -*- coding: utf-8 -*- #
"""Ensemble Code"""
import os
import shutil
import warnings
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import optuna

from utils import set_random_seed, load_yaml_file
import global_config as CFG

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_objective(oof_arr_list, label_arr):
    n_models= len(oof_arr_list)
    
    def opbjective(trial):
        blend_arr = np.zeros_like(oof_arr_list[0])
        for i in range(n_models):
            w = trial.suggest_discrete_uniform(f'w{i}', 0, 1, 0.001)
            blend_arr += oof_arr_list[i] * w
        
        return mean_squared_error(label_arr, blend_arr, squared=False)
    
    return opbjective


def main():
    """Main"""
    # # parse comand
    usage_msg = """
\n  python {0} --config_file_path <str>\n
""".format(__file__,)
    parser = ArgumentParser(prog="ensemble.py", usage=usage_msg)
    parser.add_argument("-cfg", "--config_file_path", dest="config_file_path")
    argvs = parser.parse_args()

    cfg = load_yaml_file(argvs.config_file_path)
    output_path = Path(cfg["globals"]["output_path"])
    output_path.mkdir(exist_ok=True)
    set_random_seed(cfg["globals"]["seed"])

    score_list = []
    train_all = pd.read_csv(CFG.DATA /  "train.csv")
    label_arr = train_all[CFG.CLASSES].values
    smpl_sub = pd.read_csv(CFG.DATA / "atmaCup#11_sample_submission.csv")

    exp_dirs_cls = [Path(d) for d in cfg["classification_dirs"]]
    exp_dirs_reg = [Path(d) for d in cfg["regression_dirs"]]

    oof_list_cls = [
        np.load(p / "oof_prediction_by_best_metric.npy") for p in exp_dirs_cls]
    oof_list_reg = [
        np.load(p / "oof_prediction_by_best_metric.npy") for p in exp_dirs_reg]
    sub_list_cls = [
        pd.read_csv(p / "submission_by_best_metric.csv") for p in exp_dirs_cls]
    sub_list_reg = [
        pd.read_csv(p / "submission_by_best_metric.csv") for p in exp_dirs_reg]

    # # averaging classification models
    oof_avg_cls = np.zeros_like(oof_list_cls[0])
    for oof in oof_list_cls:
        oof_avg_cls += oof
    oof_avg_cls /= len(oof_list_cls)
    score_avg_cls = mean_squared_error(label_arr, oof_avg_cls, squared=False)
    score_list.append(score_avg_cls)

    sub_avg_cls = smpl_sub.copy()
    sub_avg_cls[CFG.CLASSES] = 0.
    for sub in sub_list_cls:
        sub_avg_cls[CFG.CLASSES] += sub
    sub_avg_cls[CFG.CLASSES] /= len(sub_list_cls)
    sub_avg_cls.to_csv(output_path / "submission_avg_cls.csv", index=False)

    print(f"[avg-cls] {score_avg_cls:.4f}")

    # # averaging regression models
    oof_avg_reg = np.zeros_like(oof_list_reg[0])
    for oof in oof_list_reg:
        oof_avg_reg += oof
    oof_avg_reg /= len(oof_list_reg)
    score_avg_reg = mean_squared_error(label_arr, oof_avg_reg, squared=False)
    score_list.append(score_avg_reg)

    sub_avg_reg = smpl_sub.copy()
    sub_avg_reg[CFG.CLASSES] = 0.
    for sub in sub_list_reg:
        sub_avg_reg[CFG.CLASSES] += sub
    sub_avg_reg[CFG.CLASSES] /= len(sub_list_reg)
    sub_avg_reg.to_csv(output_path / "submission_avg_reg.csv", index=False)

    print(f"[avg-reg] {score_avg_reg:.4f}")

    # # averaging classification & regression models
    oof_avg = (oof_avg_cls + oof_avg_reg) / 2
    score_avg = mean_squared_error(label_arr, oof_avg, squared=False)
    score_list.append(score_avg)

    sub_avg = smpl_sub.copy()
    sub_avg[CFG.CLASSES] = (sub_avg_cls + sub_avg_reg) / 2
    sub_avg.to_csv(output_path / "submission_avg_all.csv", index=False)

    print(f"[avg-all] {score_avg:.4f}")

    # # weight optimization
    oof_list = oof_list_cls + oof_list_reg
    sub_list = sub_list_cls + sub_list_reg

    sampler = optuna.samplers.TPESampler(seed=cfg["globals"]["seed"])
    study = optuna.create_study(direction="minimize", sampler=sampler)
    obj = get_objective(oof_list, label_arr)
    study.optimize(obj, n_trials=1000, show_progress_bar=True)

    score_wopt = study.best_value
    score_list.append(score_wopt)
    weights =[study.best_params[f"w{i}"] for i in range(len(oof_list))]

    sub_wopt = smpl_sub.copy()
    sub_wopt[CFG.CLASSES] = 0.
    for w, sub in zip(weights, sub_list):
        sub_wopt[CFG.CLASSES] += w * sub[CFG.CLASSES]
    sub_wopt.to_csv(output_path / "submission_weight_optimization_all.csv", index=False)

    print(f"[wopt-all] {score_wopt:.4f}")

    score_df = pd.DataFrame(
        score_list,
        index=["avg-cls", "avg-reg", "avg-all", "wopt-all"],
        columns=["oof-rmse"])
    score_df.to_csv(output_path / "ensemble_scores.csv")


if __name__ == "__main__":
    main()