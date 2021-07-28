# -*- coding: utf-8 -*- #
import os
import typing as tp

import numpy as np

import torch
from torch import nn
import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions as exts
from pytorch_pfn_extras.training import triggers as trgrs


Batch = tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor]]
ModelOut = tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor], torch.Tensor]


def micro_average(
    metric_func: nn.Module,
    report_name: str, prefix="val",
    pred_index: int=-1, label_index: int=-1,
    pred_key: str="logit", label_key: str="target",
) -> tp.Callable:
    """Return Metric Wrapper for Simple Mean Metric"""
    metric_sum = [0.]
    n_examples = [0]
    
    def wrapper(batch: Batch, model_output: ModelOut, is_last_batch: bool):
        """Wrapping metric function for evaluation"""
        if isinstance(batch, tuple):
            t = batch[label_index]
        elif isinstance(batch, dict):
            t = batch[label_key]
        else:
            raise NotImplementedError

        if isinstance(model_output, tuple):
            y = model_output[pred_index]
        elif isinstance(model_output, dict):
            y = model_output[pred_key]
        else:
            y = model_output

        metric = metric_func(y, t)
        if isinstance(metric, torch.Tensor):
            metric = metric.item()
        metric_sum[0] += metric * y.shape[0]
        n_examples[0] += y.shape[0]

        if is_last_batch:
            final_metric = metric_sum[0] / n_examples[0]
            ppe.reporting.report({f"{prefix}/{report_name}": final_metric})
            # # reset state
            metric_sum[0] = 0.
            n_examples[0] = 0

    return wrapper


def calc_across_all_batchs(
    metric_func: nn.Module,
    report_name: str, prefix="val",
    pred_index: int=-1, label_index: int=-1,
    pred_key: str="logit", label_key: str="target",
) -> tp.Callable:
    """
    Return Metric Wrapper for Metrics caluculated on all data
    
    storing predictions and labels of evry batch, finally calculating metric on them.
    """
    pred_list = []
    label_list = []
    
    def wrapper(batch: Batch, model_output: ModelOut, is_last_batch: bool):
        """Wrapping metric function for evaluation"""
        if isinstance(batch, tuple):
            t = batch[label_index]
        elif isinstance(batch, dict):
            t = batch[label_key]
        else:
            raise NotImplementedError

        if isinstance(model_output, tuple):
            y = model_output[pred_index]
        elif isinstance(model_output, dict):
            y = model_output[pred_key]
        else:
            y = model_output

        pred_list.append(y.numpy())
        label_list.append(t.numpy())

        if is_last_batch:
            pred = np.concatenate(pred_list, axis=0)
            label = np.concatenate(label_list, axis=0)
            final_metric = metric_func(pred, label)
            ppe.reporting.report({f"{prefix}/{report_name}": final_metric})
            # # reset state
            pred_list[:] = []
            label_list[:] = []

    return wrapper


def str_format(target, args):
    if isinstance(args, tuple) or isinstance(args, list):
        return target.format(*args)
    if isinstance(args, dict):
        return target.format(**args)
    return target.format(args)


CONFIG_TYPES = {
    # # utils
    "__len__": lambda obj: len(obj),
    "__add__": lambda x0, x1: x0 + x1,
    "__sub__": lambda x0, x1: x0 - x1,
    "__mul__": lambda x0, x1: x0 * x1,
    "__div__": lambda x0, x1: x0 / x1,
    "__floordiv__": lambda x0, x1: x0 // x1,
    "str_format": str_format,
    "path_join": lambda args: os.path.join(*args),
    "method_call": lambda obj, method: getattr(obj, method)(),
    "method_call_with_arg": lambda obj, method, arg: getattr(obj, method)(arg),
    "method_call_with_args": lambda obj, method, args: getattr(obj, method)(*args),
    "method_call_with_kwargs": lambda obj, method, kwargs: getattr(obj, method)(**kwargs),

    # # Metric Wrapper
    "micro_average": micro_average,
    "calc_across_all_batchs": calc_across_all_batchs,

    # # PPE Extensions
    "ExtensionsManager": ppe.training.ExtensionsManager,

    "observe_lr": exts.observe_lr,
    "LogReport": exts.LogReport,
    "PlotReport": exts.PlotReport,
    "PrintReport": exts.PrintReport,
    "ProgressBar": exts.ProgressBar,
    "snapshot": exts.snapshot,
    "LRScheduler": exts.LRScheduler, 

    "MinValueTrigger": trgrs.MinValueTrigger,
    "MaxValueTrigger": trgrs.MaxValueTrigger,
    "EarlyStoppingTrigger": trgrs.EarlyStoppingTrigger,
    "ManualScheduleTrigger": trgrs.ManualScheduleTrigger,
}
