# -*- coding: utf-8 -*- #
from torch import optim
from torch.optim import lr_scheduler

CONFIG_TYPES = {
    # # Optimizer
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,

    # # Scheduler
    "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
    "OneCycleLR": lr_scheduler.OneCycleLR,
    "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts
}