# -*- coding-utf8 -*- #
from torch import nn
import timm
import lightly

from .model import get_activation
from .model import Conv1dBNActiv, Conv2dBNActiv, MLP
from .model import TimmBase, BasicImageModel

from .metric import ROCAUC, Accuracy, RMSE


CONFIG_TYPES = {
    # # model
    "timm_create_model": timm.create_model,
    "BasicImageModel": BasicImageModel,

    # # loss
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSELoss": nn.MSELoss,
    "SymNegCosineSimilarityLoss": lightly.loss.SymNegCosineSimilarityLoss,

    # # metric
    "Accuracy": Accuracy,
    "ROCAUC": ROCAUC,
    "RMSE": RMSE,
}
