# -*- coding: utf-8 -*- #
"""Config for this project"""
from pathlib import Path
from base_model import CONFIG_TYPES as MODEL_TYPES_BASE
from base_data import CONFIG_TYPES as DATA_TYPES_BASE
from base_optimizer import CONFIG_TYPES as OPTIM_TYPES_BASE
from base_pfn_extras import CONFIG_TYPES as PPE_TYPES_BASE


ROOT = Path.cwd().parent
INPUT = ROOT / "input"
DATA = INPUT / "atmaCup-11"
IMAGES = DATA / "photos"
OUTPUT = ROOT / "output"
SRC = ROOT / "src"

N_FOLDS = 5
RANDAM_SEED = 1086

CLASSES = [
    "target"
]
N_CLASSES = 4

CONFIG_TYPES = dict()

CONFIG_TYPES.update(MODEL_TYPES_BASE)
CONFIG_TYPES.update(DATA_TYPES_BASE)
CONFIG_TYPES.update(OPTIM_TYPES_BASE)
CONFIG_TYPES.update(PPE_TYPES_BASE)


# # コンペ特有のもの・追加したものがあればここで追加する.
import lightly
from data import ContrastiveImagePathLabelLazyDataset
from model import RMSEWithLogits, RMSEWithClip

TYPES_ADHOC = {
    "RMSEWithLogits": RMSEWithLogits,
    "RMSEWithClip": RMSEWithClip,
    "LightlySimSiam": lightly.models.SimSiam,
    "ContrastiveImagePathLabelLazyDataset": ContrastiveImagePathLabelLazyDataset,
}
CONFIG_TYPES.update(TYPES_ADHOC)
