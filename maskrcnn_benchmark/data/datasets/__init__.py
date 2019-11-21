# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .abstract import AbstractDataset
from .cityscapes import CityScapesDataset
from .coco import COCODataset
from .concat_dataset import ConcatDataset
from .mosquitoes import MosquitoDataset
from .mosquitoes_coco import MosquitoesCOCODataset
from .voc import PascalVOCDataset

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
# noqa F401 isort:skip

__all__ = [
    "COCODataset",
    "ConcatDataset",
    "PascalVOCDataset",
    "AbstractDataset",
    "CityScapesDataset",
    "MosquitoDataset",
    "MosquitoesCOCODataset",
]
