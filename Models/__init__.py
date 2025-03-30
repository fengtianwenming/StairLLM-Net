# -*- coding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2025/03/26 23:52:47
@Author  :   Jintong Sun
@email :   213212555@seu.edu.cn
'''
import torch
from .DataModule import DataModule
from .Config import StairLLMConfig
from .Automodel import StairLLMAutoModel, NoBugTQDMBar

__all__ = [
    "DataModule",
    "StairLLMConfig",
    "StairLLMAutoModel",
    "NoBugTQDMBar",
]

torch.set_float32_matmul_precision('medium')