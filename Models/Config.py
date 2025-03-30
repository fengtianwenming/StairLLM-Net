# -*- coding: utf-8 -*-
"""
@File    :   Config.py
@Time    :   2025/03/27 00:26:07
@Author  :   Jintong Sun
@email :   213212555@seu.edu.cn
"""

from dataclasses import dataclass


@dataclass
class StairLLMConfig:
    # require
    task_stage: str
    llm_model_name: str
    llm_model_root: str
    dataset_idx: int | list[int]
    dataset_path: str | list[str]
    
    # model params
    seq_len: int = 288
    pred_len: int = 96
    llm_dim: int = 768
    activation: str = "gelu"
    patch_len: int = 96
    dropout: float = 0.2
    weather_channels: int = 9
    wavelet_type: str = "sym4"
    wavelet_len: int = 320
    embed_channels:int = 4

    # dataset params
    use_all_data: bool = False
    dataset_name: str | None = None
    val_dataset_path: str | list[str] | None = None
    val_dataset_idx: int | list[int] | None = None

    # train_params
    lr: float = 1e-3
    swa_lr: float = 1e-3
    ckpt_path: str = "Checkpoint"
    ckpt_name: str = "default.ckpt"
    batch_size: int = 16
    max_epoch: int = 10
    cos_t_max: int = 8
    seed: int = 42
    loss: str = "mae"
    use_val:bool = False
