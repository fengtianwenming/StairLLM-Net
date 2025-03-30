# -*- coding: utf-8 -*-
"""
@File    :   StairLLM_Net.py
@Time    :   2025/03/26 23:51:57
@Author  :   Jintong Sun
@email :   213212555@seu.edu.cn
"""

"""
# -*- coding:utf-8 -*-
@Project : PvPowerSeries
@File : AutoModel.py
@Author : Sun Jintong
@Time : 2025/2/25 02:09
@Email: 213212555@seu.edu.cn
"""


import torch
import torch.nn as nn
from .Components import get_llm_model, WaveletPatchEmbedding, FFNDecoder, AutoEncoder
from .Config import StairLLMConfig


class StairLLM_Net(nn.Module):
    def __init__(self, config: StairLLMConfig):
        super().__init__()
        self.config: StairLLMConfig = config
        self.ae = AutoEncoder(config.weather_channels, config.seq_len + config.pred_len, out_channels=config.embed_channels)
        self.patch_embedding: WaveletPatchEmbedding = WaveletPatchEmbedding(
            patch_len=config.patch_len,
            d_model=config.llm_dim,
            p_seq_len=config.seq_len,
            w_seq_len=config.seq_len + config.pred_len,
            wave_len=config.wavelet_len,
            input_channels=config.embed_channels,
        )
        self.llm_model, self.tokenizer = get_llm_model(
            model_name=self.config.llm_model_name,
            model_root=self.config.llm_model_root,
            fine_tune_layernorm=True,
        )

        self.twin_num: int = config.seq_len // config.patch_len
        match config.llm_model_name:
            case "gpt2small":
                decoder_num = 4
            case "gpt2xl":
                decoder_num = 6
            case "Qwen1.5B":
                decoder_num = 8
            case _:
                raise NotImplementedError()
        self.decoder_list = nn.ModuleList(
            [
                FFNDecoder(
                    d_model=config.llm_dim,
                    twin_num=self.twin_num,
                    p=config.dropout,
                    s=2 * self.twin_num + 3,
                )
                for _ in range(decoder_num)
            ]
        )

        self.pred_head = nn.Linear(
            in_features=3 * self.config.llm_dim, out_features=self.config.pred_len
        )

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor | None, wa: torch.Tensor):
        batch_size = tgt.size(0)
        memory = self.ae(memory)
        x = self.patch_embedding(p=tgt, w=memory, wa=wa)
        hidden_states = self.llm_model(
            inputs_embeds=x, output_hidden_states=True
        ).hidden_states

        b = len(hidden_states)
        a = (b - 1) // (len(self.decoder_list) - 1)
        b = a * (len(self.decoder_list) - 1) + 1

        for idx in range(len(self.decoder_list)):
            module = self.decoder_list[idx]
            hidden_idx = a * idx - b
            x = x + hidden_states[hidden_idx]
            x = module(x)

        x = x[:, -3:, :].reshape(batch_size, -1, 3 * self.config.llm_dim)
        x = self.pred_head(x)
        x = x.reshape(batch_size, -1)
        return x
