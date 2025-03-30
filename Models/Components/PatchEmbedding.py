# -*- coding: utf-8 -*-
'''
@File    :   PatchEmbedding.py
@Time    :   2025/03/27 00:19:10
@Author  :   Jintong Sun
@email :   213212555@seu.edu.cn
'''


import torch
import torch.nn as nn
import warnings

class PowerEmbedding(nn.Module):
    def __init__(self, patch_len:int, d_model:int, seq_len:int):
        super().__init__()
        self.patch_len:int = patch_len
        self.norm = nn.LayerNorm(seq_len)
        self.d_model:int = d_model
        self.high_dim_map = nn.Linear(self.patch_len, self.d_model)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        :param x: torch.tensor shape:(batch_size, seq_len)
        :return: patched tensor shape:(batch, seq_len//patch_len, d_model)
        """
        batch_size, seq_len = x.shape
        padding_length:int = (self.patch_len - (seq_len % self.patch_len)) % self.patch_len
        x = self.norm(x)
        if padding_length > 0:
            padding = torch.zeros(batch_size, padding_length, dtype=x.dtype)
            x = torch.cat((x, padding), dim=1)
            # warnings.warn('Use zero padding in PatchEmbedding...')
        x = x.reshape(batch_size, -1, self.patch_len)
        x = self.high_dim_map(x)
        return x


class WeatherEmbedding(nn.Module):
    def __init__(self, patch_len:int, d_model:int, seq_len:int, input_channels:int = 4):
        super().__init__()
        self.patch_len:int = patch_len
        self.d_model:int = d_model
        self.input_channels:int = input_channels
        self.seq_len:int = seq_len
        self.patch_num:int = self.seq_len // self.patch_len
        self.norm = nn.LayerNorm([input_channels, seq_len])
        self.high_dim_map = nn.Linear(self.patch_len * input_channels, self.d_model)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        :param x: shape (batch_size, input_channels, seq_len) The weather data
        :return: The weather embedding shape:(batch_size, seq_len//patch_len, d_model)
        """
        batch_size, input_channels, seq_len = x.shape
        padding_length: int = (self.patch_len - (seq_len % self.patch_len)) % self.patch_len
        x = self.norm(x)
        if padding_length > 0:
            padding = torch.zeros(batch_size, input_channels, padding_length, dtype=x.dtype)
            x = torch.cat((x, padding), dim=2)
            warnings.warn('Use zero padding in PatchEmbedding...')
        x = x.permute(0, 2, 1).reshape(batch_size, -1, self.patch_len * self.input_channels)
        x = self.high_dim_map(x)
        return x


class WaveletPatchEmbedding(nn.Module):
    def __init__(self, patch_len:int, d_model:int, p_seq_len:int, w_seq_len:int, wave_len:int, input_channels:int = 4, p:float = 0.1):
        super().__init__()
        self.patch_len:int = patch_len
        self.d_model:int = d_model
        self.input_channels:int = input_channels
        self.p_seq_len:int = p_seq_len
        self.w_seq_len: int = w_seq_len
        self.power_emb = PowerEmbedding(patch_len=patch_len, d_model=d_model, seq_len=p_seq_len)
        self.wave_emb = nn.Linear(wave_len, d_model*2)
        self.weather_emb = WeatherEmbedding(patch_len=patch_len, d_model=d_model, seq_len=w_seq_len,
                                            input_channels=input_channels)
        self.dropout_p = nn.Dropout(p)
        self.dropout_wa = nn.Dropout(p)
        self.dropout_w = nn.Dropout(p)

    def forward(self, p:torch.Tensor, w:torch.Tensor,wa:torch.Tensor) -> torch.Tensor:
        patch_p = self.dropout_p(self.power_emb(p))
        patch_wa = self.dropout_wa(self.wave_emb(wa))
        patch_w = self.dropout_w(self.weather_emb(w))
        batch_size, p_len, patch_dim = patch_p.shape
        _, w_len, _ = patch_w.shape
        assert p_len < w_len
        w_first_part = patch_w[:, :p_len, :]  # shape: [B, p_len, patch_dim]
        interleaved = torch.stack([w_first_part, patch_p], dim=2)  # [B, p_len, 3, patch_dim]
        interleaved = interleaved.view(batch_size, p_len * 2, patch_dim)  # [B, 2*p_len, patch_dim]
        w_leftover = patch_w[:, p_len:, :]  # shape: [B, w_len - p_len, patch_dim]
        patch_wa = patch_wa.reshape(batch_size, -1, patch_dim)
        result = torch.cat([interleaved, w_leftover, patch_wa], dim=1)
        return result