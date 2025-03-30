# -*- coding: utf-8 -*-
'''
@File    :   AutoEncoder.py
@Time    :   2025/03/26 23:54:18
@Author  :   Jintong Sun
@email :   213212555@seu.edu.cn
'''

import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, val_channels:int, w_len:int, hidden_channels:int = 32, out_channels:int = 4):
        super(AutoEncoder, self).__init__()
        self.val_channels = val_channels
        self.w_len = w_len
        self.layernorm = nn.LayerNorm([val_channels, w_len])
        k = self.val_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_channels, kernel_size=k, stride=1, padding=k // 2),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=k, stride=1, padding=k // 2,
                      groups=hidden_channels),
            nn.GELU()
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels * val_channels, hidden_channels * val_channels // 3),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(hidden_channels * val_channels // 3, out_channels),
            nn.LayerNorm([w_len, out_channels])
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layernorm(x)
        x = self.conv(x).reshape(x.shape[0], -1, self.w_len).permute(0, 2, 1)
        x = self.ffn(x).permute(0, 2, 1)

        return x