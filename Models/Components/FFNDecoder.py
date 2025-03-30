# -*- coding: utf-8 -*-
'''
@File    :   FFNDecoder.py
@Time    :   2025/03/26 23:56:56
@Author  :   Jintong Sun
@email :   213212555@seu.edu.cn
'''

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    一个简单的前馈网络：
    FFN(x) = Dropout(激活( xW_1 + b_1 ))W_2 + b_2
    """
    def __init__(self, embed_dim, ffn_dim, dropout=0.1, activation='relu'):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"The activation {self.activation} is not supported")

    def forward(self, x) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class SeModule1d(nn.Module):
    def __init__(self, in_size, reduction=2):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_size, in_size // reduction, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm1d(in_size // reduction),
            nn.GELU(),
            nn.Conv1d(in_size // reduction, in_size, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm1d(in_size),
            nn.Sigmoid()
        )

    def forward(self, x) -> torch.Tensor:
        return x * self.se(x)

class TokenCrossModule(nn.Module):
    def __init__(self, embed_dim:int, group:int, s:int, p:float = 0.1):
        super().__init__()
        assert embed_dim % group == 0
        self.group = group
        self.linear1 = nn.Linear(s * embed_dim // group, embed_dim)
        self.linear2 = nn.Linear(embed_dim, s * embed_dim // group)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=p)
        self.se = SeModule1d(in_size=group)

    def forward(self, x:torch.Tensor):
        b, s, _ = x.shape
        x = x.reshape(b, s, self.group, -1).permute(0, 2, 1, 3).reshape(b, self.group, -1)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.norm(x)
        x = self.se(x)
        x = self.linear2(x)
        x = x.reshape(b, self.group, s, -1).permute(0, 2, 1, 3).reshape(b, s, -1)
        x = self.dropout(x)
        return x

class FFNDecoder(nn.Module):
    def __init__(self, d_model:int, twin_num:int, p=0.1, s:int|None = None, group:int = 8):
        super().__init__()
        self.d_model: int = d_model
        self.twin_num: int = twin_num
        self.norm = nn.LayerNorm(d_model)
        self.p_ff = FeedForward(d_model, int(1.414 * d_model), p, activation='gelu')
        self.w_ff = FeedForward(d_model, int(1.414 * d_model), p, activation='gelu')
        if s is None:
            self.s = 2 * twin_num + 1
        else:
            self.s = s
        self.crosser = TokenCrossModule(d_model, group, s=self.s)
        self.se = SeModule1d(in_size=self.s, reduction=self.s//3)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        res = x
        x = x + self.crosser(x)
        x= self.se(x)
        x_out = torch.zeros_like(x).to(x)
        x = self.norm(x)
        x_twin_w = self.w_ff(x[:, 0:2*self.twin_num:2, :])
        x_twin_p = self.p_ff(x[:, 1:2*self.twin_num:2, :])
        x_left = self.w_ff(x[:, 2*self.twin_num:, :])
        x_out[:, 0:2 * self.twin_num:2, :] = x_twin_w
        x_out[:, 1:2 * self.twin_num:2, :] = x_twin_p
        x_out[:, 2 * self.twin_num:, :] = x_left
        x_out = x_out + res
        return x_out