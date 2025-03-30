# -*- coding: utf-8 -*-
"""
@File    :   LLM.py
@Time    :   2025/03/27 00:06:22
@Author  :   Jintong Sun
@email :   213212555@seu.edu.cn
"""

import torch.nn as nn
from peft.mapping import get_peft_model
from peft.tuners.lora.config import LoraConfig
from transformers import (
    GPT2Config,
    GPT2Model,
    GPT2Tokenizer,
    AutoModel,
    AutoTokenizer,
    AutoConfig,
)


def get_llm_model(
    model_name: str,
    model_root: str,
    fine_tune_layernorm: bool = True,
) -> tuple:
    match model_name:
        case "gpt2small" | "gpt2xl":
            # If you could connect to the HuggingFace Hub, you can change below to download from Web.
            config = GPT2Config.from_pretrained(model_root)
            llm_model= GPT2Model.from_pretrained(
                model_root,
                local_files_only=True,
                config=config,
                ignore_mismatched_sizes=False,
            )
            tokenizer = GPT2Tokenizer.from_pretrained(
                model_root,
                local_files_only=True,
            )
        case "Qwen1.5B" | "Qwen7B":
            config = AutoConfig.from_pretrained(model_root)
            llm_model = AutoModel.from_pretrained(
                model_root,
                local_files_only=True,
                config=config,
                ignore_mismatched_sizes=False,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_root,
                local_files_only=True,
            )
        case _:
            raise NotImplementedError("We did not prepare this model")

    tokenizer.pad_token = tokenizer.eos_token

    for param in llm_model.parameters():
        param.requires_grad = False

    if fine_tune_layernorm:
        print("Try fine-tuning layer norm")
        for name, layer in llm_model.named_modules():
            if fine_tune_layernorm and isinstance(layer, nn.LayerNorm):
                for param_name, param in layer.named_parameters(recurse=False):
                    param.requires_grad = True

    return llm_model, tokenizer
