# -*- coding: utf-8 -*-
'''
@File    :   Automodel.py
@Time    :   2025/03/26 23:53:26
@Author  :   Jintong Sun
@email :   213212555@seu.edu.cn
'''

import sys

import torch
import torch.nn as nn
import lightning as L
from torch.nn import SmoothL1Loss
from torch.optim import lr_scheduler
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from .StairLLM_Net import StairLLM_Net



class BaseAutoModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model: nn.Module | None = None
        self.lr: float = self.config.lr
        L.seed_everything(config.seed)
        match self.config.loss:
            case "mse":
                self.criterion = nn.MSELoss()
            case 'mae':
                self.criterion = SmoothL1Loss()
            case _:
                self.criterion = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.strict_loading = False

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False) -> dict:
        trainable_param_names = set([
            name for name, param in self.named_parameters()
            if param.requires_grad 
        ])

        state_dict = {k: v for k, v in super().state_dict().items() if (k in trainable_param_names) or ('llm_model' not in k)}
        return state_dict

    def configure_model(self):
        raise NotImplementedError('Please implement this method in child class')

    def training_step(self, batch, batch_idx):
        """
        The training step designed by Lightning
        :param batch: (tgt,memory) tgt.shape (batch_size, seq_len) memory.shape (batch_size, channels, seq_len)
        :param batch_idx:
        :return:loss
        """
        x, y, _ = batch
        y_hat = self.model(*x)
        y = y[:, -self.config.pred_len:]
        loss = self.criterion(y_hat, y)
        log_dict = {
            'train_loss': loss.item(),
            'train_mae': self.mae(y_hat, y).item(),
            'train_mse': self.mse(y_hat, y).item(),
        }
        self.log_dict(log_dict, on_epoch=False, prog_bar=True, on_step=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self.model(*x)
        y = y[:, -self.config.pred_len:]
        mask = mask[:, -self.config.pred_len:]
        y, y_hat = y * mask, y_hat * mask
        log_dict = {
            'val_loss': self.criterion(y_hat, y).item(),
            'val_mae': self.mae(y_hat, y).item(),
            'val_mse': self.mse(y_hat, y).item(),
        }
        self.log_dict(log_dict, on_epoch=True, prog_bar=True, on_step=False, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self.model(*x)
        y = y[:, -self.config.pred_len:]
        mask = mask[:, -self.config.pred_len:]
        y, y_hat = y * mask, y_hat * mask
        log_dict = {
            'test_loss': self.criterion(y_hat, y).item(),
            'test_mae': self.mae(y_hat, y).item(),
            'test_mse': self.mse(y_hat, y).item(),
        }
        self.log_dict(log_dict)

    def predict_step(self, batch):
        x, y, mask = batch
        y_hat: torch.Tensor = self.model(*x)
        y = y[:, -self.config.pred_len:]
        mask = mask[:, -self.config.pred_len:]
        y, y_hat = y * mask, y_hat * mask
        return y.detach(), y_hat.detach()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.cos_t_max, eta_min=1e-8)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }


class NoBugTQDMBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()

    def init_validation_tqdm(self):
        bar = Tqdm(
            desc=self.validation_description,
            position=0, 
            disable=self.is_disabled,
            leave=False,  
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar



class StairLLMAutoModel(BaseAutoModel):
    def __init__(self, config):
        super().__init__(config)
        self.save_hyperparameters(ignore=['model'])

    def configure_model(self):
        if self.model is not None:
            return None
        else:
            self.model = StairLLM_Net(self.config)
