# -*- coding: utf-8 -*-
"""
@File    :   DataModule.py
@Time    :   2025/03/27 00:51:42
@Author  :   Jintong Sun
@email :   213212555@seu.edu.cn
"""

import torch
import pywt
import numpy as np
from copy import copy
import torch.utils.data as data
import pandas as pd
import warnings
from lightning import LightningDataModule
from .Config import StairLLMConfig


class BaseDataset(data.Dataset):
    # We use capacity power of each PV in the dataset to normalize the power series
    # So you could change the dict, or use other methods to normalize the power series
    power_capacity_dict = {
        idx: item
        for idx, item in enumerate(
            [
                110,
                110,
                110,
                300,
                105,
                150,
                200,
                100,
                50,
                66.74,
                40,
                60,
                39.11,
                52.07,
                30,
                7.07,
                20,
                18.6,
                20,
                19.6,
                40,
            ],
            start=1,
        )
    }

    def __init__(self, config: StairLLMConfig, mode: str = "train"):
        super().__init__()
        self.config: StairLLMConfig = config
        self.seq_len: int = config.seq_len
        self.pred_len: int = config.pred_len
        # in our dataset, The name of power series is PV_POWER
        self.power_name: str = "PV_POWER"
        if mode == "val":
            print("use val dataset")
            self.dataset_idx: int | list[int] | None = config.val_dataset_idx
            self.dataset_path: str | list[str] = config.val_dataset_path
        else:
            self.dataset_idx: int | list[int] | None = config.dataset_idx
            self.dataset_path: str | list[str] = config.dataset_path

        if isinstance(self.dataset_path, str) and isinstance(self.dataset_idx, int):
            self.raw_data: pd.DataFrame = pd.read_excel(
                self.dataset_path, sheet_name="Sheet1"
            )
            if self.dataset_idx is not None:
                divisor = self.power_capacity_dict[self.dataset_idx]
                self.raw_data[self.power_name] = (
                    self.raw_data[self.power_name] / divisor * 100
                )
        elif isinstance(self.dataset_path, list) and isinstance(self.dataset_idx, list):
            if not self.config.use_all_data:
                raise ValueError(
                    f"Using dataset lists in {config.task_stage} only supports use_all_data = True"
                )
            dataframes = [
                pd.read_excel(file, sheet_name="Sheet1") for file in self.dataset_path
            ]
            assert len(self.dataset_idx) == len(dataframes)
            for idx, df in zip(self.dataset_idx, dataframes):
                divisor = self.power_capacity_dict[idx]
                df[self.power_name] = df[self.power_name] / divisor * 100
            self.raw_data: pd.DataFrame = pd.concat(dataframes, ignore_index=True)
        else:
            ValueError(
                f"Got dataset_idx and dataset_path are {type(self.dataset_idx)} and {type(self.dataset_path)}, please check your config"
            )

        self.raw_data: pd.DataFrame = self.raw_data.dropna()

        self._make_night_mask()

        self.total_len: int = len(self.raw_data)
        if config.use_all_data:
            warnings.warn(
                f"use all data to {config.task_stage}, be cautious of data leaks"
            )
            self.start_idx: int = 0
            self.end_idx: int = self.total_len - self.seq_len - self.pred_len
        else:
            match mode:
                case "train":
                    self.start_idx: int = 0
                    self.end_idx: int = int(0.6 * self.total_len)
                case "val":
                    self.start_idx: int = int(0.6 * self.total_len)
                    self.end_idx: int = int(0.7 * self.total_len)
                case "test":
                    self.start_idx: int = int(0.7 * self.total_len)
                    self.end_idx: int = self.total_len
                case _:
                    raise NotImplementedError(
                        "please make sure the mode in [train, val, test]"
                    )

        self.data: pd.DataFrame = self.raw_data.iloc[self.start_idx : self.end_idx]

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def _make_night_mask(self):
        self.raw_data["DATE_TIME"] = pd.to_datetime(self.raw_data["DATE_TIME"])
        self.raw_data["mask"] = self.raw_data["hour"] = np.where(
            (self.raw_data["DATE_TIME"].dt.hour >= 6)
            & (self.raw_data["DATE_TIME"].dt.hour < 18),
            1.0,
            0.0,
        )


class Dataset(BaseDataset):
    def __init__(self, config, mode="train"):
        super().__init__(config, mode)
        self.var_list = None
        # The weather data we use, if you have other weather data, you can add them here, and don't forget to change the config
        # We did not normalize the weather data, but if you need, you can do it after here
        self.var_list = [
            "HUMIDITY",
            "PRESSURE",
            "RAIN",
            "TEMPERATURE",
            "SWDOWN",
            "SWDDIR",
            "SWDDNI",
            "SWDDIF",
            "PRECIPITATION",
        ]
        self.wavelet_type = self.config.wavelet_type

    def get_wavelets(self, tgt):
        coeffs = pywt.wavedec(tgt, self.wavelet_type)
        coeffs = np.concatenate(coeffs)
        return coeffs

    def __getitem__(self, idx):
        """
        tgt:Power data input in model to predict
        memory:Weather data input in model to predict
        y:The prediction that model predicts
        :param idx:
        :return:
        """
        tgt_end_idx: int = idx + self.seq_len
        y_start_idx: int = idx + self.pred_len
        y_end_idx: int = y_start_idx + self.seq_len
        memory_start_idx: int = idx
        memory_end_idx: int = memory_start_idx + self.seq_len + self.pred_len
        tgt = self.data[self.power_name].iloc[idx:tgt_end_idx].to_numpy()
        y = self.data[self.power_name].iloc[y_start_idx:y_end_idx].to_numpy()
        mask = self.data["mask"].iloc[y_start_idx:y_end_idx].to_numpy()
        wavelet = torch.tensor(self.get_wavelets(tgt), dtype=torch.float32)
        tgt = torch.tensor(copy(tgt), dtype=torch.float32)
        y = torch.tensor(copy(y), dtype=torch.float32)
        mask = torch.tensor(copy(mask), dtype=torch.float32)
        memory = (
            self.data[self.var_list].iloc[memory_start_idx:memory_end_idx].to_numpy()
        )
        memory = torch.tensor(memory, dtype=torch.float32).permute(1, 0)

        return (tgt, memory, wavelet), y, mask


class BaseDataModule(LightningDataModule):
    def __init__(self, config, cls_dataset):
        super().__init__()
        self.config = config
        self.train_dataset: BaseDataset | None = None
        self.val_dataset: BaseDataset | None = None
        self.test_dataset: BaseDataset | None = None
        self.pred_dataset: BaseDataset | None = None
        self.cls_dataset = cls_dataset

    def setup(self, stage: str = None):
        if stage == "fit":
            self.train_dataset = self.cls_dataset(self.config)
            if self.config.use_val:
                self.val_dataset = self.cls_dataset(self.config, mode="val")
        elif stage == "test":
            self.test_dataset = self.cls_dataset(self.config)
        elif stage == "predict":
            self.pred_dataset = self.cls_dataset(self.config)

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=12,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return (
            data.DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                num_workers=4,
                persistent_workers=True,
            )
            if self.config.use_val else None
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            num_workers=4,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return data.DataLoader(
            self.pred_dataset,
            batch_size=self.config.batch_size,
            num_workers=4,
            persistent_workers=True,
        )


class DataModule(BaseDataModule):
    def __init__(self, config):
        super().__init__(config, Dataset)
