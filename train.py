# -*- coding: utf-8 -*-
"""
@File    :   train.py
@Time    :   2025/03/27 00:17:54
@Author  :   Jintong Sun
@email :   213212555@seu.edu.cn
"""

import lightning as L
from Models import StairLLMAutoModel, DataModule, StairLLMConfig, NoBugTQDMBar


def train(dataset_idx: list[int] | int, val_dataset_idx: list[int] | int | None):
    llm_model_name: str = "gpt2small"
    config = StairLLMConfig(
        task_stage="train",
        llm_model_name=llm_model_name,
        llm_model_root=f"./PretrainedModels/{llm_model_name}",
        dataset_idx=dataset_idx,
        dataset_path=(
            [f"./dataset/pv{idx}_filtered.xlsx" for idx in dataset_idx]
            if isinstance(dataset_idx, list)
            else f"./dataset/pv{dataset_idx}_filtered.xlsx"
        ),
        val_dataset_idx=val_dataset_idx,
        val_dataset_path=(
            [f"./dataset/pv{idx}_filtered.xlsx" for idx in val_dataset_idx]
            if isinstance(val_dataset_idx, list)
            else f"./dataset/pv{val_dataset_idx}_filtered.xlsx"
        ),
        batch_size=256,
        patch_len=96,
        llm_dim=768,
        lr=2e-3,
        use_all_data=True,
        max_epoch=6,
        swa_lr=2e-3,
        loss="mae",
        wavelet_len=320,
        use_val=False if val_dataset_idx is None else True,
    )
    dm = DataModule(config)
    automodel = StairLLMAutoModel(config)
    trainer = L.Trainer(
        max_epochs=config.max_epoch,
        log_every_n_steps=4,
        devices="auto",
        fast_dev_run=0,
        accelerator="cuda",
        strategy="auto",
        precision="32",
        callbacks=[NoBugTQDMBar()],
        enable_checkpointing=False,
        limit_val_batches=None if config.use_val else 0.0,
    )
    trainer.fit(model=automodel, datamodule=dm, ckpt_path=None)
    trainer.save_checkpoint(
        f"Checkpoints/{llm_model_name}_trainable_{dataset_idx}.ckpt".replace("[", "").replace("]", "").replace(", ", "_")
    )

def main():
    for dataset_idx in [
        [1, 2, 3, 4, 5, 6, 7, 8],
    ]:
        train_idx = dataset_idx
        print(f'{train_idx=}')
        train(train_idx, None)


if __name__ == "__main__":
    main()
