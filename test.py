# -*- coding: utf-8 -*-
"""
@File    :   test.py
@Time    :   2025/03/27 01:26:56
@Author  :   Jintong Sun
@email :   213212555@seu.edu.cn
"""
import lightning as L
from Models import StairLLMAutoModel, DataModule, StairLLMConfig


def test(dataset_idx: list[int] | int, ckpt_idx: list[int] | int):
    llm_model_name: str = "gpt2small"
    config = StairLLMConfig(
        task_stage="test",
        llm_model_name=llm_model_name,
        llm_model_root=f"./PretrainedModels/{llm_model_name}",
        llm_dim=768,
        dataset_idx=dataset_idx,
        dataset_path=(
            [f"./dataset/pv{idx}_filtered.xlsx" for idx in dataset_idx]
            if isinstance(dataset_idx, list)
            else f"./dataset/pv{dataset_idx}_filtered.xlsx"
        ),
        val_dataset_path=None,
        batch_size=256,
        patch_len=96,
        use_all_data=True,
        wavelet_len=320,
        use_val=False
    )
    dm = DataModule(config)
    automodel = StairLLMAutoModel.load_from_checkpoint(
        checkpoint_path=f"Checkpoints/{llm_model_name}_trainable_{ckpt_idx}.ckpt".replace("[", "").replace("]", "").replace(", ", "_"),
        config=config,
    )
    trainer = L.Trainer(log_every_n_steps=4, devices=1)
    trainer.test(model=automodel, datamodule=dm, ckpt_path=None)

def main():
    full_dataset_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]
    for ckpt_idx in [
        [1, 2, 3, 4, 5, 6, 7, 8],
    ]:
        dataset_idx = [idx for idx in full_dataset_idx if idx not in ckpt_idx]
        print(f'{dataset_idx=}, {ckpt_idx=}')
        test(dataset_idx, ckpt_idx)

if __name__ == "__main__":
    main()
