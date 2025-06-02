from src.data_modules.dataset import CustomDataset
from src.data_modules.dataloader import CustomDataLoader
import lightning as L
import pandas as pd


def build_dataset_from_config(trial_config, df, mode="train"):
    dataset = CustomDataset(data=df, mode=mode, config=trial_config)
    return dataset


def build_dataloader_from_config(config, dataset, mode="train"):
    batch_size = config["data"].get("batch_size", 16)
    num_workers = config["data"].get("num_workers", 1)
    use_persistent_workers = True if num_workers > 0 else False
    drop_last = config["data"].get("drop_last", False)
    return CustomDataLoader(
        dataset=dataset,
        config=config,
        mode=mode,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=use_persistent_workers,
        drop_last=drop_last,
    )


def build_data(config, df, mode="train"):
    dataset = build_dataset_from_config(config, df, mode=mode)
    dataloader = build_dataloader_from_config(config, dataset, mode=mode)
    return dataloader
