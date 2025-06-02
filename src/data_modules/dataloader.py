import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, config, mode="train", **kwargs):
        self.mode = "train" if mode.lower() == "train" else "valid"
        self.config = config
        self.dataset = dataset

        if self.mode == "train":
            data_config = config.get("data")
            if data_config is None:
                raise ValueError(
                    "Configuration error: 'data' section is missing in config for CustomDataLoader."
                )

            target_column_name = data_config.get("target_column")
            if target_column_name is None:
                raise ValueError(
                    "Configuration error: 'data.target_column' is missing in config for CustomDataLoader."
                )

            if (
                not hasattr(dataset, "data")
                or target_column_name not in dataset.data.columns
            ):
                raise ValueError(
                    f"Dataset does not have 'data' attribute or target column '{target_column_name}' is not in dataset.data.columns."
                )

            targets = np.array(dataset.data[target_column_name])
            class_sample_count = np.bincount(targets.astype(int))
            weight = 1.0 / (
                class_sample_count + 1e-6
            )  # 0으로 나누는 것을 방지하기 위해 작은 값 추가
            samples_weight = weight[targets.astype(int)]
            sampler = WeightedRandomSampler(
                samples_weight, len(samples_weight), replacement=True
            )
            # train 모드에서는 WeightedRandomSampler를 사용
            super().__init__(
                dataset,
                sampler=sampler,
                **kwargs,  # batch_size 등 DataLoader의 다른 인자들을 전달
            )
        else:
            # valid 또는 test 모드에서는 sampler 없이 순차적으로 데이터를 로드 (shuffle=False가 기본값)
            super().__init__(dataset, shuffle=False, **kwargs)
