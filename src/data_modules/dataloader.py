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

            # target 컬럼에서 클래스별 weight 계산
            if (
                not hasattr(dataset, "data")
                or target_column_name not in dataset.data.columns
            ):
                raise ValueError(
                    f"Dataset does not have 'data' attribute or target column '{target_column_name}' is not in dataset.data.columns."
                )

            targets = np.array(dataset.data[target_column_name])
            class_sample_count = np.bincount(targets.astype(int))
            weight = 1.0 / (class_sample_count + 1e-6)
            samples_weight = weight[targets.astype(int)]
            sampler = WeightedRandomSampler(
                samples_weight, len(samples_weight), replacement=True
            )
            super().__init__(
                dataset,
                sampler=sampler,
                **kwargs,
            )
        else:
            super().__init__(dataset, shuffle=False, **kwargs)

    def __iter__(self):
        if self.mode == "train":
            for batch in super().__iter__():
                # 배치가 dict/list/tuple 등 다양한 형태일 수 있으므로, 길이 체크를 유연하게 처리
                batch_size = None
                if isinstance(batch, dict):
                    # dict일 경우, 첫 번째 텐서의 batch 차원
                    first_tensor = next(iter(batch.values()))
                    batch_size = (
                        first_tensor.size(0)
                        if hasattr(first_tensor, "size")
                        else len(first_tensor)
                    )
                elif isinstance(batch, (list, tuple)):
                    # 리스트/튜플일 경우, 첫 번째 요소의 batch 차원
                    first_tensor = batch[0]
                    batch_size = (
                        first_tensor.size(0)
                        if hasattr(first_tensor, "size")
                        else len(first_tensor)
                    )
                else:
                    # 기타: 텐서 자체가 배치일 수도 있음
                    batch_size = batch.size(0) if hasattr(batch, "size") else len(batch)
                if batch_size == 1:
                    continue  # 배치사이즈 1이면 스킵
                yield batch
        else:
            yield from super().__iter__()
