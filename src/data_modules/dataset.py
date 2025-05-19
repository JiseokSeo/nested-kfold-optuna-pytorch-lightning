import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, List
import cv2
import os
import pandas as pd
import numpy as np
import logging

# from PIL import Image # PIL 사용 안 함 (cv2 사용)
# from torchvision.transforms import ToTensor # image_transform에서 처리
from src.data_modules.transformer import transforms

custom_dataset_logger = logging.getLogger(__name__ + ".CustomDataset")


class CustomDataset(Dataset):
    def __init__(self, data: pd.DataFrame, mode: str, config: Dict):
        self.data = data.copy()
        self.current_mode_for_transform = "train" if mode == "train" else "valid"
        if self.current_mode_for_transform not in transforms:
            raise KeyError(
                f"Mode '{self.current_mode_for_transform}' not found in transforms keys. Available keys: {list(transforms.keys())}"
            )
        self.transform = transforms[
            self.current_mode_for_transform
        ]()  # self.image_transform 내부에서 사용됨

        self.config = config
        self.data_config = config.get("data")
        if self.data_config is None:
            raise ValueError(
                "Configuration error: 'data' section is missing in config."
            )

        self.image_dir = self.data_config.get("image_dir")
        self.use_image = self.data_config.get("use_image", False)
        self.use_csv = self.data_config.get("use_csv", False)

        self.id_column = self.data_config.get("id_column")
        if self.id_column is None:
            raise ValueError(
                "Configuration error: 'data.id_column' is missing in config."
            )

        self.target_column = self.data_config.get("target_column")
        if self.target_column is None:
            raise ValueError(
                "Configuration error: 'data.target_column' is missing in config."
            )

        self.feature_columns: List[str] = self.data_config.get("feature_columns", [])

        if self.use_image and not self.image_dir:
            raise ValueError(
                "config['data']['image_dir'] must be specified when 'use_image' is True."
            )
        if self.use_csv and not self.feature_columns:
            custom_dataset_logger.warning(
                f"Warning: 'use_csv' is True but 'feature_columns' is empty or not provided. CSV tensor will be None or empty."
            )
            raise ValueError(
                "Configuration error: 'data.feature_columns' is missing in config."
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        custom_dataset_logger.info(
            f"[CustomDataset __getitem__] Called for index {idx}. Mode: {self.current_mode_for_transform}"
        )
        row = self.data.iloc[idx]
        image_id = row[self.id_column]

        image_tensor = torch.empty(0)
        if self.use_image:
            try:
                image_np = self.get_image(image_id)
                image_tensor = self.image_transform(image_np)
            except FileNotFoundError as e:
                custom_dataset_logger.error(
                    f"[CustomDataset __getitem__] FileNotFoundError for image_id {image_id}: {e.filename if hasattr(e, 'filename') else str(e)}"
                )
                raise e

        csv_tensor = torch.empty(0)
        if self.use_csv:
            if not self.feature_columns:  # feature_columns가 비어있으면 빈 텐서
                custom_dataset_logger.debug(
                    f"No feature_columns defined for CSV, returning empty tensor for {image_id}"
                )
            else:
                try:
                    csv_features_series = self.get_csv_features(row)
                    if csv_features_series.empty or csv_features_series.isnull().all():
                        custom_dataset_logger.warning(
                            f"For id {image_id}, no valid CSV features or all NaN for columns: {self.feature_columns}. Returning empty tensor."
                        )
                        # csv_tensor는 이미 torch.empty(0)으로 초기화됨
                    else:
                        csv_tensor = self.to_csv_tensor(
                            csv_features_series
                        )  # 원래 방식 호출
                except (
                    KeyError
                ) as e:  # get_csv_features 내부에서 발생할 수 있는 KeyError
                    custom_dataset_logger.error(
                        f"CSV feature column error for ID {image_id}: {e}. Available columns: {row.index.tolist()}"
                    )
                    raise e

        target_tensor = torch.empty(0)
        if self.target_column:
            try:
                target_val = row[self.target_column]
                target_tensor = torch.tensor(target_val, dtype=torch.float32)
            except KeyError:
                custom_dataset_logger.error(
                    f"Target column '{self.target_column}' not found for ID {image_id}."
                )
                if self.config.get("runtime", {}).get(
                    "raise_error_on_missing_target", True
                ):
                    raise ValueError(
                        f"Target column '{self.target_column}' not found for ID {image_id}."
                    )
                target_tensor = torch.tensor(0.0, dtype=torch.float32)  # 임시 타겟
        else:
            raise ValueError(
                f"Target column '{self.target_column}' not found for ID {image_id}."
            )

        return {
            "image_id": image_id,
            "image": image_tensor,
            "csv": csv_tensor,
            "target": target_tensor,
        }

    def get_image(self, image_id: Any) -> np.ndarray:
        image_filename = f"{image_id}.jpg"  # 기본 확장자
        if self.data_config and self.data_config.get("image_filename_extension"):
            image_filename = f"{image_id}{self.data_config['image_filename_extension']}"

        # image_dir이 None일 수 있으므로 확인
        if not self.image_dir:
            error = FileNotFoundError("Image directory (image_dir) is not configured.")
            # error.filename = None # filename 속성 추가 안 함
            raise error

        image_path = os.path.join(self.image_dir, image_filename)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            error = FileNotFoundError(f"Image not found at path: {image_path}")
            error.filename = image_path  # type: ignore
            raise error
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_csv_features(self, row: pd.Series) -> pd.Series:
        if not self.feature_columns:
            return pd.Series(dtype=np.float32)

        valid_columns = [col for col in self.feature_columns if col in row.index]
        if not valid_columns:
            custom_dataset_logger.warning(
                f"None of the specified feature_columns {self.feature_columns} found in data row with columns {row.index.tolist()}. Returning empty Series."
            )
            return pd.Series(dtype=np.float32)

        # 누락된 컬럼에 대한 경고 (선택 사항)
        missing_cols = [col for col in self.feature_columns if col not in valid_columns]
        if missing_cols:
            custom_dataset_logger.warning(
                f"Missing CSV feature columns for row {row.get(self.id_column, 'UnknownID')}: {missing_cols}. Present columns: {valid_columns}"
            )

        return row[valid_columns]

    def to_csv_tensor(self, csv_features: pd.Series) -> torch.Tensor:
        if csv_features.empty:  # 이중 확인
            return torch.empty(0)
        numpy_array = csv_features.values.astype(np.float32)
        return torch.from_numpy(numpy_array)

    def image_transform(self, image: np.ndarray) -> torch.Tensor:
        # self.transform은 __init__에서 초기화됨 (Albumentations Compose 객체)
        # 이 Compose 객체는 ToTensorV2 등을 포함하여 CHW, float32 텐서를 반환해야 함
        augmented = self.transform(image=image)
        image_tensor = augmented["image"]
        return image_tensor

    def _get_transform(self):
        # 이 메소드는 __init__에서 self.transform을 설정하는데 사용되었던 것으로 보이나,
        # 현재 __init__에서는 src.data_modules.transformer.transforms 딕셔너리를 직접 사용합니다.
        # 만약 이 메소드가 다른 곳에서 호출되지 않는다면 제거해도 무방할 수 있습니다.
        # 현재 코드에서는 호출되지 않으므로 pass로 둡니다.
        pass
