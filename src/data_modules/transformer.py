# src/data/transforms.py (value 인자 복원)
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2  # OpenCV 사용 명시

# --- 기본 설정값 ---
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def default_train():
    # 학습용 변환 파이프라인
    return A.Compose(
        [
            A.LongestMaxSize(max_size=224, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(
                min_height=224,
                min_width=224,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.1),
            A.Rotate(
                limit=(-1, 1),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                p=0.2, brightness_limit=0.05, contrast_limit=0.05
            ),
            A.GaussNoise(std_range=[0.01, 0.05], noise_scale_factor=0.5, p=0.1),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )


def default_val_test():
    # 검증 및 테스트용 변환 파이프라인
    return A.Compose(
        [
            A.LongestMaxSize(max_size=224, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(
                min_height=224,
                min_width=224,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )


# 함수 정의 이후에 transforms 딕셔너리 할당
transforms = {"train": default_train, "valid": default_val_test}
