# src/models/image_extractor.py
import torch
import torch.nn as nn

# from torchvision import models # torchvision.models는 timm을 사용할 경우 직접 필요하지 않을 수 있음
import timm  # PyTorch Image Models 라이브러리 사용

# Store output dimensions for known models
IMAGE_FEATURE_DIMS = {
    "efficientnet_b0": 1280,
    "efficientnet_b2": 1408,
    "convnext_tiny": 768,
    "convnext_small": 768,  # <-- convnext_small 추가
    "resnet50": 2048,
}


class ImageExtractor(nn.Module):
    """
    다양한 이미지 백본 모델로부터 특징(feature)을 추출하는 제네릭 래퍼 클래스.
    주로 timm 라이브러리를 사용합니다.
    """

    def __init__(self, config):
        """
        Args:
            config (dict): {
                'name': 모델 이름 (timm에서 지원하는 이름),
                'params': { 'pretrained': True, ... }
            }
        """
        super().__init__()
        if isinstance(config, dict):
            self.model_name = config.get("name", "convnext_tiny")
            params = config.get("params", {})
        else:
            raise ValueError("ImageExtractor config는 dict여야 합니다.")
        self.pretrained = params.pop("pretrained", True)
        self.extra_params = params
        try:
            self.backbone = timm.create_model(
                self.model_name,
                pretrained=self.pretrained,
                num_classes=0,
                **self.extra_params,
            )
            self.output_dim = self.backbone.num_features
            print(
                f"Successfully created '{self.model_name}' backbone. Output feature dimension: {self.output_dim}"
            )
            if self.model_name in IMAGE_FEATURE_DIMS:
                if IMAGE_FEATURE_DIMS[self.model_name] != self.output_dim:
                    print(
                        f"Warning: Output dimension mismatch for {self.model_name}. "
                        f"Stored: {IMAGE_FEATURE_DIMS[self.model_name]}, Inferred: {self.output_dim}. "
                        f"Updating stored value."
                    )
                    IMAGE_FEATURE_DIMS[self.model_name] = self.output_dim
            else:
                print(
                    f"Warning: Output dimension for {self.model_name} ({self.output_dim}) was not pre-defined in IMAGE_FEATURE_DIMS. Adding it now."
                )
                IMAGE_FEATURE_DIMS[self.model_name] = self.output_dim
        except Exception as e:
            raise ValueError(
                f"Could not create image model '{self.model_name}' using timm. Error: {e}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """이미지 배치(x)를 받아 특징 벡터를 반환합니다."""
        # x shape: (batch_size, channels, height, width)
        features = self.backbone(x)
        # features shape: (batch_size, self.output_dim)
        return features


def build_image_extractor(name: str, params: dict) -> ImageExtractor:
    """
    설정(config) 기반으로 ImageExtractor 인스턴스를 생성하는 팩토리 함수.

    Args:
        name (str): 사용할 모델의 이름 (timm에서 지원하는 이름).
        params (dict): ImageExtractor 생성자에 전달될 파라미터 딕셔너리
                       (예: {'pretrained': True, 'drop_path_rate': 0.1}).

    Returns:
        ImageExtractor 인스턴스.
    """
    print(f"\n--- Building Image Extractor ---")
    print(f"Requested model name: '{name}', Params: {params}")
    # params 딕셔너리에서 'pretrained' 같은 특정 키를 분리하거나 기본값을 설정할 수 있음
    pretrained = params.pop(
        "pretrained", True
    )  # params에서 pretrained를 꺼내고 없으면 True 사용
    print(f"Pretrained weights: {pretrained}")

    # 나머지 params는 **kwargs로 전달
    try:
        extractor = ImageExtractor(config={"name": name, "params": params})
        print(f"Successfully built ImageExtractor for '{name}'.")
        return extractor
    except Exception as e:
        print(f"Error building ImageExtractor '{name}': {e}")
        raise e
