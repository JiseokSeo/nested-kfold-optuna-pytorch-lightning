import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
import logging

# 로거 설정
logger = logging.getLogger(__name__)

from src.model_modules.components.basemodel import BaseModel
from src.model_modules.components.image_extractor import ImageExtractor
from src.model_modules.components.csv_extractor import MLPCSVExtractor
from src.model_modules.components.fusion import ConcatFusion, GatingFusion, FusionLayer
from src.model_modules.components.classifier import MLPClassifier
from src.trainer.metrics import test_metrics


class MultiModalModel(BaseModel):
    def __init__(self, config: Dict):
        super().__init__()
        logger.info("[MultiModalModel __init__] START")
        self.save_hyperparameters(config)
        self.model_config = self.hparams.get("model", {})
        self.data_config = self.hparams.get("data", {})
        self.trainer_config_hparams = self.hparams.get("trainer", {})

        self.use_image = self.data_config.get("use_image")
        self.use_csv = self.data_config.get("use_csv")
        self.use_fusion = True if (self.use_image and self.use_csv) else False
        self.batch_size = self.data_config.get("batch_size")

        if not self.use_image and not self.use_csv:
            raise ValueError(
                "Model config must specify at least 'image_extractor' or 'csv_extractor'."
            )

        self.image_extractor = None
        self.csv_extractor = None
        self.fusion_layer = None
        self.classifier = None

        current_dim = 0
        image_dim = 0
        csv_dim = 0

        if self.use_image:
            img_ext_config = self.model_config.get("image_extractor")
            if img_ext_config is None:
                raise ValueError(
                    "Image extractor config ('model.image_extractor') is missing "
                    "when 'data.use_image' is True."
                )
            self.image_extractor = ImageExtractor(img_ext_config)
            image_dim = self.image_extractor.output_dim
            current_dim = image_dim
            logger.info(
                f"[MultiModalModel __init__] ImageExtractor created. Output_dim: {image_dim}"
            )

        if self.use_csv:
            csv_ext_config = self.model_config.get("csv_extractor")
            if csv_ext_config is None:
                raise ValueError(
                    "CSV extractor config ('model.csv_extractor') is missing "
                    "when 'data.use_csv' is True."
                )
            self.csv_extractor = MLPCSVExtractor(csv_ext_config)
            csv_dim = self.csv_extractor.output_dim
            if not self.use_image:
                current_dim = csv_dim
            logger.info(
                f"[MultiModalModel __init__] MLPCSVExtractor created. Output_dim: {csv_dim}"
            )

        if self.use_image and self.use_csv and self.model_config.get("fusion"):
            fusion_config = self.model_config["fusion"]
            if "params" not in fusion_config:
                fusion_config["params"] = {}
            fusion_config["params"]["image_dim"] = image_dim
            fusion_config["params"]["csv_dim"] = csv_dim
            self.fusion_layer = FusionLayer(fusion_config)
            current_dim = self.fusion_layer.output_dim
            logger.info(
                f"[MultiModalModel __init__] FusionLayer created. Output_dim: {current_dim}"
            )
        elif self.use_image and self.use_csv and not self.model_config.get("fusion"):
            current_dim = image_dim + csv_dim
            logger.info(
                f"[MultiModalModel __init__] Fusion not used, current_dim (sum of image/csv): {current_dim}"
            )

        if self.model_config.get("classifier"):
            classifier_config = self.model_config["classifier"]
            if "params" not in classifier_config:
                classifier_config["params"] = {}
            classifier_config["params"]["input_dim"] = current_dim
            num_classes = self.trainer_config_hparams.get("num_classes")
            if num_classes is None or num_classes <= 0:
                raise ValueError(
                    f"'trainer.num_classes' must be defined and positive. Found: {num_classes}"
                )
            classifier_config["params"]["num_classes"] = num_classes
            self.classifier = MLPClassifier(classifier_config)
            logger.info("[MultiModalModel __init__] MLPClassifier created.")
        else:
            raise ValueError("Classifier must be specified in the model config.")

        if current_dim <= 0:
            raise ValueError(
                f"Classifier input dimension is not positive ({current_dim}). Check model configuration."
            )

        criterion_cfg = self.trainer_config_hparams.get(
            "criterion", {"name": "BCEWithLogitsLoss", "params": {}}
        )
        criterion_name = criterion_cfg.get("name", "BCEWithLogitsLoss")
        criterion_params = criterion_cfg.get("params", {})
        if criterion_name == "BCEWithLogitsLoss":
            self.criterion = nn.BCEWithLogitsLoss(**criterion_params)
            logger.info(
                f"[MultiModalModel __init__] Criterion '{criterion_name}' created."
            )
        elif criterion_name == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss(**criterion_params)
            logger.info(
                f"[MultiModalModel __init__] Criterion '{criterion_name}' created."
            )
        else:
            raise NotImplementedError(f"지원하지 않는 손실 함수: {criterion_name}")
        logger.info(
            f"[MultiModalModel __init__] Criterion '{criterion_name}' created with params {criterion_params}."
        )

        logger.info("[MultiModalModel __init__] END")

    def forward(self, image=None, csv=None):
        features = None
        image_features = None
        csv_features = None

        if self.use_image and image is not None and image.numel() > 0:
            image_features = self.image_extractor(image)

        if self.use_csv and csv is not None and csv.numel() > 0:
            csv_features = self.csv_extractor(csv)

        if self.use_fusion:
            if image_features is None or csv_features is None:
                raise ValueError(
                    "Fusion is enabled, but valid features for both image and csv are not available. "
                    "Check data and configuration. One or both inputs might be dummy tensors (e.g. torch.empty(0))."
                )

            if self.fusion_layer is None:
                features = torch.cat((image_features, csv_features), dim=1)
            else:
                features = self.fusion_layer(image_features, csv_features)
        elif image_features is not None:
            features = image_features
        elif csv_features is not None:
            features = csv_features
        else:
            raise ValueError(
                "No valid features available from any modality (image or csv). "
                "Both inputs appear to be dummy tensors or data loading failed silently."
            )

        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        logger.info(
            f"[TRAINING_STEP] START - Epoch {self.current_epoch}, Batch {batch_idx}"
        )
        image = batch["image"]
        csv = batch["csv"]
        target = batch["target"]
        logits = self.forward(image=image, csv=csv)
        loss = self.criterion(logits.squeeze(), target.float())

        logger.info(
            f"[TRAINING_STEP] Calculated loss: {loss.item()} - Epoch {self.current_epoch}, Batch {batch_idx}"
        )
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(
                f"[TRAINING_STEP] CRITICAL: NaN or Inf loss detected! Loss: {loss.item()}. Epoch {self.current_epoch}, Batch {batch_idx}"
            )
            # Optuna가 문제를 인지하도록 여기서 오류를 발생시킬 수 있습니다.
            # raise ValueError("NaN or Inf loss detected during training step.")

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        logger.info(
            f"[TRAINING_STEP] END - Epoch {self.current_epoch}, Batch {batch_idx}"
        )
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        csv = batch["csv"]
        target = batch["target"]
        logits = self.forward(image=image, csv=csv)
        if not hasattr(self, "val_logits"):
            self.val_logits = []
            self.val_targets = []
        self.val_logits.append(logits.detach().cpu())
        self.val_targets.append(target.detach().cpu())

    def on_validation_epoch_end(self):
        if (
            not hasattr(self, "val_logits")
            or not hasattr(self, "val_targets")
            or not self.val_logits
            or not self.val_targets
        ):
            logger.warning(
                "[V_EPOCH_END] val_logits or val_targets is missing or empty. Skipping metric calculation."
            )
            # Ensure lists are cleared even if they were partially populated or just initialized
            self.val_logits = []
            self.val_targets = []
            # Log available callback metrics even if validation did not produce new ones
            if (
                self.trainer
                and hasattr(self.trainer, "callback_metrics")
                and self.trainer.callback_metrics
            ):
                logger.info(
                    f"[V_EPOCH_END] Metrics currently in trainer.callback_metrics: {list(self.trainer.callback_metrics.keys())}"
                )
            else:
                logger.info(
                    "[V_EPOCH_END] trainer.callback_metrics not available or empty at this point (no new validation outputs)."
                )
            return

        all_logits = torch.cat(self.val_logits, dim=0)
        all_targets = torch.cat(self.val_targets, dim=0)

        # val_loss 로깅
        val_loss = self.criterion(all_logits.squeeze(), all_targets.float())
        self.log(
            "val_loss",
            val_loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        logger.info(f"[V_EPOCH_END] Logged PTL metric 'val_loss': {val_loss.item()}")

        # 사용된 로그 및 타겟 리스트 초기화
        self.val_logits = []
        self.val_targets = []

    def test_step(self, batch, batch_idx):
        image = batch["image"]
        csv = batch["csv"]
        target = batch["target"]
        image_id = batch["image_id"]
        logits = self.forward(image=image, csv=csv)

        if not hasattr(self, "test_logits"):
            self.test_logits = []
            self.test_targets = []
            self.image_ids = []

        self.test_logits.append(logits.detach().cpu())
        self.test_targets.append(target.detach().cpu())
        self.image_ids.append(image_id.detach().cpu())

    def on_test_epoch_end(self):
        logger.info("[TEST_EPOCH_END] START")
        if (
            not hasattr(self, "test_logits")
            or not hasattr(self, "test_targets")
            or not self.test_logits
            or not self.test_targets
        ):
            logger.warning(
                "[TEST_EPOCH_END] test_logits or test_targets is missing or empty. Skipping metric calculation."
            )
            # Ensure lists are cleared/initialized even if they were partially populated or just initialized
            self.test_logits = []
            self.test_targets = []
            self.image_ids = []
            logger.info("[TEST_EPOCH_END] END (skipped)")
            return

        all_logits = torch.cat(self.test_logits, dim=0)
        all_targets = torch.cat(self.test_targets, dim=0)
        all_image_ids = torch.cat(self.image_ids, dim=0)

        # test_loss 로깅
        try:
            # Ensure batch_size is available, default to a common value or handle if not critical for prog_bar
            current_batch_size = self.batch_size if hasattr(self, "batch_size") else 1
            test_loss = self.criterion(all_logits.squeeze(), all_targets.float())
            self.log(
                "test_loss",
                test_loss,
                on_epoch=True,
                prog_bar=True,
                batch_size=current_batch_size,
            )
            logger.info(
                f"[TEST_EPOCH_END] Logged PTL metric 'test_loss': {test_loss.item()}"
            )
        except Exception as e:
            logger.error(
                f"[TEST_EPOCH_END] Error calculating or logging test_loss: {e}"
            )
            # Decide if we should proceed or return
            # For now, let's log and proceed to other metrics if possible

        # 다른 메트릭들 계산 및 로깅
        try:
            logits_np = all_logits.numpy()
            targets_np = all_targets.numpy()
            metrics = test_metrics(logits_np, targets_np, prefix="test_")
            logger.info(
                f"[TEST_EPOCH_END] Metrics calculated by calculate_metrics: {list(metrics.keys())}"
            )

            current_batch_size = self.batch_size if hasattr(self, "batch_size") else 1
            for k, v in metrics.items():
                self.log(
                    k, v, on_epoch=True, prog_bar=True, batch_size=current_batch_size
                )
                logger.info(f"[TEST_EPOCH_END] Logged PTL metric '{k}': {v}")

            return {
                "test_loss": test_loss,
                "test_auroc": metrics["test_auroc"],
                "test_f1": metrics["test_f1"],
                "test_TP": metrics["test_TP"],
                "test_TN": metrics["test_TN"],
                "test_FP": metrics["test_FP"],
                "test_FN": metrics["test_FN"],
                "logits": all_logits,
                "target": all_targets,
                "image_ids": all_image_ids,
            }

        except Exception as e:
            logger.error(
                f"[TEST_EPOCH_END] Error calculating or logging other metrics: {e}"
            )

        # 사용된 로그 및 타겟 리스트 초기화
        self.test_logits = []
        self.test_targets = []
        logger.info("[TEST_EPOCH_END] END")

    def configure_optimizers(self):
        logger.info("[CONFIGURE_OPTIMIZERS] START")

        # 옵티마이저 설정 로드 및 생성
        optim_cfg = self.trainer_config_hparams.get(
            "optimizer", {"name": "AdamW", "params": {"lr": 1e-3}}
        )
        optim_name = optim_cfg.get("name", "AdamW")
        optim_params = optim_cfg.get("params", {}).copy()  # 원본 변경 방지를 위해 복사

        if "lr" not in optim_params:  # lr이 명시적으로 없으면 기본값 사용
            optim_params["lr"] = 1e-3
            logger.warning(
                f"Optimizer learning rate not found in config, using default: {optim_params['lr']}"
            )

        if optim_name == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), **optim_params)
        # elif optim_name == "Adam": # 사용자에 의해 삭제된 부분
        #     optimizer = torch.optim.Adam(self.parameters(), **optim_params)
        else:
            logger.error(f"Unsupported optimizer: {optim_name}")
            raise NotImplementedError(f"지원하지 않는 optimizer: {optim_name}")
        logger.info(f"Optimizer '{optim_name}' created with params: {optim_params}")

        # 스케줄러 관련 코드 전체 삭제
        # scheduler_cfg = self.trainer_config_hparams.get("scheduler")

        # if scheduler_cfg and scheduler_cfg.get("name") == "OneCycleLR":
        #    ... (기존 스케줄러 로직 모두 삭제)
        # else:
        #    ... (기존 else 로직 모두 삭제)

        logger.info("[CONFIGURE_OPTIMIZERS] No scheduler will be used as per request.")
        logger.info(f"[CONFIGURE_OPTIMIZERS] Optimizer: {optimizer}")
        logger.info("[CONFIGURE_OPTIMIZERS] END")
        return optimizer
