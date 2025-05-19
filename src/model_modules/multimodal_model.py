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
from src.trainer.metrics import calculate_metrics


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

        self.one_cycle_total_steps = None
        self.one_cycle_max_lr = None
        self.one_cycle_additional_params = {}
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

        # 다른 메트릭들 계산 및 로깅
        logits_np = all_logits.numpy()
        targets_np = all_targets.numpy()
        metrics = calculate_metrics(logits_np, targets_np, prefix="val_")
        logger.info(
            f"[V_EPOCH_END] Metrics calculated by calculate_metrics: {list(metrics.keys())}"
        )

        monitored_metric_name = self.trainer_config_hparams.get(
            "metric_to_monitor", "val_custom"
        )
        logger.info(
            f"[V_EPOCH_END] Metric to monitor (from config trainer.metric_to_monitor): '{monitored_metric_name}'"
        )

        current_batch_size = self.batch_size
        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, prog_bar=True, batch_size=current_batch_size)
            logger.info(f"[V_EPOCH_END] Logged PTL metric '{k}': {v}")

        # EarlyStopping이 사용할 monitored_metric_name이 실제로 로깅되었는지 확인
        if monitored_metric_name in metrics:
            logger.info(
                f"[V_EPOCH_END] Monitored metric '{monitored_metric_name}' (value: {metrics[monitored_metric_name]}) is present in calculated metrics and was logged."
            )
        # 기존의 val_custom <-> val_auroc 연동 로직 (만약 monitored_metric_name이 'val_custom'이고, 'val_auroc'만 계산된 경우)
        elif (
            monitored_metric_name == "val_custom"
            and "val_auroc" in metrics
            and "val_custom" not in metrics
        ):
            val_auroc_value = metrics["val_auroc"]
            self.log(
                "val_custom",
                val_auroc_value,
                on_epoch=True,
                prog_bar=True,
                batch_size=current_batch_size,
            )
            logger.info(
                f"[V_EPOCH_END] Monitored metric was 'val_custom', not in metrics. Logged 'val_custom' using 'val_auroc' value: {val_auroc_value}"
            )
        else:
            logger.warning(
                f"[V_EPOCH_END] Monitored metric '{monitored_metric_name}' was NOT FOUND in calculated metrics ({list(metrics.keys())}). "
                f"EarlyStopping might fail if it relies on this metric name directly via self.log()."
            )

        # 콜백이 접근 가능한 최종 메트릭 목록 로깅
        if (
            self.trainer
            and hasattr(self.trainer, "callback_metrics")
            and self.trainer.callback_metrics
        ):
            logger.info(
                f"[V_EPOCH_END] Final metrics available in trainer.callback_metrics: {list(self.trainer.callback_metrics.keys())}"
            )
            if monitored_metric_name not in self.trainer.callback_metrics:
                logger.warning(
                    f"[V_EPOCH_END] CRITICAL: Monitored metric '{monitored_metric_name}' IS NOT in trainer.callback_metrics! EarlyStopping WILL LIKELY FAIL."
                )
            else:
                logger.info(
                    f"[V_EPOCH_END] Monitored metric '{monitored_metric_name}' IS PRESENT in trainer.callback_metrics."
                )
        else:
            logger.info(
                "[V_EPOCH_END] trainer.callback_metrics not available or empty at the end of on_validation_epoch_end."
            )

        # 사용된 로그 및 타겟 리스트 초기화
        self.val_logits = []
        self.val_targets = []

    def test_step(self, batch, batch_idx):
        image = batch["image"]
        csv = batch["csv"]
        target = batch["target"]
        logits = self.forward(image=image, csv=csv)
        loss = self.criterion(logits.squeeze(), target.float())
        current_batch_size = self.batch_size
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=current_batch_size,
        )
        logits_np = logits.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        metrics = calculate_metrics(logits_np, target_np, prefix="test_")
        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, prog_bar=True, batch_size=current_batch_size)
        print(f"test_loss: {loss}, test_auroc: {metrics['test_auroc']}")
        return {"test_loss": loss, "logits": logits.detach(), "target": target.detach()}

    def set_scheduler_params(
        self, total_steps: int, max_lr: float, additional_params: Optional[Dict] = None
    ):
        self.one_cycle_total_steps = total_steps
        self.one_cycle_max_lr = max_lr
        if additional_params:
            self.one_cycle_additional_params = additional_params

    def configure_optimizers(self):
        logger.info("[CONFIGURE_OPTIMIZERS] START")
        optim_cfg = self.trainer_config_hparams.get(
            "optimizer", {"name": "AdamW", "params": {}}
        )
        optim_name = optim_cfg.get("name", "AdamW")

        default_lr = optim_cfg.get("params", {}).get("lr", 1e-3)

        current_max_lr_for_scheduler = (
            self.one_cycle_max_lr if self.one_cycle_max_lr is not None else default_lr
        )

        optimizer_initial_lr = default_lr

        final_optim_params = optim_cfg.get("params", {}).copy()
        final_optim_params["lr"] = optimizer_initial_lr

        if optim_name == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), **final_optim_params)
        elif optim_name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), **final_optim_params)
        else:
            raise NotImplementedError(f"지원하지 않는 optimizer: {optim_name}")

        # hparams에서 use_scheduler 플래그 확인 (기본값 True로 설정하여 호환성 유지)
        use_scheduler_flag = self.trainer_config_hparams.get("use_scheduler", True)

        if not use_scheduler_flag:
            print(
                "configure_optimizers: use_scheduler is False. Returning optimizer only."
            )
            logger.info(
                "[CONFIGURE_OPTIMIZERS] use_scheduler is False. Returning optimizer only."
            )
            logger.info(f"[CONFIGURE_OPTIMIZERS] Optimizer object: {optimizer}")
            return optimizer

        scheduler_cfg_from_hparams = self.trainer_config_hparams.get("scheduler")

        if (
            self.one_cycle_total_steps
            and self.one_cycle_max_lr
            and scheduler_cfg_from_hparams
            and scheduler_cfg_from_hparams.get("name") == "OneCycleLR"
        ):
            print(
                f"MultimodalModel: Configuring OneCycleLR scheduler in configure_optimizers with max_lr={current_max_lr_for_scheduler}, total_steps={self.one_cycle_total_steps}"
            )

            defined_scheduler_params = scheduler_cfg_from_hparams.get(
                "params", {}
            ).copy()
            defined_scheduler_params.update(self.one_cycle_additional_params)

            lr_scheduler_config = {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=current_max_lr_for_scheduler,
                    total_steps=self.one_cycle_total_steps,
                    **defined_scheduler_params,
                ),
                "interval": "step",
                "frequency": 1,
                "name": "one_cycle_lr_model_managed",
            }
            logger.info(f"[CONFIGURE_OPTIMIZERS] Optimizer object: {optimizer}")
            logger.info(
                f"[CONFIGURE_OPTIMIZERS] LR Scheduler config: {lr_scheduler_config}"
            )
            return [optimizer], [lr_scheduler_config]
        else:
            print(
                "configure_optimizers: Scheduler conditions not met or scheduler not OneCycleLR. Returning optimizer only."
            )
            logger.info(
                "[CONFIGURE_OPTIMIZERS] Scheduler conditions not met or scheduler not OneCycleLR. Returning optimizer only."
            )
            logger.info(f"[CONFIGURE_OPTIMIZERS] Optimizer object: {optimizer}")
            return optimizer
