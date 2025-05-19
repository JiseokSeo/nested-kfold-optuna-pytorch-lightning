# src/train/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, _LRScheduler
from torch.cuda import amp
import numpy as np
import time
import logging
import os
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple

# 프로젝트 모듈 임포트
from .metrics import calculate_metrics
from src.utils.early_stopping import EarlyStopping

# from src.utils.metrics import MetricsLogger # MetricsLogger 클래스는 확인되지 않아 주석 처리 또는 삭제
from src.utils.file_utils import save_json, ensure_dir

# from src.utils.torch_utils import get_device, save_model # torch_utils.py 파일 및 해당 함수 확인되지 않아 주석 처리 또는 삭제
from src.models.multimodal_model import MultiModalModel

logger = logging.getLogger(__name__)


class FoldTrainer:
    """단일 폴드에 대한 학습 및 평가를 처리합니다."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[_LRScheduler],
        device: torch.device,
        config: Dict,
        fold_num: int,
        trial_num: Optional[int] = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.fold = fold_num
        self.trial_num_str = f"Trial {trial_num} | " if trial_num is not None else ""
        self.logger = logger

        self.epochs = config["training"]["epochs"]
        # amp는 설정과 관계없이 CUDA 사용 가능 여부에 따라 결정
        self.use_amp = torch.cuda.is_available() and config["training"].get(
            "use_amp", True
        )
        self.gradient_clip_val = config["training"].get("gradient_clip_val")

        self.early_stopping_patience = config["training"]["early_stopping_patience"]
        self.min_epochs_before_early_stopping = config["training"].get(
            "min_epochs_before_early_stopping", 0
        )
        self.metric_to_monitor = config["training"]["metric_to_monitor"]
        self.monitor_mode = config["training"]["monitor_mode"]
        self.early_stopper = EarlyStopping(
            patience=self.early_stopping_patience,
            mode=self.monitor_mode,
            verbose=True,
            metric_name=self.metric_to_monitor,
            delta=config["training"].get("early_stopping_delta", 0.0),
        )

        self.scaler = amp.GradScaler(enabled=self.use_amp)

        self.best_score = -np.inf if self.monitor_mode == "maximize" else np.inf
        self.best_model_state = None
        self.logger.info(
            f"{self.trial_num_str}Fold {self.fold}: Trainer initialized. AMP: {self.use_amp}, Device: {self.device}"
        )

    def _train_epoch(
        self, train_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """단일 에포크 동안 모델을 학습시킵니다."""
        self.model.train()

        # CUDA 결정성 설정 로깅 추가
        if torch.cuda.is_available():
            self.logger.info(
                f"{self.trial_num_str}Fold {self.fold}: CUDA 결정성 설정 - deterministic: {torch.backends.cudnn.deterministic}, benchmark: {torch.backends.cudnn.benchmark}"
            )

        total_loss = 0.0
        valid_batch_count = 0
        start_time = time.time()

        progress_bar = tqdm(
            train_loader,
            desc=f"{self.trial_num_str}Fold {self.fold} Train",
            leave=False,
            dynamic_ncols=True,
        )

        for batch_idx, batch in enumerate(progress_bar):
            image_batch = batch.get("image")
            csv_batch = batch.get("csv")
            targets = batch.get("target")

            if targets is None:
                self.logger.warning(
                    f"{self.trial_num_str}Fold {self.fold}: Target not found in batch {batch_idx}. Skipping batch."
                )
                continue
            targets = targets.to(self.device, non_blocking=True)

            if image_batch is not None:
                image_batch = image_batch.to(self.device, non_blocking=True)
            if csv_batch is not None:
                if csv_batch.dtype != torch.float32:
                    csv_batch = csv_batch.float()
                csv_batch = csv_batch.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                try:
                    outputs = self.model(image=image_batch, csv=csv_batch)
                except ValueError as e:
                    self.logger.error(
                        f"{self.trial_num_str}Fold {self.fold}: Error during model forward pass at batch {batch_idx}: {e}",
                        exc_info=True,
                    )
                    continue
                except Exception as e:
                    self.logger.error(
                        f"{self.trial_num_str}Fold {self.fold}: Unexpected error during model forward pass at batch {batch_idx}: {e}",
                        exc_info=True,
                    )
                    continue
                try:
                    loss = self.criterion(outputs, targets)
                except Exception as e:
                    self.logger.error(
                        f"{self.trial_num_str}Fold {self.fold}: Error during loss calculation at batch {batch_idx}: {e}",
                        exc_info=True,
                    )
                    continue

            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(
                    f"{self.trial_num_str}Fold {self.fold}: NaN or Inf loss detected at batch {batch_idx}. Skipping optimizer step."
                )
                continue

            if loss.grad_fn is None and loss.requires_grad:
                if loss.item() == 0.0:
                    self.logger.debug(
                        f"{self.trial_num_str}Fold {self.fold}: Loss is 0.0 and has no grad_fn at batch {batch_idx}. Skipping optimizer step."
                    )
                else:
                    self.logger.warning(
                        f"{self.trial_num_str}Fold {self.fold}: Loss has requires_grad=True but no grad_fn at batch {batch_idx} (value: {loss.item():.4f}). Skipping optimizer step."
                    )
                continue

            try:
                self.scaler.scale(loss).backward()

                if self.gradient_clip_val is not None and self.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.scheduler and isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()

                current_loss = loss.item()
                total_loss += current_loss
                valid_batch_count += 1
                progress_bar.set_postfix(loss=f"{current_loss:.4f}")

            except RuntimeError as e:
                self.logger.error(
                    f"{self.trial_num_str}Fold {self.fold}: RuntimeError during backward/step at batch {batch_idx}: {e}",
                    exc_info=True,
                )
                continue
            except Exception as e:
                self.logger.error(
                    f"{self.trial_num_str}Fold {self.fold}: Unexpected error during backward/step at batch {batch_idx}: {e}",
                    exc_info=True,
                )
                continue

        avg_loss = total_loss / valid_batch_count if valid_batch_count > 0 else 0.0
        epoch_time = time.time() - start_time
        self.logger.info(
            f"{self.trial_num_str}Fold {self.fold}: Epoch Train completed. Avg Loss: {avg_loss:.4f}, Valid Batches: {valid_batch_count}/{len(train_loader)}"
        )
        return avg_loss, epoch_time

    def _evaluate_epoch(
        self, valid_loader: torch.utils.data.DataLoader, prefix="val_"
    ) -> Tuple[float, Dict[str, Any]]:
        """단일 에포크 동안 모델을 평가합니다."""
        self.model.eval()
        total_loss = 0.0
        valid_batch_count = 0
        all_preds_logits = []
        all_targets = []

        with torch.no_grad():
            progress_bar = tqdm(
                valid_loader,
                desc=f"{self.trial_num_str}Fold {self.fold} {prefix.strip('_').capitalize()}",
                leave=False,
                dynamic_ncols=True,
            )
            for batch_idx, batch in enumerate(progress_bar):
                image_batch = batch.get("image")
                csv_batch = batch.get("csv")
                targets = batch.get("target")

                if targets is None:
                    self.logger.warning(
                        f"{self.trial_num_str}Fold {self.fold}: Target not found in {prefix} batch {batch_idx}. Skipping."
                    )
                    continue
                targets = targets.to(self.device, non_blocking=True)

                if image_batch is not None:
                    image_batch = image_batch.to(self.device, non_blocking=True)
                if csv_batch is not None:
                    if csv_batch.dtype != torch.float32:
                        csv_batch = csv_batch.float()
                    csv_batch = csv_batch.to(self.device, non_blocking=True)

                try:
                    with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                        outputs = self.model(image=image_batch, csv=csv_batch)
                        loss = self.criterion(outputs, targets)
                except ValueError as e:
                    self.logger.error(
                        f"{self.trial_num_str}Fold {self.fold} Eval: Error during model forward pass at batch {batch_idx}: {e}",
                        exc_info=True,
                    )
                    continue
                except Exception as e:
                    self.logger.error(
                        f"{self.trial_num_str}Fold {self.fold} Eval: Unexpected error during model forward/loss at batch {batch_idx}: {e}",
                        exc_info=True,
                    )
                    continue

                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(
                        f"{self.trial_num_str}Fold {self.fold}: NaN or Inf {prefix}loss detected at batch {batch_idx}. Contribution ignored."
                    )
                    continue

                total_loss += loss.item()
                valid_batch_count += 1
                all_preds_logits.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / valid_batch_count if valid_batch_count > 0 else 0.0

        metrics = {}
        if not all_targets or not all_preds_logits:
            self.logger.warning(
                f"{self.trial_num_str}Fold {self.fold}: No valid batches found for {prefix}evaluation."
            )
            metrics = calculate_metrics(np.array([]), np.array([]), prefix=prefix)
            metrics[f"{prefix}loss"] = avg_loss
        else:
            try:
                all_preds_logits_np = np.concatenate(all_preds_logits, axis=0)
                all_targets_np = np.concatenate(all_targets, axis=0)
                metric_calc_strategy = self.config.get("training", {}).get(
                    "metric_weight_strategy", "balanced"
                )
                metrics = calculate_metrics(
                    all_preds_logits_np,
                    all_targets_np,
                    prefix=prefix,
                    metric_weight_strategy=metric_calc_strategy,
                )
                metrics[f"{prefix}loss"] = avg_loss
            except Exception as e:
                self.logger.error(
                    f"{self.trial_num_str}Fold {self.fold}: Error calculating {prefix}metrics: {e}",
                    exc_info=True,
                )
                metrics = calculate_metrics(np.array([]), np.array([]), prefix=prefix)
                metrics[f"{prefix}loss"] = avg_loss

        return avg_loss, metrics

    def run_fold(self, train_loader, valid_loader):
        """단일 폴드에 대한 전체 학습 및 검증 루프를 실행합니다."""
        self.logger.info(
            f"{self.trial_num_str}Starting Fold {self.fold} Training for {self.epochs} epochs..."
        )

        # 학습 시작 전 CUDA 결정성 확인
        if torch.cuda.is_available():
            if not torch.backends.cudnn.deterministic:
                self.logger.warning(
                    f"{self.trial_num_str}Fold {self.fold}: CUDA 결정성 설정이 비활성화되어 있습니다. 재현성을 보장할 수 없습니다."
                )

        start_fold_time = time.time()

        for epoch in range(self.epochs):
            epoch_num = epoch + 1
            epoch_start_time = time.time()

            train_loss, train_time = self._train_epoch(train_loader)
            val_loss, val_metrics = self._evaluate_epoch(valid_loader, prefix="val_")

            current_lr = self.optimizer.param_groups[0]["lr"]
            log_msg = (
                f"{self.trial_num_str}Fold {self.fold} | Epoch {epoch_num}/{self.epochs} | "
                f"Time: {train_time:.1f}s / {(time.time() - epoch_start_time):.1f}s | "
                f"LR: {current_lr:.1e} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            )
            main_metric_val = val_metrics.get(self.metric_to_monitor)
            if isinstance(main_metric_val, (float, np.number)) and not np.isnan(
                main_metric_val
            ):
                log_msg += f"{self.metric_to_monitor}: {main_metric_val:.4f}"
            else:
                log_msg += f"{self.metric_to_monitor}: N/A"
            self.logger.info(log_msg)

            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    current_metric = val_metrics.get(self.metric_to_monitor)
                    if (
                        current_metric is not None
                        and isinstance(current_metric, (float, np.number))
                        and not np.isnan(current_metric)
                    ):
                        self.scheduler.step(current_metric)
                    elif current_metric is None:
                        self.logger.warning(
                            f"Metric '{self.metric_to_monitor}' not found for ReduceLROnPlateau step at epoch {epoch_num}."
                        )
                elif not isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()

            monitored_metric_value = val_metrics.get(
                self.metric_to_monitor,
                val_loss if "loss" in self.metric_to_monitor.lower() else None,
            )

            if monitored_metric_value is None:
                self.logger.warning(
                    f"{self.trial_num_str}Fold {self.fold}: Monitored metric '{self.metric_to_monitor}' not found in validation metrics. Using validation loss for early stopping."
                )
                monitored_metric_value = val_loss

            if epoch >= self.min_epochs_before_early_stopping:
                if self.early_stopper.step(
                    monitored_metric_value,
                    {k: v.clone().cpu() for k, v in self.model.state_dict().items()},
                ):
                    self.logger.info(
                        f"{self.trial_num_str}Fold {self.fold}: Early stopping triggered at epoch {epoch + 1}. Best {self.metric_to_monitor}: {self.early_stopper.best_score:.4f}"
                    )
                    self.best_model_state = self.early_stopper.get_best_model_state()
                    break
            else:
                self.early_stopper.update_best(
                    monitored_metric_value,
                    {k: v.clone().cpu() for k, v in self.model.state_dict().items()},
                )
                if self.early_stopper.best_score is not None and (
                    (
                        self.early_stopper.mode == "maximize"
                        and self.early_stopper.best_score > self.best_score
                    )
                    or (
                        self.early_stopper.mode == "minimize"
                        and self.early_stopper.best_score < self.best_score
                    )
                ):
                    self.logger.debug(
                        f"{self.trial_num_str}Fold {self.fold}: Epoch {epoch+1}: New best score ({self.early_stopper.best_score:.4f}) before min_epochs (within EarlyStopping). Updating FoldTrainer's best model state."
                    )
                    self.best_score = self.early_stopper.best_score
                    self.best_model_state = self.early_stopper.get_best_model_state()

            current_best_from_stopper = self.early_stopper.best_score
            if current_best_from_stopper is not None:
                is_trainer_better = False
                if self.early_stopper.mode == "maximize":
                    is_trainer_better = current_best_from_stopper > self.best_score
                else:  # minimize
                    is_trainer_better = current_best_from_stopper < self.best_score

                if is_trainer_better:
                    self.logger.info(
                        f"{self.trial_num_str}Fold {self.fold}: Epoch {epoch+1}: New best {self.metric_to_monitor} for FoldTrainer: {current_best_from_stopper:.4f} (was {self.best_score:.4f})"
                    )
                    self.best_score = current_best_from_stopper
                    self.best_model_state = self.early_stopper.get_best_model_state()

            if self.early_stopper.early_stop:
                break

        self.logger.info(
            f"{self.trial_num_str}Fold {self.fold} finished training. Best {self.metric_to_monitor} (val): {self.best_score:.4f}. Total time: {time.time() - start_fold_time:.2f}s"
        )

        return self.best_score, self.best_model_state

    def test_fold(self, test_loader):
        """폴드의 최적 모델을 사용하여 테스트 세트에서 평가하고 예측 결과를 반환합니다."""
        if self.best_model_state is None:
            self.logger.warning(
                f"{self.trial_num_str}Fold {self.fold}: No best model state found to test."
            )
            return {}, {}

        best_score_str = (
            f"{self.best_score:.4f}"
            if self.best_score is not None and not np.isinf(self.best_score)
            else "N/A"
        )

        self.logger.info(
            f"{self.trial_num_str}Testing Fold {self.fold} with best model (Val Score: {best_score_str})..."
        )
        try:
            self.model.load_state_dict(self.best_model_state)
        except Exception as e:
            self.logger.error(
                f"Error loading best model state dict for fold {self.fold}: {e}",
                exc_info=True,
            )
            return {}, {}

        # 평가 모드로 전환 전 CUDA 결정성 확인
        if torch.cuda.is_available():
            self.logger.info(
                f"{self.trial_num_str}Fold {self.fold}: 테스트 실행 중 CUDA 결정성 설정 - deterministic: {torch.backends.cudnn.deterministic}, benchmark: {torch.backends.cudnn.benchmark}"
            )

        test_loss, test_metrics = self._evaluate_epoch(test_loader, prefix="test_")
        self.logger.info(
            f"{self.trial_num_str}Fold {self.fold} Test Results: {test_metrics}"
        )

        predictions_data = {
            "ids": [],
            "targets": [],
            "predictions": [],
            "metrics": test_metrics,
        }
        try:
            self.model.eval()
            all_preds_logits = []
            all_targets = []
            all_ids = []
            with torch.no_grad():
                for batch in tqdm(
                    test_loader,
                    desc=f"{self.trial_num_str}Fold {self.fold} Test Preds",
                    leave=False,
                    dynamic_ncols=True,
                ):
                    image_batch = batch.get("image")
                    csv_batch = batch.get("csv")
                    targets = batch.get("target")
                    ids = batch.get("id")

                    if targets is None:
                        continue

                    if image_batch is not None:
                        image_batch = image_batch.to(self.device, non_blocking=True)
                    if csv_batch is not None:
                        if csv_batch.dtype != torch.float32:
                            csv_batch = csv_batch.float()
                        csv_batch = csv_batch.to(self.device, non_blocking=True)

                    with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
                        outputs = self.model(image=image_batch, csv=csv_batch)

                    all_preds_logits.append(outputs.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                    if ids:
                        all_ids.extend(ids)

            if all_ids:
                predictions_data["ids"] = all_ids
                if all_targets:
                    predictions_data["targets"] = (
                        np.concatenate(all_targets, axis=0).flatten().tolist()
                    )
                if all_preds_logits:
                    predictions_data["predictions"] = (
                        np.concatenate(all_preds_logits, axis=0).flatten().tolist()
                    )

        except Exception as e:
            self.logger.error(
                f"Error getting predictions for saving in test_fold: {e}", exc_info=True
            )

        return test_metrics, predictions_data
