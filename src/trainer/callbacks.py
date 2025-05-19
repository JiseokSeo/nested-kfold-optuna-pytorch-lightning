import os
import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping
import torch
import logging

logger = logging.getLogger(__name__)


class BestModelCallback(Callback):
    """
    검증 성능에 따라 최적 모델의 state_dict를 메모리에 저장하는 콜백.
    """

    def __init__(self, monitor: str, mode: str = "min", verbose: bool = False):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best_score = float("-inf") if self.mode == "max" else float("inf")
        self.best_model_state_dict = None
        self.current_epoch_score = None
        self.best_epoch = -1

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        logger.info(
            f"[BestModelCallback] on_validation_end called at epoch {trainer.current_epoch}. Monitoring: '{self.monitor}', Mode: '{self.mode}'"
        )
        logs = trainer.callback_metrics
        if not logs:
            logger.warning(
                f"[BestModelCallback] trainer.callback_metrics is empty. Cannot get score for '{self.monitor}'."
            )
            return

        logger.info(
            f"[BestModelCallback] Available metrics in trainer.callback_metrics: {list(logs.keys())}"
        )
        score = logs.get(self.monitor)
        logger.info(f"[BestModelCallback] Score for '{self.monitor}': {score}")

        if score is None:
            if self.verbose:
                logger.warning(
                    f"BestModelCallback: Metric '{self.monitor}' not found in logs. Skipping."
                )
            return

        self.current_epoch_score = score
        improved_score = (self.mode == "max" and score > self.best_score) or (
            self.mode == "min" and score < self.best_score
        )

        if self.best_model_state_dict is None or improved_score:
            self.best_score = score
            self.best_model_state_dict = {
                k: v.cpu().clone() for k, v in pl_module.state_dict().items()
            }
            self.best_epoch = trainer.current_epoch
            if self.verbose:
                print(
                    f"BestModelCallback: New best model saved with {self.monitor}={self.best_score:.4f} at epoch {self.best_epoch}"
                )

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if self.verbose and self.best_model_state_dict is not None:
            print(
                f"BestModelCallback: Training finished. Best model had {self.monitor}={self.best_score:.4f}."
            )
        elif self.verbose:
            print(
                "BestModelCallback: Training finished, but no best model was saved (possibly due to metric not found or no improvement)."
            )


def get_callbacks_from_config(config, fold_idx=None):
    trainer_config = config["trainer"]
    callbacks = []

    # 설정 파일에서 metric_to_monitor 값을 가져옴. 기본값은 "val_loss"
    monitor_metric = trainer_config.get("metric_to_monitor", "val_loss")
    # val_custom을 val_loss로 강제 변경하는 로직을 완전히 제거합니다.
    # 이제 config 파일에서 "metric_to_monitor": "val_custom"으로 설정하면
    # monitor_metric은 "val_custom"이 됩니다.

    early_stopping = EarlyStopping(
        monitor=monitor_metric,  # config 파일의 값을 그대로 사용
        mode=trainer_config["monitor_mode"],
        patience=trainer_config["early_stopping_patience"],
        min_delta=trainer_config.get("early_stopping_delta", 0.0),
        verbose=True,
    )
    callbacks.append(early_stopping)

    best_model_callback = BestModelCallback(
        monitor=monitor_metric,
        mode=trainer_config["monitor_mode"],
        verbose=True,
    )
    callbacks.append(best_model_callback)

    return callbacks
