import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import logging  # 표준 로깅 모듈 import

# 이 파일 전용 로거 생성
builder_file_logger = logging.getLogger(__name__)


def build_trainer_from_config(
    config,
    fold_idx=None,
    reproducible: bool = False,
    debug: bool = False,
    callbacks=None,
):
    trainer_config = config["trainer"]
    log_save_dir = config["dir"]["log_save_path"]
    exp_name = config["experiment_name"]

    logger_name = exp_name
    if fold_idx is not None:
        logger_name = f"{exp_name}/fold_{fold_idx}"

    logger = TensorBoardLogger(
        save_dir=log_save_dir, name=logger_name, default_hp_metric=False
    )
    # TensorBoardLogger 객체에 .info()를 호출하는 대신, 파일 스코프의 로거를 사용
    builder_file_logger.info(
        f"[TrainerBuilder] TensorBoardLogger (PTL) created at {log_save_dir}/{logger_name}"
    )

    torch.backends.cudnn.benchmark = not reproducible

    gradient_clip_value = trainer_config.get("gradient_clip_val")
    min_epochs_setting = trainer_config.get("min_epochs_before_early_stopping", 1)
    patience = trainer_config.get("early_stopping_patience", 10)

    fast_dev_run = True if debug else False
    trainer = L.Trainer(
        max_epochs=trainer_config["epochs"],
        min_epochs=min_epochs_setting,
        accelerator="auto",
        devices=1,
        deterministic=reproducible,
        logger=logger,
        enable_model_summary=True,
        enable_progress_bar=True,
        log_every_n_steps=1,
        enable_checkpointing=True,
        gradient_clip_val=gradient_clip_value,
        num_sanity_val_steps=0,
        fast_dev_run=fast_dev_run,
        callbacks=callbacks,
    )
    # TensorBoardLogger 객체에 .info()를 호출하는 대신, 파일 스코프의 로거를 사용
    builder_file_logger.info(
        f"[TrainerBuilder] L.Trainer object created. Max epochs: {trainer.max_epochs}, Min epochs: {trainer.min_epochs}"
    )

    return trainer
