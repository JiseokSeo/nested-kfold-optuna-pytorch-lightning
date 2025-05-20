import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.utils import get_fitted_scaler, do_scaling, save_scaler
from src.model_modules.builder import build_model_from_config
from src.data_modules.builder import build_data
from src.trainer.builder import build_trainer_from_config
from src.trainer.callbacks import InMemoryBestModelSaver
from pytorch_lightning.callbacks import EarlyStopping
import os
import logging
import torch
import torch.nn as nn
import lightning as L

logger = logging.getLogger(__name__)


def objective(
    trial,
    config,
    train_data,
    y,
    n_inner_folds,
    build_model_fn=build_model_from_config,
    build_trainer_fn=build_trainer_from_config,
    build_data_fn=build_data,
    metric_key="val_loss",
    reproducible: bool = False,
    debug: bool = False,
):
    """
    Optuna objective 함수. 빌더 함수들을 인자로 받아 동적으로 객체를 생성.
    """
    from configs.search_range import get_trial_params
    from src.utils import merge_config

    trial_params = get_trial_params(trial, config)
    trial_config = merge_config(config, trial_params)
    inner_kf = StratifiedKFold(
        n_splits=n_inner_folds, shuffle=True, random_state=config.get("seed", 42)
    )
    inner_scores = []
    best_epochs = []

    for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(
        inner_kf.split(train_data, y)
    ):

        inner_train = train_data.iloc[inner_train_idx]
        inner_val = train_data.iloc[inner_val_idx]
        if debug:
            logger.debug(
                f"[DEBUG] Inner fold {inner_fold_idx} input data 샘플:\n{inner_train.head(3)}"
            )
            logger.debug(
                f"[DEBUG] Inner fold {inner_fold_idx} input data shape: {inner_train.shape}"
            )
        logger.info(
            f"[Objective] Inner fold {inner_fold_idx}: inner_train shape: {inner_train.shape}, inner_val shape: {inner_val.shape}"
        )
        if inner_val.empty:
            logger.warning(
                f"[Objective] CRITICAL: inner_val is EMPTY for inner_fold {inner_fold_idx}. Validation will likely not run or be effective."
            )

        model = build_model_fn(trial_config)
        logger.info(f"[Objective] Model built: {type(model).__name__}")

        in_memory_callback = InMemoryBestModelSaver(monitor="val_loss", mode="min")
        early_stopping_callback = EarlyStopping(
            monitor=trial_config["trainer"]["metric_to_monitor"],
            patience=trial_config["trainer"]["early_stopping_patience"],
            mode=trial_config["trainer"]["monitor_mode"],
            verbose=False,
        )
        callbacks = [in_memory_callback, early_stopping_callback]

        trainer = build_trainer_fn(
            trial_config,
            fold_idx=inner_fold_idx,
            reproducible=reproducible,
            debug=debug,
            callbacks=callbacks,
        )

        scaler = get_fitted_scaler(trial_config, inner_train)
        use_csv_flag = trial_config.get("data", {}).get("use_csv", False)

        if scaler is not None:
            scaler_save_dir = os.path.join(
                trial_config["dir"]["scaler_save_path"], f"fold_{inner_fold_idx}"
            )
            os.makedirs(scaler_save_dir, exist_ok=True)
            save_scaler(scaler, os.path.join(scaler_save_dir, "scaler.pkl"))
            logger.info(f"Scaler saved for inner_fold {inner_fold_idx}.")
        else:
            if use_csv_flag:
                logger.error(
                    f"CRITICAL: Scaler is None for inner_fold {inner_fold_idx} even though 'data.use_csv' is True. "
                    "This indicates an issue with 'columns_to_scale', the data itself, or the scaler fitting process. "
                    "Refer to logs from 'get_fitted_scaler' in 'utils.py' for more details."
                )
                raise ValueError(
                    f"Scaler is None for inner_fold {inner_fold_idx} despite 'data.use_csv' being True. "
                    "This implies an issue with data preprocessing or configuration for CSV data."
                )
            else:
                logger.info(
                    f"Scaler is None for inner_fold {inner_fold_idx} because 'data.use_csv' is False. "
                    "This is expected, and scaler fitting/saving will be skipped."
                )

        scaled_inner_train = do_scaling(
            trial_config.get("data", {}).get("columns_to_scale", []),
            inner_train,
            scaler,
        )
        logger.info(
            f"[Objective] scaled_inner_train shape: {scaled_inner_train.shape if scaled_inner_train is not None else 'None'}"
        )
        if scaled_inner_train is not None and scaled_inner_train.empty:
            logger.warning(
                f"[Objective] CRITICAL: scaled_inner_train is EMPTY for inner_fold {inner_fold_idx}. Training will likely not run."
            )

        scaled_inner_val = do_scaling(
            trial_config.get("data", {}).get("columns_to_scale", []), inner_val, scaler
        )
        train_dataloader = build_data_fn(trial_config, scaled_inner_train, mode="train")
        logger.info(
            f"[Objective] train_dataloader created for inner_fold {inner_fold_idx}. Type: {type(train_dataloader)}"
        )
        if train_dataloader is not None:
            try:
                logger.info(
                    f"[Objective] train_dataloader.dataset length: {len(train_dataloader.dataset) if hasattr(train_dataloader, 'dataset') else 'N/A (no dataset attr)'}"
                )
                logger.info(
                    f"[Objective] train_dataloader batch_size: {train_dataloader.batch_size if hasattr(train_dataloader, 'batch_size') else 'N/A'}"
                )
                logger.info(
                    f"[Objective] len(train_dataloader) (num batches): {len(train_dataloader)}"
                )
                if len(train_dataloader) > 0:
                    logger.info(
                        f"[Objective] train_dataloader appears to have batches."
                    )
                else:
                    logger.warning(
                        f"[Objective] train_dataloader has 0 batches. Training will likely be skipped or ineffective."
                    )
            except Exception as e:
                logger.error(
                    f"[Objective] Error inspecting train_dataloader for inner_fold {inner_fold_idx}: {e}"
                )
        else:
            logger.warning(
                f"[Objective] train_dataloader is None for inner_fold {inner_fold_idx}. Training will be skipped."
            )

        val_dataloader = build_data_fn(trial_config, scaled_inner_val, mode="val")
        logger.info(
            f"[Objective] val_dataloader created for inner_fold {inner_fold_idx}. Type: {type(val_dataloader)}"
        )
        if val_dataloader is not None:
            try:
                logger.info(
                    f"[Objective] val_dataloader.dataset length: {len(val_dataloader.dataset) if hasattr(val_dataloader, 'dataset') else 'N/A (no dataset attr)'}"
                )
                logger.info(
                    f"[Objective] val_dataloader batch_size: {val_dataloader.batch_size if hasattr(val_dataloader, 'batch_size') else 'N/A'}"
                )
                logger.info(
                    f"[Objective] len(val_dataloader) (num batches): {len(val_dataloader)}"
                )
                if len(val_dataloader) > 0:
                    logger.info(f"[Objective] val_dataloader appears to have batches.")
                else:
                    logger.warning(
                        f"[Objective] val_dataloader has 0 batches. Validation might be skipped or ineffective."
                    )
            except Exception as e:
                logger.error(
                    f"[Objective] Error inspecting val_dataloader for inner_fold {inner_fold_idx}: {e}"
                )
        else:
            logger.warning(
                f"[Objective] val_dataloader is None for inner_fold {inner_fold_idx}. Validation will be skipped."
            )

        if debug or True:
            logger.info(f"[INFO] Trainer.fit() 시작 (inner_fold {inner_fold_idx})")
        try:
            trainer.fit(model, train_dataloader, val_dataloader)
            logger.info(
                f"[OBJECTIVE_FIT_END] Trial {trial.number if trial else 'N/A'}, Inner Fold {inner_fold_idx}: Finished trainer.fit() for model {type(model).__name__}"
            )
        except RuntimeError as e:
            logger.error(
                f"[OBJECTIVE_RUNTIME_ERROR] Trial {trial.number if trial else 'N/A'}, Inner Fold {inner_fold_idx}: Runtime error during trainer.fit(). Error: {str(e)}"
            )
            if "Early stopping conditioned on metric" in str(
                e
            ) and "which is not available" in str(e):
                logger.error("EarlyStopping metric issue detected during fit.")
            raise
        except Exception as e:
            logger.error(
                f"[OBJECTIVE_UNEXPECTED_ERROR] Trial {trial.number if trial else 'N/A'}, Inner Fold {inner_fold_idx}: Unexpected error during trainer.fit(). Error: {type(e).__name__} - {str(e)}"
            )
            logger.exception(
                f"[OBJECTIVE_UNEXPECTED_ERROR_STACKTRACE] Trial {trial.number if trial else 'N/A'}, Inner Fold {inner_fold_idx}"
            )
            raise

        if debug or True:
            logger.info(f"[INFO] 최종모델 검증 시작 (inner_fold {inner_fold_idx})")
        metrics = trainer.test(model, val_dataloader)
        inner_scores.append(metrics["test_auroc"])

        # InMemoryBestModelSaver.best_epoch는 텐서일 수 있으므로 .item()으로 Python 숫자로 변환
        current_best_epoch = in_memory_callback.best_epoch
        if torch.is_tensor(current_best_epoch):
            best_epochs.append(current_best_epoch.item())
        elif isinstance(current_best_epoch, (int, float)):
            best_epochs.append(current_best_epoch)  # 이미 숫자면 그대로 사용
        else:  # 예상치 못한 타입이면 -1 또는 적절한 기본값 사용 및 경고
            logger.warning(
                f"Unexpected type for best_epoch: {type(current_best_epoch)}. Storing -1."
            )
            best_epochs.append(-1)

    trial.set_user_attr("inner_scores", inner_scores)
    trial.set_user_attr("best_epochs", best_epochs)
    # best_epoch 저장
    final_score = np.mean(inner_scores)
    logger.info(
        f"[Objective] Trial finished. Average score over inner folds (val_auc): {final_score}"
    )
    return final_score
