import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.utils import get_fitted_scaler, do_scaling, save_scaler
from src.model_modules.builder import build_model_from_config
from src.data_modules.builder import build_data
from src.trainer.builder import build_trainer_from_config
from src.trainer.scheduler_utils import get_one_cycle_lr_params
from src.trainer.callbacks import BestModelCallback
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
    checkpoint_for_final_logging = None

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

        trainer_instance, current_checkpoint = build_trainer_fn(
            trial_config,
            fold_idx=inner_fold_idx,
            reproducible=reproducible,
            debug=debug,
        )
        if inner_fold_idx == n_inner_folds - 1:
            checkpoint_for_final_logging = current_checkpoint

        trial_config["_trainer_instance"] = trainer_instance

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

        use_scheduler_flag = trial_config.get("trainer", {}).get("use_scheduler", True)

        if use_scheduler_flag:
            scheduler_params = get_one_cycle_lr_params(trial_config, train_dataloader)
            if scheduler_params:
                logger.info(f"Objective: Setting scheduler params: {scheduler_params}")
                model.set_scheduler_params(**scheduler_params)
            else:
                logger.warning(
                    "Objective: Scheduler params not set or not applicable (e.g., scheduler name is None or utils returned None). Scheduler will not be set."
                )
        else:
            logger.info("Objective: use_scheduler is False. Skipping scheduler setup.")

        if "_trainer_instance" in trial_config:
            del trial_config["_trainer_instance"]

        if (
            current_checkpoint
            and hasattr(current_checkpoint, "best_model_state_dict")
            and current_checkpoint.best_model_state_dict
        ):
            logger.info(
                f"[Objective] Loading best model state_dict from checkpoint for model {type(model).__name__}"
            )
            model.load_state_dict(current_checkpoint.best_model_state_dict)

        if debug or True:
            logger.info(f"[INFO] Trainer.fit() 시작 (inner_fold {inner_fold_idx})")
        try:
            trainer_instance.fit(model, train_dataloader, val_dataloader)
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

        best_model_cb = None
        for cb in trainer_instance.callbacks:
            if isinstance(cb, BestModelCallback):
                best_model_cb = cb
                break

        if best_model_cb and best_model_cb.best_model_state_dict:
            logger.info(
                f"[Objective] Loading best model state_dict for validation. Best score: {best_model_cb.best_score} at epoch {best_model_cb.best_epoch}"
            )
            model.load_state_dict(best_model_cb.best_model_state_dict)
        else:
            logger.error(
                f"[Objective] CRITICAL: No best model state_dict found from BestModelCallback for {type(model).__name__}. "
                f"Callback state: monitor='{best_model_cb.monitor if best_model_cb else 'N/A'}', score={best_model_cb.best_score if best_model_cb else 'N/A'}"
            )

        if debug or True:
            logger.info(f"[INFO] Trainer.validate() 시작 (inner_fold {inner_fold_idx})")
        val_metrics = trainer_instance.validate(
            model, dataloaders=val_dataloader, verbose=False
        )
        if debug or True:
            logger.info(f"[INFO] Trainer.validate() 종료 (inner_fold {inner_fold_idx})")
        if debug:
            logger.debug(f"[DEBUG] Validation metrics: {val_metrics}")
            if hasattr(model, "val_logits"):
                logger.debug(
                    f"[DEBUG] 예측 샘플: {getattr(model, 'val_logits', [])[:3]}"
                )

        current_score = 0.0
        if val_metrics:
            fetched_score = val_metrics[0].get(metric_key)

            if fetched_score is not None:
                current_score = fetched_score
                logger.info(
                    f"[Objective] Using '{metric_key}' for current_score: {current_score}"
                )
            else:
                logger.warning(
                    f"[Objective] Metric '{metric_key}' not found in val_metrics[0]: {val_metrics[0].keys()}. Trying 'val_loss'."
                )
                current_score = val_metrics[0].get("val_loss", 0.0)
                logger.info(
                    f"[Objective] Used fallback 'val_loss' for current_score: {current_score}"
                )
        else:
            logger.warning(
                f"[Objective] trainer.validate() returned empty metrics list for inner_fold {inner_fold_idx}. Using score 0.0."
            )

        logger.info(
            f"[Objective] Inner fold {inner_fold_idx} completed. Score: {current_score}"
        )
        inner_scores.append(current_score)

    trial.set_user_attr("inner_scores", inner_scores)
    # best_epoch 저장
    if best_model_cb is not None:
        trial.set_user_attr("best_epoch", best_model_cb.best_epoch)
    final_score = np.mean(inner_scores)
    logger.info(
        f"[Objective] Trial finished. Average score over inner folds ({metric_key}): {final_score}"
    )
    return final_score
