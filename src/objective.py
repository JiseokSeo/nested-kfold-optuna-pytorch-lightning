import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.utils import get_fitted_scaler, do_scaling, save_scaler
from src.model_modules.builder import build_model_from_config
from src.data_modules.builder import build_data
from src.trainer.builder import build_trainer_from_config
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
    Optuna objective 함수. ModelCheckpoint를 사용하여 최적 모델 가중치를 관리.
    """
    from configs.search_range import get_trial_params
    from src.utils import merge_config

    trial_params = get_trial_params(trial, config)
    trial_config = merge_config(config, trial_params)
    inner_kf = StratifiedKFold(
        n_splits=n_inner_folds, shuffle=True, random_state=config.get("seed", 42)
    )
    inner_scores = []
    all_inner_best_epochs_approx = []

    for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(
        inner_kf.split(train_data, y)
    ):
        logger.info(
            f"[Objective] Starting Inner Fold {inner_fold_idx + 1}/{n_inner_folds}"
        )

        inner_train_df = train_data.iloc[inner_train_idx]
        inner_val_df = train_data.iloc[inner_val_idx]
        if debug:
            logger.debug(
                f"[DEBUG] Inner fold {inner_fold_idx} input data 샘플:\n{inner_train_df.head(3)}"
            )
            logger.debug(
                f"[DEBUG] Inner fold {inner_fold_idx} input data shape: {inner_train_df.shape}"
            )
        logger.info(
            f"[Objective] Inner fold {inner_fold_idx}: inner_train shape: {inner_train_df.shape}, inner_val shape: {inner_val_df.shape}"
        )
        if inner_val_df.empty:
            logger.warning(
                f"[Objective] CRITICAL: inner_val is EMPTY for inner_fold {inner_fold_idx}. Validation will likely not run or be effective."
            )

        model = build_model_fn(trial_config)
        logger.info(
            f"[Objective] Inner Fold {inner_fold_idx}: Model built: {type(model).__name__}"
        )

        monitor_metric = trial_config["trainer"].get("metric_to_monitor", "val_loss")
        monitor_mode = trial_config["trainer"].get("monitor_mode", "min")

        early_stopping_callback = L.pytorch.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=trial_config["trainer"]["early_stopping_patience"],
            mode=monitor_mode,
            verbose=False,
        )

        checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=1,
            save_weights_only=True,
            dirpath=None,
            filename=f"best_model_trial_{trial.number if trial else 'na'}_fold_{inner_fold_idx}",
        )
        callbacks = [early_stopping_callback, checkpoint_callback]

        trainer = build_trainer_fn(
            trial_config,
            fold_idx=inner_fold_idx,
            reproducible=reproducible,
            debug=debug,
            callbacks=callbacks,
        )

        scaler = get_fitted_scaler(trial_config, inner_train_df)
        use_csv_flag = trial_config.get("data", {}).get("use_csv", False)

        if scaler is not None:
            scaler_save_dir = os.path.join(
                trial_config["dir"].get(
                    "scaler_save_path", "results/default_scaler_path"
                ),
                f"trial_{trial.number if trial else 'N_A'}",
                f"outer_fold_TODO",
                f"inner_fold_{inner_fold_idx}",
            )
            os.makedirs(scaler_save_dir, exist_ok=True)
            save_scaler(scaler, os.path.join(scaler_save_dir, "scaler.pkl"))
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
            inner_train_df,
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
            trial_config.get("data", {}).get("columns_to_scale", []),
            inner_val_df,
            scaler,
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

        loaded_best_weights = False
        if (
            hasattr(checkpoint_callback, "best_model_state_dict")
            and checkpoint_callback.best_model_state_dict
        ):
            try:
                model.load_state_dict(checkpoint_callback.best_model_state_dict)
                logger.info(
                    f"[Objective] Inner Fold {inner_fold_idx}: Loaded best model weights from ModelCheckpoint's in-memory state_dict for testing."
                )
                loaded_best_weights = True
            except Exception as e:
                logger.warning(
                    f"[Objective] Inner Fold {inner_fold_idx}: Failed to load from best_model_state_dict: {e}. Trying path if available."
                )

        if (
            not loaded_best_weights
            and checkpoint_callback.best_model_path
            and os.path.exists(checkpoint_callback.best_model_path)
        ):
            try:
                checkpoint = torch.load(checkpoint_callback.best_model_path)
                if "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                    logger.info(
                        f"[Objective] Inner Fold {inner_fold_idx}: Loaded best model weights from path {checkpoint_callback.best_model_path} (extracted 'state_dict') for testing."
                    )
                    loaded_best_weights = True
                else:
                    model.load_state_dict(checkpoint)
                    logger.info(
                        f"[Objective] Inner Fold {inner_fold_idx}: Loaded best model weights from path {checkpoint_callback.best_model_path} (assumed raw state_dict) for testing."
                    )
                    loaded_best_weights = True
            except Exception as e:
                logger.warning(
                    f"[Objective] Inner Fold {inner_fold_idx}: Failed to load from best_model_path {checkpoint_callback.best_model_path}: {e}. Testing with last model state."
                )

        if not loaded_best_weights:
            logger.warning(
                f"[Objective] Inner Fold {inner_fold_idx}: Could not load best model weights from ModelCheckpoint. Testing with the model's current (last) state."
            )

        epoch_of_best_or_stop = trainer.current_epoch
        if early_stopping_callback.stopped_epoch > 0:
            epoch_of_best_or_stop = early_stopping_callback.stopped_epoch
        all_inner_best_epochs_approx.append(epoch_of_best_or_stop)

        logger.info(
            f"[Objective] Inner Fold {inner_fold_idx}: Starting trainer.test() with {'best' if loaded_best_weights else 'last'} model weights. Approx. best epoch: {epoch_of_best_or_stop}"
        )

        test_results_list = trainer.test(
            model, dataloaders=val_dataloader, verbose=False
        )
        if test_results_list:
            test_metrics_dict = test_results_list[0]
            current_test_auroc = test_metrics_dict.get("test_auroc", 0.0)
        else:
            logger.warning(
                f"[Objective] Inner Fold {inner_fold_idx}: trainer.test() returned empty results."
            )
            current_test_auroc = 0.0

        inner_scores.append(current_test_auroc)
        logger.info(
            f"[Objective] Inner Fold {inner_fold_idx}: test_auroc: {current_test_auroc}"
        )

    trial.set_user_attr("inner_fold_test_aurocs", inner_scores)
    trial.set_user_attr("inner_fold_best_epochs_approx", all_inner_best_epochs_approx)

    final_score = np.mean(inner_scores) if inner_scores else 0.0
    logger.info(
        f"[Objective] Trial {trial.number if trial else 'N/A'} finished. Average test_auroc over inner folds: {final_score}"
    )
    return final_score
