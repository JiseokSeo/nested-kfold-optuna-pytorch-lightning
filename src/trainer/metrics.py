import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
)  # Removed unused: accuracy_score, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)

# Individual metric calculation functions with error handling


def _calculate_auroc(
    probs_np: np.ndarray, targets_np: np.ndarray, log_prefix: str
) -> float:
    if probs_np.size == 0 or targets_np.size == 0:
        logger.warning(
            f"[{log_prefix}_auroc] Probabilities or targets array is empty. Returning 0.0."
        )
        return 0.0
    try:
        if len(np.unique(targets_np)) < 2:
            logger.warning(
                f"[{log_prefix}_auroc] Only one class present in y_true. ROC AUC score is not defined. Returning 0.0."
            )
            return 0.0
        return float(roc_auc_score(targets_np, probs_np))
    except ValueError as ve:
        logger.warning(f"[{log_prefix}_auroc] ValueError: {ve}. Returning 0.0.")
        return 0.0
    except Exception as e:
        logger.error(f"[{log_prefix}_auroc] Exception: {e}. Returning 0.0.")
        return 0.0


def _calculate_f1(
    preds_np: np.ndarray, targets_np: np.ndarray, log_prefix: str
) -> float:
    if preds_np.size == 0 or targets_np.size == 0:
        logger.warning(
            f"[{log_prefix}_f1] Predictions or targets array is empty. Returning 0.0."
        )
        return 0.0
    try:
        return float(f1_score(targets_np, preds_np, zero_division=0))
    except ValueError as ve:
        logger.warning(f"[{log_prefix}_f1] ValueError: {ve}. Returning 0.0.")
        return 0.0
    except Exception as e:
        logger.error(f"[{log_prefix}_f1] Exception: {e}. Returning 0.0.")
        return 0.0


def _calculate_confusion_matrix_components(
    preds_np: np.ndarray, targets_np: np.ndarray, log_prefix: str
) -> tuple[int, int, int, int]:
    if preds_np.size == 0 or targets_np.size == 0:
        logger.warning(
            f"[{log_prefix}_cm] Predictions or targets array is empty. Returning (0,0,0,0) for (TN,FP,FN,TP)."
        )
        return 0, 0, 0, 0  # TN, FP, FN, TP
    try:
        cm = confusion_matrix(targets_np, preds_np, labels=[0, 1])
        # Ensure cm has the expected shape (2,2) before attempting to ravel
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return int(tn), int(fp), int(fn), int(tp)
        elif cm.shape == (1, 1):  # Only one class predicted and present
            # This case implies all predictions match the single true class or all are incorrect for a different single true class
            # This is tricky. It depends on what label=[0,1] does. If targets_np only has 0s, and preds_np only has 0s => cm might be [[N,0],[0,0]] or just [[N]]
            # Let's assume if not (2,2), something is off for typical binary classification reporting, or only one class was in both preds and targets.
            # For simplicity and robustness against unexpected cm shapes not equalling (2,2) due to single class in preds/targets:
            # if only 0s in targets and 0s in preds: tp=0, fn=0, fp=0, tn=len(targets_np)
            # if only 1s in targets and 1s in preds: tp=len(targets_np), fn=0, fp=0, tn=0
            # This gets complex. Sticking to the (2,2) assumption for now, and if not, log and return zeros.
            logger.warning(
                f"[{log_prefix}_cm] Confusion matrix shape is not (2,2) but {cm.shape}. This might happen if only one class is present in targets or predictions. Returning (0,0,0,0)."
            )
            return 0, 0, 0, 0
        else:  # Any other unexpected shape
            logger.warning(
                f"[{log_prefix}_cm] Unexpected confusion matrix shape: {cm.shape}. Returning (0,0,0,0)."
            )
            return 0, 0, 0, 0

    except ValueError as ve:
        logger.warning(
            f"[{log_prefix}_cm] ValueError calculating confusion matrix: {ve}. Returning (0,0,0,0)."
        )
        return 0, 0, 0, 0
    except Exception as e:
        logger.error(
            f"[{log_prefix}_cm] Exception calculating confusion matrix: {e}. Returning (0,0,0,0)."
        )
        return 0, 0, 0, 0


def _common_metrics_logic(
    logits_np: np.ndarray, targets_np: np.ndarray, prefix: str
) -> dict:
    metrics = {}
    probs_np = np.array([])
    preds_np = np.array([])

    try:
        if logits_np.ndim > 1 and logits_np.shape[1] == 1:
            logits_np_squeezed = logits_np.squeeze(axis=1)
        else:
            logits_np_squeezed = logits_np

        if logits_np_squeezed.size > 0:
            probs_np = 1 / (1 + np.exp(-logits_np_squeezed))  # Sigmoid
            preds_np = (probs_np >= 0.5).astype(int)
        else:
            logger.warning(
                f"[{prefix}common_logic] Logits array is empty. Predictions and probabilities will be empty."
            )
    except Exception as e:
        logger.error(
            f"[{prefix}common_logic] Error converting logits to predictions/probabilities: {e}"
        )

    metrics[f"{prefix}auroc"] = _calculate_auroc(probs_np, targets_np, prefix)
    metrics[f"{prefix}f1"] = _calculate_f1(preds_np, targets_np, prefix)

    # Removed: accuracy, precision, recall, specificity, custom metric

    tn, fp, fn, tp = _calculate_confusion_matrix_components(
        preds_np, targets_np, prefix
    )
    metrics[f"{prefix}TP"] = tp
    metrics[f"{prefix}TN"] = tn
    metrics[f"{prefix}FP"] = fp
    metrics[f"{prefix}FN"] = fn

    return metrics


def test_metrics(
    logits_np: np.ndarray, targets_np: np.ndarray, prefix: str = "test_"
) -> dict:
    """Calculates metrics for test set."""
    logger.debug(f"[test_metrics] Calculating test metrics with prefix '{prefix}'")
    return _common_metrics_logic(logits_np, targets_np, prefix)
