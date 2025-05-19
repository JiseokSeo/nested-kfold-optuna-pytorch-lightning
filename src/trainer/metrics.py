# src/train/metrics.py
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score,
)


# --- 유틸리티 함수 (기존과 동일) ---
def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Numpy를 사용한 시그모이드 함수"""
    x = np.clip(x, -500, 500)  # exp 계산 시 극단적인 값 제한
    return 1 / (1 + np.exp(-x))


def prediction_diversity_penalty(confusion_matrix_data) -> float:
    """
    예측 다양성 패널티: 모델이 한 클래스만 예측하는 경우 큰 패널티를 부여.
    1에 가까울수록 다양한 예측, 0에 가까울수록 한 클래스만 예측.
    """
    if isinstance(confusion_matrix_data, list):
        if len(confusion_matrix_data) == 2 and len(confusion_matrix_data[0]) == 2:
            tn, fp = confusion_matrix_data[0]
            fn, tp = confusion_matrix_data[1]
        else:
            print(
                "Warning: Invalid format for confusion_matrix_data list. Returning PDP 0.0"
            )
            return 0.0
    elif isinstance(
        confusion_matrix_data, np.ndarray
    ) and confusion_matrix_data.shape == (2, 2):
        tn, fp, fn, tp = confusion_matrix_data.ravel()
    else:
        print(
            f"Warning: Invalid type or shape for confusion_matrix_data "
            f"({type(confusion_matrix_data)}, shape={getattr(confusion_matrix_data, 'shape', 'N/A')}). "
            f"Returning PDP 0.0"
        )
        return 0.0

    total = tn + fp + fn + tp
    if total == 0:
        return 0.0

    pred_neg_count = tn + fn
    pred_pos_count = fp + tp

    if total == 0:  # 모든 예측이 0인 경우 방지 (위에서 이미 처리하지만, 명시적으로)
        pred_neg_ratio = 0.0
        pred_pos_ratio = 0.0
    else:
        pred_neg_ratio = pred_neg_count / total
        pred_pos_ratio = pred_pos_count / total

    if pred_neg_ratio > 0 and pred_pos_ratio > 0:
        return 2 * min(pred_neg_ratio, pred_pos_ratio)
    else:
        return 0.0


# --- 입력 데이터 전처리 및 유효성 검사 함수 ---
def _validate_and_preprocess_inputs(
    y_pred_logits: np.ndarray, y_true: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool, dict | None]:
    """
    입력 데이터의 유효성을 검사하고 전처리합니다.
    Returns:
        (processed_logits, processed_true, is_valid, default_metrics_if_invalid)
        is_valid가 False일 경우, default_metrics_if_invalid에 기본 지표 값을 담아 반환합니다.
    """
    if not isinstance(y_pred_logits, np.ndarray) or not isinstance(y_true, np.ndarray):
        print("Error: Inputs must be numpy arrays. Returning default metrics.")
        # 이전 코드에서는 TypeError를 발생시켰으나, 여기서는 기본 메트릭 반환 로직을 따름
        return y_pred_logits, y_true, False, _get_default_metrics_dict()

    if y_pred_logits.ndim == 0 or y_true.ndim == 0:
        print(
            "Error: Inputs must have at least one dimension. Returning default metrics."
        )
        return y_pred_logits, y_true, False, _get_default_metrics_dict()

    y_true_flat = y_true.flatten().astype(int)
    y_pred_logits_flat = y_pred_logits.flatten()

    if y_true_flat.shape != y_pred_logits_flat.shape:
        print(
            f"Error: Shape mismatch: y_true {y_true_flat.shape} vs y_pred_logits {y_pred_logits_flat.shape}. Returning default metrics."
        )
        return y_pred_logits_flat, y_true_flat, False, _get_default_metrics_dict()

    valid_mask = ~np.isnan(y_pred_logits_flat) & ~np.isnan(y_true_flat)
    if not np.all(valid_mask):
        nan_count = np.sum(~valid_mask)
        print(
            f"Warning: Found {nan_count} NaN(s) in predictions or targets. Removing affected samples."
        )
        if nan_count == len(y_true_flat):
            print("Warning: All samples have NaN values. Returning default metrics.")
            return y_pred_logits_flat, y_true_flat, False, _get_default_metrics_dict()

        y_pred_logits_flat = y_pred_logits_flat[valid_mask]
        y_true_flat = y_true_flat[valid_mask]

    unique_labels = np.unique(y_true_flat)
    if len(y_true_flat) == 0 or len(unique_labels) < 2:
        warning_msg = (
            "Not enough data"
            if len(y_true_flat) == 0
            else f"Only one class ({unique_labels[0]}) present"
        )
        print(
            f"Warning: {warning_msg} ({len(y_true_flat)} valid samples). Cannot calculate some metrics. Returning default metrics."
        )
        # 단일 클래스라도 CM은 계산 가능하므로, 이 부분은 주 메트릭 계산 함수에서 처리하도록 is_valid는 True로 두되,
        # AUROC/AUPRC 등이 0이 되도록 하는 로직은 각 계산 함수에서 처리.
        # 여기서는 완전 계산 불가능한 심각한 경우만 False로 처리.
        # 이 프로젝트의 경우, 단일 클래스면 AUROC/AUPRC를 0으로 반환하므로, is_valid=True로 하고 개별 함수에서 처리.
        # 만약 이 상태를 'invalid'로 보고 바로 기본값을 반환하려면 is_valid=False와 default_metrics를 반환
        if len(y_true_flat) == 0:  # 데이터가 아예 없으면 계산 불가
            return y_pred_logits_flat, y_true_flat, False, _get_default_metrics_dict()

    return y_pred_logits_flat, y_true_flat, True, None


def _get_default_metrics_dict() -> dict:
    """모든 지표를 0으로 초기화한 기본 딕셔너리를 반환합니다."""
    return {
        "auroc": 0.0,
        "auprc": 0.0,
        "f1": 0.0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
        "TP": 0,
        "mcc_norm": 0.5,  # MCC는 -1~1이므로 정규화 시 0.5가 중립
        "bal_acc": 0.0,
        "pdp": 0.0,
        "custom": 0.0,
    }


# --- 개별 지표 계산 함수 ---
def _calculate_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        print(
            "Warning: Only one class present in y_true. AUROC is not defined, returning 0.0."
        )
        return 0.0
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError as e:
        print(f"Could not calculate AUROC: {e}. Setting to 0.0")
        return 0.0


def _calculate_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        print(
            "Warning: Only one class present in y_true. AUPRC is not well-defined, returning 0.0."
        )
        return 0.0
    try:
        return average_precision_score(y_true, y_prob)
    except ValueError as e:
        print(f"Could not calculate AUPRC: {e}. Setting to 0.0")
        return 0.0


def _calculate_f1(y_true: np.ndarray, y_pred_binary: np.ndarray) -> float:
    return f1_score(y_true, y_pred_binary, zero_division=0)


def _calculate_confusion_matrix_elements(
    y_true: np.ndarray, y_pred_binary: np.ndarray
) -> tuple[int, int, int, int]:
    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)  # 정수형으로 명시적 변환


def _calculate_mcc_normalized(y_true: np.ndarray, y_pred_binary: np.ndarray) -> float:
    try:
        mcc_val = matthews_corrcoef(y_true, y_pred_binary)
        return (mcc_val + 1) / 2  # -1~1 범위를 0~1 범위로 정규화
    except Exception as e:  # 모든 예외 처리 (예: 단일 클래스로 예측되는 경우 등)
        print(f"Could not calculate MCC: {e}. Setting normalized MCC to 0.5 (neutral).")
        return 0.5  # 오류 발생 시 중간값


def _calculate_balanced_accuracy(
    y_true: np.ndarray, y_pred_binary: np.ndarray
) -> float:
    try:
        # sklearn 0.24.1 이상에서는 `adjusted=True` 옵션으로 imbalance-adjusted balanced accuracy 사용 가능
        return balanced_accuracy_score(y_true, y_pred_binary)
    except Exception as e:
        print(f"Could not calculate Balanced Accuracy: {e}. Setting to 0.0.")
        return 0.0


def _calculate_pdp_metric(y_true: np.ndarray, y_pred_binary: np.ndarray) -> float:
    """Confusion Matrix를 내부적으로 계산하여 PDP를 반환"""
    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    return prediction_diversity_penalty(cm)


# --- 커스텀 가중 점수 계산 함수 ---
def _calculate_custom_weighted_score(
    f1: float,
    auroc: float,
    auprc: float,
    pdp: float,
    mcc_norm: float,
    bal_acc: float,
    weights: dict | None = None,
) -> float:
    """주요 지표들을 사용하여 가중 합산된 커스텀 점수를 계산합니다."""
    if weights is None:  # 기본 가중치
        weights = {
            "f1": 0.30,
            "auroc": 0.60,
            "auprc": 0.00,
            "pdp": 0.00,
            "mcc": 0.00,
            "bal_acc": 0.10,
        }

    score = (
        weights.get("f1", 0) * f1
        + weights.get("auroc", 0) * auroc
        + weights.get("auprc", 0) * auprc
        + weights.get("pdp", 0) * pdp
        + weights.get("mcc", 0) * mcc_norm
        + weights.get("bal_acc", 0) * bal_acc
    )
    # PDP 패널티 로직 (선택적)
    # if pdp < 0.3:
    #     penalty_factor = pdp / 0.3
    #     score *= penalty_factor
    return float(score)


# --- 메인 함수 ---
def calculate_metrics(
    y_pred_logits: np.ndarray,
    y_true: np.ndarray,
    prefix: str = "val_",
    threshold: float = 0.5,
    custom_metric_weights: dict | None = None,  # 가중치 외부 주입 가능
) -> dict:
    """
    주어진 로짓과 실제 레이블을 사용하여 다양한 분류 지표를 계산합니다.
    각 지표 계산은 내부 헬퍼 함수로 분리되었습니다.
    """
    metrics = {}

    # 1. 입력 유효성 검사 및 전처리
    y_pred_logits, y_true, is_valid, default_metrics = _validate_and_preprocess_inputs(
        y_pred_logits, y_true
    )
    if not is_valid:
        # 기본값 딕셔너리의 각 키에 prefix를 붙여서 반환
        return {f"{prefix}{k}": v for k, v in default_metrics.items()}

    # 2. 확률 및 이진 예측 생성
    y_prob = sigmoid_np(y_pred_logits)
    y_pred_binary = (y_prob >= threshold).astype(int)

    # 3. 개별 기본 지표 계산
    metrics[f"{prefix}auroc"] = _calculate_auroc(y_true, y_prob)
    # metrics[f"{prefix}auprc"] = _calculate_auprc(y_true, y_prob)
    metrics[f"{prefix}f1"] = _calculate_f1(y_true, y_pred_binary)
    # tn, fp, fn, tp = _calculate_confusion_matrix_elements(y_true, y_pred_binary)
    # metrics[f"{prefix}TN"] = tn
    # metrics[f"{prefix}FP"] = fp
    # metrics[f"{prefix}FN"] = fn
    # metrics[f"{prefix}TP"] = tp

    # 4. 추가 지표 계산
    # metrics[f"{prefix}mcc_norm"] = _calculate_mcc_normalized(y_true, y_pred_binary)
    metrics[f"{prefix}bal_acc"] = _calculate_balanced_accuracy(y_true, y_pred_binary)
    # metrics[f"{prefix}pdp"] = _calculate_pdp_metric(y_true, y_pred_binary)  # PDP 계산

    # 5. 커스텀 가중 점수 계산
    custom_score = _calculate_custom_weighted_score(
        f1=metrics[f"{prefix}f1"],
        auroc=metrics[f"{prefix}auroc"],
        auprc=0.0,  # metrics.get(f"{prefix}auprc", 0.0),
        pdp=0.0,  # metrics.get(f"{prefix}pdp", 0.0),
        mcc_norm=0.5,  # metrics.get(f"{prefix}mcc_norm", 0.5),
        bal_acc=metrics[f"{prefix}bal_acc"],
        weights=(
            {
                "f1": 0.30,
                "auroc": 0.60,
                # "auprc": 0.00,
                # "pdp": 0.00,
                # "mcc": 0.00,
                "bal_acc": 0.10,
            }
            if custom_metric_weights is None
            else custom_metric_weights
        ),
    )
    metrics[f"{prefix}custom"] = custom_score

    return metrics
