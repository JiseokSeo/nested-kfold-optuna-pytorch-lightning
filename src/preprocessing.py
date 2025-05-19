# src/data/preprocessing.py (수정 완료)
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import joblib
import os
import logging

logger = logging.getLogger(__name__)


def get_scaler(scaler_type: str = "RobustScaler"):
    """Creates a scaler instance."""
    scaler_type_lower = scaler_type.lower()
    logger.debug(f"Creating scaler instance of type: {scaler_type_lower}")
    if scaler_type_lower == "robustscaler":
        return RobustScaler()
    elif scaler_type_lower == "standardscaler":
        return StandardScaler()
    elif scaler_type_lower == "minmaxscaler":
        return MinMaxScaler()
    else:
        logger.warning(f"Unsupported scaler type '{scaler_type}'. Using RobustScaler.")
        return RobustScaler()


def save_scaler_with_columns(scaler_info: dict, filepath: str):
    """Saves the dictionary containing the scaler object and column list."""
    if (
        not isinstance(scaler_info, dict)
        or "scaler" not in scaler_info
        or "columns" not in scaler_info
    ):
        logger.error(
            f"Invalid scaler_info format. Expected dict with 'scaler' and 'columns'. Got: {type(scaler_info)}"
        )
        return

    try:
        # Ensure directory exists
        dir_path = os.path.dirname(filepath)
        if (
            dir_path
        ):  # Check if dirname is not empty (e.g., for relative paths in current dir)
            os.makedirs(dir_path, exist_ok=True)
        joblib.dump(scaler_info, filepath)
        logger.info(f"Scaler and columns saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving scaler info: {e}", exc_info=True)


def load_scaler_with_columns(filepath: str) -> dict | None:
    """Loads the dictionary containing the scaler object and column list."""
    if not os.path.exists(filepath):
        logger.error(f"Error: Scaler info file not found at {filepath}")
        return None
    try:
        scaler_info = joblib.load(filepath)
        if (
            not isinstance(scaler_info, dict)
            or "scaler" not in scaler_info
            or "columns" not in scaler_info
        ):
            logger.error(
                f"Loaded object from {filepath} is not a valid scaler_info dictionary."
            )
            return None
        logger.info(f"Scaler and columns loaded from {filepath}")
        return scaler_info
    except FileNotFoundError:  # joblib.load() can also raise FileNotFoundError
        logger.error(f"Error: Scaler info file not found at {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error loading scaler info: {e}", exc_info=True)
        return None
