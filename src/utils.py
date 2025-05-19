from typing import List, Optional
from copy import deepcopy
import os
import json
from glob import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib
import logging

# 로거 설정 (utils.py용)
logger = logging.getLogger(__name__)


def generate_layer_dims(
    input_dim,
    n_layers,
    n_units,
    output_dim,
    structure,
) -> List[int]:
    """
    Generates the output dimensions for each layer in the MLP.

    Args:
        input_dim: Dimension of the input features.
        n_layers: The total number of layers (Linear + Act + Norm + Dropout blocks).
                  Must be >= 1. If 1, only the output layer is created.
        n_units: The target number of units for hidden layers (behavior depends on structure).
                 Must be specified if n_layers > 1.
        output_dim: The final output dimension of the MLP. Must be positive.
        structure: The structure type ('constant', 'pyramid', 'funnel', 'hourglass').

    Returns:
        A list containing the output dimension of each layer. The length of the list
        will be n_layers.
    """
    print(input_dim, n_layers, n_units, output_dim, structure)
    input_dim = max(1, input_dim)
    output_dim = max(1, output_dim)  # Ensure output_dim is positive

    if n_layers is None or n_layers < 1:
        # Raise error or default? Let's require n_layers >= 1
        raise ValueError("n_layers must be specified and be at least 1.")

    if n_layers == 1:
        # Only one layer directly maps input to output
        return [output_dim]

    # If n_layers > 1, n_units must be provided
    if n_units is None or n_units < 1:
        raise ValueError("n_units must be specified and positive when n_layers > 1.")
    n_units = max(1, n_units)  # Ensure n_units is positive

    # hidden_layer_count is the number of dimensions we need to generate in the list
    hidden_layer_count = n_layers

    dims = []
    current_dim = input_dim

    if structure == "constant":
        # All hidden layers have n_units, last layer has output_dim
        dims = [n_units] * (hidden_layer_count - 1) + [output_dim]

    elif structure == "pyramid":
        # Linearly increase from input_dim towards n_units (or beyond if needed to reach output_dim)
        # We need n_layers-1 intermediate points between input_dim and output_dim
        # Let's aim for n_units at the peak (around middle layer)
        peak_units = max(input_dim, output_dim, n_units)
        # Simple linear interpolation for demonstration
        for i in range(hidden_layer_count - 1):
            # Interpolate between input_dim and peak_units for first half
            # Interpolate between peak_units and output_dim for second half (approx)
            target_dim = (
                int(
                    input_dim
                    + (peak_units - input_dim) * (i + 1) / (hidden_layer_count - 1)
                )
                if i < hidden_layer_count / 2
                else int(
                    peak_units
                    + (output_dim - peak_units)
                    * (i + 1 - (hidden_layer_count / 2))
                    / (hidden_layer_count / 2)
                )
            )

            # This simple linear interp might not be ideal, just an example.
            # A more common pyramid increases then decreases. Let's refine.
            # Increase towards n_units, then decrease towards output_dim
            increase_steps = hidden_layer_count // 2
            decrease_steps = hidden_layer_count - increase_steps

            if i < increase_steps:
                step = (n_units - input_dim) / max(1, increase_steps)
                current_dim = int(input_dim + (i + 1) * step)
            else:  # Decrease phase (including middle if odd number of layers)
                start_decrease_dim = (
                    dims[-1] if dims else n_units
                )  # Start from last dim or n_units
                steps_into_decrease = i - increase_steps + 1
                step = (output_dim - start_decrease_dim) / max(1, decrease_steps)
                current_dim = int(start_decrease_dim + steps_into_decrease * step)

            dims.append(max(1, current_dim))  # Ensure positive
        dims[-1] = output_dim  # Force last dim

    elif structure == "funnel":
        # Linearly decrease from max(input_dim, n_units) towards output_dim
        start_dim = max(input_dim, n_units)
        step = (output_dim - start_dim) / max(
            1, hidden_layer_count
        )  # Step can be negative
        for i in range(hidden_layer_count - 1):
            dims.append(max(1, int(start_dim + (i + 1) * step)))
        dims.append(output_dim)  # Add final output dim

    else:  # Default: hourglass
        first_third_count = max(1, hidden_layer_count // 3)
        last_third_count = max(1, hidden_layer_count // 3)
        middle_count = max(0, hidden_layer_count - first_third_count - last_third_count)

        peak_units = max(input_dim, output_dim, n_units)

        # Expansion phase
        current_dim = float(input_dim)
        if first_third_count > 0:
            exp_factor = (
                (peak_units / current_dim) ** (1 / first_third_count)
                if current_dim > 0
                else 1
            )
            for _ in range(first_third_count):
                current_dim *= exp_factor
                dims.append(max(1, int(current_dim)))

        # Middle phase
        dims.extend([max(1, peak_units)] * middle_count)

        # Reduction phase
        current_dim = float(
            dims[-1] if dims else peak_units
        )  # Start reduction from last generated dim
        if last_third_count > 0:
            red_factor = (
                (output_dim / current_dim) ** (1 / last_third_count)
                if current_dim > 0
                else 1
            )
            for _ in range(
                last_third_count - 1
            ):  # Generate intermediate reduction points
                current_dim *= red_factor
                dims.append(max(1, int(current_dim)))
            dims.append(output_dim)  # Ensure final layer is exactly output_dim

    # Ensure final list length is correct and last element is output_dim
    if len(dims) != hidden_layer_count:
        raise ValueError(
            f"Warning: Generated dims length mismatch ({len(dims)} vs {hidden_layer_count}). Adjusting."
        )

    # Ensure all dims are positive integers
    final_dims = [max(1, int(d)) for d in dims]

    return final_dims


def merge_config(config, best_params):
    """
    config(dict)와 best_params(dict)를 받아, best_params의 중첩 키를 config에 반영한 새 config를 반환
    예시: best_params = {'model.classifier.params.n_layers': 3, ...}
    """
    config = deepcopy(config)
    for k, v in best_params.items():
        keys = k.split(".")
        d = config
        for key in keys[:-1]:
            if key not in d or not isinstance(d[key], dict):
                d[key] = {}
            d = d[key]
        d[keys[-1]] = v
    return config


def collect_expriments():
    """
    configs/{exp}.json 파일을 읽고, exp_params_sample.json 형태의 실험 목록을 반환
    """
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    config_files = [
        f
        for f in glob(os.path.join(config_dir, "*.json"))
        if not f.endswith("sample.json")
    ]
    experiments = []
    for config_path in config_files:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        exp_name = config.get(
            "experiment_name", os.path.splitext(os.path.basename(config_path))[0]
        )
        n_outer_folds = config.get("trainer", {}).get("n_outer_folds", 1)
        for outer_folder_num in range(n_outer_folds):
            exp_dict = {
                "filename": f"{exp_name}_{outer_folder_num}.json",
                "exp_name": exp_name,
                "outer_folder_num": outer_folder_num,
                "best_params": {},  # 실제 실험에서는 best_params를 채워야 함
                "trials": [],  # 실제 실험에서는 trials를 채워야 함
            }
            experiments.append(exp_dict)
    return experiments


def collect_expriments_config(config_dir: str):
    """
    config_dir 하위의 *.json 파일을 모두 읽어 config 딕셔너리 리스트로 반환
    """
    config_files = [
        f
        for f in glob(os.path.join(config_dir, "*.json"))
        if not f.endswith("sample.json")
    ]
    experiments = []
    for config_path in config_files:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        experiments.append(config)
    return experiments


def collect_search_range(config_dir: str):
    """
    config_dir 하위의 search_range.py 모듈에서 get_trial_params 함수를 반환
    """
    import importlib.util
    import sys
    import os

    search_range_path = os.path.join(config_dir, "search_range.py")
    module_name = "search_range"
    spec = importlib.util.spec_from_file_location(module_name, search_range_path)
    search_range_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = search_range_module
    spec.loader.exec_module(search_range_module)
    return search_range_module.get_trial_params


def save_json(data, path):
    """
    주어진 데이터를 JSON 형식으로 지정된 경로에 저장합니다.
    필요한 상위 디렉토리가 없으면 자동으로 생성합니다.

    Args:
        data: JSON으로 저장할 데이터 (dict, list 등).
        path: JSON 파일을 저장할 경로.
    """
    # 파일의 디렉토리 경로를 추출합니다.
    directory = os.path.dirname(path)

    # 디렉토리가 존재하지 않으면 생성합니다. exist_ok=True는 이미 존재해도 에러가 나지 않게 합니다.
    if directory:  # 경로에 디렉토리 부분이 있는 경우에만 실행
        os.makedirs(directory, exist_ok=True)

    # 파일을 열고 JSON 데이터를 저장합니다.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_study_module_from_config(config):
    model = something()
    trainer = some()
    return model, trainer


def load_data(config: dict, csv_path: str, use_columns: list = None) -> pd.DataFrame:
    data_config = config.get("data", {})
    id_col = data_config.get("id_column")
    target_col = data_config.get("target_column")

    if not id_col or not target_col:
        # 이 경우는 config 자체의 문제이므로, 여기서 처리하기보다 config 유효성 검사에서 처리하는 것이 나을 수 있음
        # 하지만 load_data 호출 시점에 config가 전달되므로, 여기서 방어적으로 체크 가능
        raise ValueError(
            "Configuration error: 'data.id_column' and 'data.target_column' must be specified in config."
        )

    if not csv_path or not os.path.exists(csv_path):
        logger.error(f"Critical: CSV file not found at {csv_path}. This is required.")
        raise FileNotFoundError(f"Required CSV file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding="utf-8", dtype={"id": str})
    except Exception as e:
        logger.error(f"Critical: Error loading data from {csv_path}: {e}")
        raise RuntimeError(f"Failed to load CSV data from {csv_path}") from e

    # 필수 컬럼 (id, target) 존재 확인
    missing_essential_cols = []
    if id_col not in df.columns:
        missing_essential_cols.append(id_col)
    if target_col not in df.columns:
        missing_essential_cols.append(target_col)

    if missing_essential_cols:
        msg = f"Critical: Essential columns {missing_essential_cols} not found in {csv_path}. These are required based on config."
        logger.error(msg)
        raise ValueError(msg)

    if use_columns:
        # use_columns에는 id_col과 target_col이 포함되어 있다고 가정하거나, 여기서 추가 검증 필요
        # 여기서는 use_columns가 사용자가 명시적으로 선택한 컬럼들이라고 가정
        valid_columns = [col for col in use_columns if col in df.columns]
        missing_specified_cols = [col for col in use_columns if col not in df.columns]

        if missing_specified_cols:
            logger.warning(
                f"Specified columns {missing_specified_cols} in 'use_columns' not found in {csv_path}. They will be ignored."
            )

        if not valid_columns:
            # use_columns가 지정되었는데 유효한게 하나도 없으면 문제. 하지만 id, target은 이미 위에서 검증.
            # use_columns에 id, target 외 다른 것을 지정했는데 그것들이 다 없다면? -> 경고 후 id, target만 사용 or 오류
            # 현재는 유효한 것만 선택하므로, 최소 id, target은 포함됨 (위의 필수 컬럼 체크로 인해)
            # 만약 use_columns가 id, target을 포함하지 않게 잘못 설정될 수도 있다면 추가 방어 필요
            logger.warning(
                f"None of the specified 'use_columns' are valid (or 'use_columns' is empty). "
                f"Proceeding with all columns, but this might indicate a config issue."
            )
            # 이 경우, 모든 컬럼을 사용하거나, 아니면 id_col, target_col과 feature_columns만 사용하는 등 정책 필요
            # 여기서는 일단 모든 컬럼 사용하도록 두되, use_columns의 의도와 다를 수 있음.
            # 하지만 use_columns의 주 목적이 id, target 외 추가 컬럼 필터링이라면,
            # valid_columns가 비었을 때 (id, target 외에는 없는 경우) 아래 df[valid_columns]가 어떻게 동작할지 고려.
            # 아래 로직은 valid_columns가 비면 모든 컬럼을 쓰게 될 것.
            # 차라리 id_col, target_col은 항상 포함하고, use_columns는 추가적인 것들로 간주하는 것이 나을 수 있음.
            # 여기서는 use_columns의 의도를 '지정된 것만 사용'으로 보고, valid_columns가 비면 df를 그대로 반환.
            # (이전에는 df[valid_columns]를 바로 했음)
            if (
                not valid_columns
            ):  # valid_columns가 정말 비었다면 (use_columns에 지정한 것들이 하나도 없다면)
                logger.warning(
                    f"No valid columns found from use_columns in {csv_path}. Returning DataFrame with all columns. Review 'use_columns' in config."
                )
                # 또는, 여기서도 id_col, target_col만 선택하는 것이 더 안전할 수 있음
                # return df[[id_col, target_col]] # 이렇게 하면 feature_columns가 있어도 무시됨
                return df  # 일단 전체 반환

        # id, target 컬럼은 항상 포함되도록 보장 (use_columns에 없더라도)
        final_columns_to_use = set(valid_columns)
        final_columns_to_use.add(id_col)
        final_columns_to_use.add(target_col)
        return df[list(final_columns_to_use)]

    # use_columns가 None이면 모든 컬럼 반환 (단, id, target 존재는 위에서 확인됨)
    return df


def get_fitted_scaler(config, df: pd.DataFrame = None):
    scaler_type = config.get("data", {}).get("scaler", "StandardScaler")
    scaler = None
    if scaler_type == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_type == "RobustScaler":
        scaler = RobustScaler()
    else:
        logger.warning(
            f"Unsupported scaler type: {scaler_type}. No scaler will be used."
        )
        return None

    if df is not None and config.get("data", {}).get("use_csv", False):
        columns_to_scale = config.get("data", {}).get("columns_to_scale", [])
        if columns_to_scale:
            # Check if all columns_to_scale exist in df
            missing_cols = [col for col in columns_to_scale if col not in df.columns]
            if missing_cols:
                logger.error(
                    f"Error fitting scaler: Columns {missing_cols} not found in DataFrame. Scaler will not be fitted."
                )
                return None  # 존재하지 않는 컬럼이 있으면 None 반환

            if not df[columns_to_scale].empty:  # 데이터가 비어있지 않은 경우에만 fit
                try:
                    scaler.fit(df[columns_to_scale])
                    logger.info(
                        f"Scaler {scaler_type} fitted on columns: {columns_to_scale}"
                    )
                except Exception as e:
                    logger.error(
                        f"An unexpected error occurred during scaler fitting: {e}"
                    )
                    return None
            else:
                logger.warning(
                    f"Data for scaling is empty for columns: {columns_to_scale}. Scaler will not be fitted."
                )
                return None  # 빈 데이터프레임으로 fit 시도 방지
        else:
            logger.warning(
                "columns_to_scale is empty in config. Scaler will not be fitted."
            )
            return None  # 스케일링할 컬럼이 없으면 None 반환 (또는 scaler 객체 그대로 반환 후 do_scaling에서 처리)
    elif not config.get("data", {}).get("use_csv", False):
        logger.info("use_csv is False. Scaler will not be fitted.")
        return None  # use_csv가 False면 스케일러를 fit하지 않고 None 반환
    elif df is None:
        logger.warning("DataFrame is None. Scaler will not be fitted.")
        return None  # DataFrame이 None이면 스케일러를 fit하지 않고 None 반환

    return scaler


def save_scaler(scaler, path):
    """
    sklearn scaler 객체를 지정한 경로에 저장
    """
    joblib.dump(scaler, path)


def do_scaling(columns_to_scale, df: pd.DataFrame, scaler):
    if scaler and columns_to_scale and isinstance(df, pd.DataFrame):
        # Check if all columns_to_scale exist in df
        missing_cols = [col for col in columns_to_scale if col not in df.columns]
        if missing_cols:
            logger.error(
                f"Error during scaling: Columns {missing_cols} not found in DataFrame. Returning original DataFrame."
            )
            return df

        df_scaled = df.copy()
        try:
            df_scaled[columns_to_scale] = scaler.transform(df[columns_to_scale])
            logger.info(f"Data scaled for columns: {columns_to_scale}")
        except ValueError as e:
            logger.error(
                f"ValueError during scaling transformation (e.g., scaler not fitted or feature mismatch): {e}. Returning original DataFrame."
            )
            return df
        except KeyError as e:  # 이미 위에서 체크했지만, 방어적으로 추가
            logger.error(
                f"KeyError during scaling transformation: {e}. Returning original DataFrame."
            )
            return df
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during scaling transformation: {e}. Returning original DataFrame."
            )
            return df
        return df_scaled
    elif not scaler:
        logger.info("Scaler is None. Scaling will not be applied.")
    elif not columns_to_scale:
        logger.info("columns_to_scale is empty. Scaling will not be applied.")
    elif not isinstance(df, pd.DataFrame):
        logger.error("Input df is not a pandas DataFrame. Scaling will not be applied.")

    return df  # scaler가 None이거나, columns_to_scale이 없거나, df가 DataFrame이 아니면 원본 반환


def save_scaler(scaler, path):
    """
    sklearn scaler 객체를 지정한 경로에 저장
    """
    joblib.dump(scaler, path)


def get_logger(log_path=None, experiment_name="experiment"):
    """
    log_path가 주어지면 해당 경로에 로그를 저장하는 logger를 반환.
    log_path가 None이면 기본 logger를 반환.
    """
    import sys
    import logging

    logging.basicConfig(
        level=logging.INFO,  # 기본 로그 레벨
        format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(f"{experiment_name}.log"),  # 로그 파일 이름 지정
            logging.StreamHandler(sys.stdout),  # 콘솔 출력 핸들러
        ],
    )

    # 기존 핸들러 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handlers.insert(0, logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger(__name__)
