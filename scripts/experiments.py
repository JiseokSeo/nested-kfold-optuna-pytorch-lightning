from src.utils import (
    merge_config,
    load_data,
    get_study_module_from_config,
    save_json,
    get_logger,
)
from configs.search_range import get_trial_params
from sklearn.model_selection import StratifiedKFold
import numpy as np
import optuna
import json
import os
from src.objective import objective
import lightning as L
import logging

# 아래 빌더 함수들은 실제 구현 위치에 맞게 import
from src.model_modules.builder import build_model_from_config
from src.trainer.builder import build_trainer_from_config
from src.data_modules.builder import build_data


def run_experiments(experiment_config, reproducible: bool = False, debug: bool = False):
    """
    하나의 실험(실험 config)에 대해 outer k-fold를 돌며,
    각 fold별로 Optuna 탐색 결과(result_report)를 수집하고 파일로 저장.
    """
    experiment_name = experiment_config["experiment_name"]
    log_dir = f"result/{experiment_name}/logs/log"
    log_path = os.path.join(log_dir, "experiment_run.log")
    logger = get_logger(log_path)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info(f"실험 {experiment_name}의 로그를 {log_path}에 저장합니다.")

    SEED = experiment_config["seed"]
    L.seed_everything(SEED, workers=reproducible)

    outer_kf = StratifiedKFold(
        n_splits=experiment_config["trainer"]["n_outer_folds"],
        shuffle=True,
        random_state=SEED,
    )
    df = load_data(
        experiment_config,
        experiment_config["data"]["csv_path"],
        experiment_config["data"].get("use_columns"),
    )
    y = df[experiment_config["data"]["target_column"]]
    experiment_result = {}

    # config에서 skip_fold 리스트 가져오기 (없으면 빈 리스트)
    folds_to_skip = experiment_config.get("optuna", {}).get("skip_fold", [])
    if folds_to_skip:
        print(f"Skipping outer folds: {folds_to_skip}")

    for outer_fold_idx, (train_idx, test_idx) in enumerate(outer_kf.split(df, y)):
        # 현재 outer_fold_idx가 folds_to_skip 리스트에 있는지 확인
        if outer_fold_idx in folds_to_skip:
            print(f"Skipping outer fold {outer_fold_idx} as per config.")
            experiment_result[str(outer_fold_idx)] = {
                "status": "skipped",
                "reason": "Defined in optuna.skip_fold",
            }
            continue  # 다음 outer fold로 넘어감

        print(
            f"outer fold {outer_fold_idx} train class dist: {np.bincount(y.iloc[train_idx].values.astype(int))}"
        )
        outer_train = df.iloc[train_idx]
        # outer_test = df.iloc[test_idx]  # 필요시 사용 (nested k-fold에서 외부 test set)

        # inner loop + optuna 탐색 및 fold별 파일 저장
        result_report = run_inner_optuna_search(
            outer_train,
            experiment_config,
            outer_fold_idx,
            reproducible=reproducible,
            debug=debug,
        )
        result_report_without_trials = {
            k: v for k, v in result_report.items() if k != "trials"
        }
        experiment_result[str(outer_fold_idx)] = result_report_without_trials

    save_json(
        experiment_result,
        f"{experiment_config['dir']['result_save_path']}/{experiment_config['experiment_name']}_best_params.json",
    )
    return experiment_result


def run_inner_optuna_search(
    train_data, config, outer_fold_idx, reproducible: bool = False, debug: bool = False
):
    """
    outer fold의 train_data에 대해 Optuna로 inner k-fold 탐색을 수행,
    best_params와 trial별 metric을 포함한 result_report를 파일로 저장하고 반환.
    """
    n_inner_folds = config["trainer"]["n_inner_folds"]
    y = train_data[config["data"]["target_column"]]

    storage_path = config["optuna"].get("storage_path")
    if storage_path:
        project_root = config["dir"].get("project_root", "")
        if project_root and not storage_path.startswith("sqlite:///"):
            abs_path = os.path.join(project_root, storage_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            storage_path = f"sqlite:///{abs_path}"
    else:
        storage_path = None

    study_name = f"{config['experiment_name']}_{outer_fold_idx}"

    # Pruner 설정
    pruner_config = config["optuna"].get("pruner")
    pruner = None
    if pruner_config == "MedianPruner":  # 예시: 문자열로 Pruner 지정
        pruner = optuna.pruners.MedianPruner()
    elif pruner_config == "SuccessiveHalvingPruner":
        pruner = optuna.pruners.SuccessiveHalvingPruner()
    # 다른 Pruner 종류들도 필요에 따라 추가 가능
    elif (
        pruner_config is not None
    ):  # 다른 문자열이나 객체가 올 경우 경고 또는 에러 처리
        print(
            f"Warning: Unknown pruner type '{pruner_config}' specified. No pruner will be used."
        )

    def optuna_objective(trial):
        return objective(
            trial=trial,
            config=config,
            train_data=train_data,
            y=y,
            n_inner_folds=n_inner_folds,
            metric_key="val_auroc",
            reproducible=reproducible,
            debug=debug,
        )

    study = optuna.create_study(
        direction=config["optuna"]["direction"],
        storage=storage_path,
        study_name=study_name,
        pruner=pruner,  # pruner 인자 추가
        load_if_exists=True,
    )

    # Timeout 설정
    timeout_setting = config["optuna"].get("timeout")  # None일 수 있음

    study.optimize(
        optuna_objective,
        n_trials=config["optuna"]["n_trials_inner"],
        n_jobs=config["optuna"].get("n_jobs", 1),
        timeout=timeout_setting,  # timeout 인자 추가
    )
    result_report = make_result_report(
        study, config["experiment_name"], str(outer_fold_idx)
    )

    parameter_save_path = config["dir"]["parameter_save_path"]
    save_json(result_report, f"{parameter_save_path}/{study_name}.json")

    return result_report


def make_result_report(study, exp_name, outer_folder_num):
    trials_report = []
    for trial in study.trials:
        trial_report = {
            "trial_number": trial.number,
            "inner_loop_metric": trial.user_attrs.get("inner_scores", []),
            "mean_inner_loop_metric": trial.value,
            "params": trial.params,
            "best_epoch": trial.user_attrs.get("best_epoch", -1),
        }
        trials_report.append(trial_report)

    # 스터디의 best_trial에 대한 best_epoch도 추가
    best_trial_epoch = -1
    if study.best_trial:
        best_trial_epoch = study.best_trial.user_attrs.get("best_epoch", -1)

    result_report = {
        "exp_name": exp_name,
        "outer_folder_num": outer_folder_num,
        "best_params": study.best_params,
        "best_trial_value": study.best_value,
        "best_trial_epoch": best_trial_epoch,
        "trials": trials_report,
    }
    return result_report


def make_all_exp_best_params(all_exp_params):
    """
    여러 실험의 fold별 result_report를 받아 all_exp_best_params.json 포맷으로 변환
    (실험명-폴더번호별 best_params만 포함)
    """
    all_exp_best_params = {}
    for exp_name, folds in all_exp_params.items():
        for fold_num, report in folds.items():
            key = f"{exp_name}_{fold_num}"
            all_exp_best_params[key] = {
                "exp_name": report["exp_name"],
                "outer_folder_num": report["outer_folder_num"],
                "best_params": report["best_params"],
            }
    return all_exp_best_params
