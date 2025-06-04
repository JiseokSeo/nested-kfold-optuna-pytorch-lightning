import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import warnings

# "num_workers"와 "71"을 포함하는 UserWarning을 무시합니다.
# 정규 표현식을 사용하여 좀 더 유연하게 매칭할 수 있습니다.
warnings.filterwarnings("ignore", category=UserWarning, message=r".*num_workers.*71.*")

from src.utils import collect_expriments_config, save_json, load_json
from scripts.experiments import run_experiments

# from scripts.test_experiments import run_test, save_result


def search_best_params(
    reproducible: bool = False, debug: bool = False
):  # reproducible 인자 추가
    experiments_config = collect_expriments_config(os.path.join("configs"))

    all_exp_params = {}
    for experiment_config in experiments_config:  # 모든 실험 반복
        exp_params = run_experiments(
            experiment_config,
            reproducible=reproducible,
            debug=debug,  # reproducible 인자 전달
        )
        all_exp_params[experiment_config["experiment_name"]] = exp_params
    save_json(all_exp_params, os.path.join("result", "all_exp_params.json"))


def test_best_params(reproducible: bool = True):
    # # config 로드
    # experiments_config = collect_expriments_config("configs/")
    # # 찾은 최적 파라미터 로드
    # all_exp_params = load_json("result/all_exp_params.json")
    # # config와 최적 파라미터 merge하여 exp당 최종 파라미터 도출

    # all_results = {}
    # for experiment_config in experiments_config:
    #     experiment_name = experiment_config["experiment_name"]
    #     best_params = all_exp_params[experiment_name]
    #     results = run_test(experiment_config, best_params, reproducible=reproducible)
    #     all_results[experiment_name] = results

    # # 외부 폴더 테스트 데이터로 테스트 수행
    # # 결과 저장
    # save_result(all_results, "result/all_results.json")
    pass


def main(need_search=True, need_test=False):
    if need_search:
        search_best_params(debug=False)
    if need_test:
        # test_best_params 호출 시에는 항상 재현성을 True로 하거나, 별도 인자로 관리합니다.
        # 여기서는 일단 test_best_params 내부에서 True로 고정한다고 가정합니다.
        test_best_params()


if __name__ == "__main__":
    main(need_search=True, need_test=False)
