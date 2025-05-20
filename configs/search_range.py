import optuna


def get_trial_params(trial: optuna.Trial, config) -> dict:
    """Optuna 트라이얼을 위한 모든 조정 가능한 파라미터를 제안합니다. (카테고리컬 옵션 고정)"""
    USE_IMAGE = config["data"].get("use_image")
    USE_CSV = config["data"].get("use_csv")
    trial_params = {}

    # --- 이미지 추출기 (고정) ---
    if USE_IMAGE:
        # img_model_name = trial.suggest_categorical(...) # 제거
        trial_params["model.image_extractor.name"] = "convnext_tiny"
        trial_params["model.image_extractor.params.pretrained"] = True

    # --- CSV 추출기 (카테고리컬 고정) ---
    if USE_CSV:
        trial_params["model.csv_extractor.name"] = "res"  # 고정
        trial_params["model.csv_extractor.params.n_layers"] = trial.suggest_int(
            "model.csv_extractor.n_layers", 2, 8
        )
        trial_params["model.csv_extractor.params.n_units"] = trial.suggest_int(
            "model.csv_extractor.n_units", 64, 256, step=32
        )
        trial_params["model.csv_extractor.params.structure"] = "constant"  # 고정
        trial_params["model.csv_extractor.params.dropout"] = trial.suggest_float(
            "model.csv_extractor.dropout", 0.1, 0.4
        )
        trial_params["model.csv_extractor.params.output_dim"] = trial.suggest_int(
            "model.csv_extractor.output_dim", 64, 256, step=32
        )
        trial_params["model.csv_extractor.params.activation"] = "silu"  # 고정
        trial_params["model.csv_extractor.params.use_norm"] = True  # 고정

    # --- Fusion (고정) ---
    if USE_IMAGE and USE_CSV:
        trial_params["model.fusion.name"] = "concat"  # 고정

    trial_params["model.classifier.name"] = "mlp"  # 고정
    trial_params["model.classifier.params.n_layers"] = trial.suggest_int(
        "model.classifier.n_layers", 2, 5
    )
    trial_params["model.classifier.params.n_units"] = trial.suggest_int(
        "model.classifier.n_units", 64, 256, step=32
    )
    trial_params["model.classifier.params.structure"] = "constant"  # 고정
    trial_params["model.classifier.params.dropout"] = trial.suggest_float(
        "model.classifier.dropout", 0.1, 0.4
    )
    trial_params["model.classifier.params.activation"] = "silu"  # 고정
    trial_params["model.classifier.params.use_norm"] = True  # 고정
    trial_params["model.classifier.params.output_dim"] = config.get("trainer", {}).get(
        "num_classes"
    )

    # --- 옵티마이저 (고정) ---

    trial_params["trainer.optimizer.name"] = "AdamW"  # 고정
    trial_params["trainer.optimizer.params.lr"] = trial.suggest_float(
        "trainer.optimizer.lr", 5e-6, 5e-4, log=True
    )
    trial_params["trainer.optimizer.params.weight_decay"] = trial.suggest_float(
        "trainer.optimizer.weight_decay", 1e-7, 1e-3, log=True
    )

    # --- 그래디언트 클리핑 ---
    use_gradient_clip = trial.suggest_categorical(
        "trainer.use_gradient_clip", [True, False]
    )
    trial_params["trainer.use_gradient_clip"] = use_gradient_clip
    if use_gradient_clip:
        trial_params["trainer.gradient_clip_val"] = trial.suggest_float(
            "trainer.gradient_clip_val", 1.0, 10.0
        )
    else:
        trial_params["trainer.gradient_clip_val"] = None

    # --- 스케줄러 (고정 및 핵심 파라미터 탐색) ---
    # use_scheduler = trial.suggest_categorical("trainer.use_scheduler", [True, False])
    # trial_params["trainer.use_scheduler"] = use_scheduler

    # if use_scheduler:
    #     trial_params["trainer.scheduler.name"] = "OneCycleLR"  # 고정
    #     trial_params["trainer.scheduler.params.pct_start"] = trial.suggest_float(
    #         "trainer.scheduler.pct_start", 0.1, 0.5, step=0.05
    #     )
    #     trial_params["trainer.scheduler.params.div_factor"] = trial.suggest_float(
    #         "trainer.scheduler.div_factor", 10.0, 50.0, step=5.0
    #     )
    #     trial_params["trainer.scheduler.params.final_div_factor"] = trial.suggest_float(
    #         "trainer.scheduler.final_div_factor", 1e3, 1e5, log=True
    #     )
    # else:
    #     trial_params["trainer.scheduler.name"] = None
    #     trial_params["trainer.scheduler.params"] = {}

    # --- 손실 함수 (고정) ---
    trial_params["trainer.criterion.name"] = "BCEWithLogitsLoss"
    trial_params["trainer.criterion.params"] = {}

    return trial_params
