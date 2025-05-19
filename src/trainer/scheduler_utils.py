import logging

# 로거 설정
logger = logging.getLogger(__name__)


def get_one_cycle_lr_params(trial_config, train_dataloader):
    """
    OneCycleLR 스케줄러에 필요한 파라미터 (total_steps, max_lr, additional_params)를 계산하여 반환합니다.
    trial_config와 train_dataloader를 기반으로 계산합니다.
    파라미터 계산이 불가능하거나 스케줄러가 OneCycleLR이 아니면 None을 반환합니다.
    """
    scheduler_config = trial_config.get("trainer", {}).get("scheduler", {})
    if not scheduler_config or scheduler_config.get("name") != "OneCycleLR":
        logger.info("Scheduler is not configured or not OneCycleLR.")
        return None

    max_epochs = trial_config.get("trainer", {}).get("epochs")
    grad_accum_steps = trial_config.get("trainer", {}).get(
        "gradient_accumulation_steps", 1
    )

    num_training_steps_per_epoch = 0
    if hasattr(train_dataloader, "__len__") and len(train_dataloader) > 0:
        # 일반적인 DataLoader의 경우
        num_training_steps_per_epoch = len(train_dataloader) // grad_accum_steps
        if (
            num_training_steps_per_epoch == 0 and len(train_dataloader) > 0
        ):  # grad_accum_steps가 len(train_dataloader)보다 클때
            logger.warning(
                f"len(train_dataloader) ({len(train_dataloader)}) < grad_accum_steps ({grad_accum_steps}). "
                f"num_training_steps_per_epoch is forced to 1. Consider adjusting gradient_accumulation_steps."
            )
            num_training_steps_per_epoch = 1

    elif (
        hasattr(
            trainer := trial_config.get("_trainer_instance"),
            "estimated_stepping_batches",
        )
        and trainer.estimated_stepping_batches is not None
    ):
        # IterableDataset 등을 사용하고 Trainer에 의해 estimated_stepping_batches가 설정된 경우
        # 이 값은 전체 학습 스텝이므로, 에포크당 스텝으로 변환해야 함
        if max_epochs and max_epochs > 0:
            num_training_steps_per_epoch = (
                trainer.estimated_stepping_batches // max_epochs
            )
            if (
                num_training_steps_per_epoch == 0
                and trainer.estimated_stepping_batches > 0
            ):  # max_epochs가 너무 커서 0이 된 경우
                num_training_steps_per_epoch = 1  # 최소 1스텝
            logger.info(
                f"Using trainer.estimated_stepping_batches for scheduler: {trainer.estimated_stepping_batches} total, {num_training_steps_per_epoch} per epoch."
            )
        else:
            logger.warning(
                "trainer.estimated_stepping_batches is available, but max_epochs is not valid for calculating steps per epoch. Defaulting num_training_steps_per_epoch to a config value or 1."
            )
            # max_epochs 정보가 없을 경우, 설정 파일의 값을 사용하거나 기본값 사용
            num_training_steps_per_epoch = trial_config.get("trainer", {}).get(
                "num_training_steps_per_epoch_fallback", 1
            )

    else:  # Iterative Dataloader의 경우 또는 len() 사용 불가
        logger.warning(
            "Could not determine len(train_dataloader) for scheduler. "
            "Attempting to use 'num_training_steps_per_epoch_fallback' from config, or defaulting to 1. "
            "This might be incorrect for IterableDataset."
        )
        num_training_steps_per_epoch = trial_config.get("trainer", {}).get(
            "num_training_steps_per_epoch_fallback",
            1,  # config에서 이 값을 명시적으로 제공할 수 있도록 fallback 이름 변경
        )
        if num_training_steps_per_epoch == 1 and not trial_config.get(
            "trainer", {}
        ).get("num_training_steps_per_epoch_fallback"):
            logger.info(
                "Hint: Consider setting 'trainer.num_training_steps_per_epoch_fallback' in config if using IterableDataset and len(dataloader) is not available."
            )

    if not max_epochs or max_epochs <= 0:
        logger.error(
            "max_epochs not configured or invalid. Cannot calculate total_steps for scheduler."
        )
        return None

    if num_training_steps_per_epoch <= 0:
        logger.error(
            f"num_training_steps_per_epoch is {num_training_steps_per_epoch}, which is invalid. Cannot calculate total_steps."
        )
        return None

    total_steps = num_training_steps_per_epoch * max_epochs
    max_lr = (
        trial_config.get("trainer", {}).get("optimizer", {}).get("params", {}).get("lr")
    )
    additional_params = scheduler_config.get("params", {})

    if total_steps > 0 and max_lr is not None:
        logger.info(
            f"Calculated scheduler params: total_steps={total_steps}, max_lr={max_lr}, additional_params={additional_params}"
        )
        return {
            "total_steps": total_steps,
            "max_lr": max_lr,
            "additional_params": additional_params,
        }
    else:
        logger.error(
            f"Could not prepare scheduler params. total_steps={total_steps}, max_lr={max_lr}"
        )
        return None
