from pytorch_lightning.callbacks import Callback
import copy
import torch
import logging

logger = logging.getLogger(__name__)


class InMemoryBestModelSaver(Callback):
    def __init__(self, monitor="val_loss", mode="min"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_score = torch.tensor(float("inf") if mode == "min" else -float("inf"))
        self.best_weights = None
        self.best_epoch = torch.tensor(-1)

    def on_validation_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            logger.warning(
                f"[InMemoryBestModelSaver] Monitored metric '{self.monitor}' not found in callback_metrics. Skipping."
            )
            return

        if not isinstance(current, torch.Tensor):
            current = torch.tensor(current, device=self.best_score.device)

        is_better = (
            (current < self.best_score)
            if self.mode == "min"
            else (current > self.best_score)
        )
        if is_better:
            self.best_score = current
            self.best_weights = copy.deepcopy(pl_module.state_dict())
            self.best_epoch = torch.tensor(trainer.current_epoch)
            logger.debug(
                f"[InMemoryBestModelSaver] New best model saved at epoch {self.best_epoch.item()} with {self.monitor}: {self.best_score.item()}"
            )

    def on_test_start(self, trainer, pl_module):
        if self.best_weights is not None:
            pl_module.load_state_dict(self.best_weights)
            logger.info(
                f"[InMemoryBestModelSaver] Loaded best model weights from epoch {self.best_epoch.item()} (score: {self.best_score.item()}) for testing."
            )
        else:
            logger.warning(
                "[InMemoryBestModelSaver] No best weights to load for testing. Using current model weights."
            )

    def state_dict(self):
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
        }

    def load_state_dict(self, state_dict):
        self.monitor = state_dict.get("monitor", self.monitor)
        self.mode = state_dict.get("mode", self.mode)
        self.best_score = state_dict.get("best_score", self.best_score)
        self.best_epoch = state_dict.get("best_epoch", self.best_epoch)
        logger.debug(
            f"[InMemoryBestModelSaver] State loaded. Best score: {self.best_score.item()}, Best epoch: {self.best_epoch.item()}"
        )
