import lightning as L


class BaseModel(L.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Forward pass must be implemented in the child class."
        )
