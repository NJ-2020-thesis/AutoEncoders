from pytorch_lightning.loggers.neptune import NeptuneLogger

class Logger:
    def __init__(self, exp_name: str, max_epochs: int,
                 batch_size: int, learning_rate: float):
        self.neptune_logger = NeptuneLogger(
            api_key="ANONYMOUS",
            project_name="shared/pytorch-ae-trainer",
            close_after_fit=False,
            experiment_name=exp_name,
            params={"max_epochs": max_epochs,
                    "batch_size": batch_size,
                    "lr": learning_rate},  # Optional,
            tags=["pytorch-lightning", "mlp"],
            # upload_source_files=['*.py', '*.yaml'],
            upload_stderr=False,
            upload_stdout=False
        )

    def get_logger(self):
        return self.neptune_logger