from typing import Any

import torch
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, Recall, Precision, F1Score

import model


class TrainModule(pl.LightningModule):
    def __init__(
        self,
        train_config: dict[str, Any],
        model_config: dict[str, Any],
        input_size: int,
        seq_len: int,
    ):
        super().__init__()

        self.train_setting = train_config.get("setting")
        if self.train_setting is None:
            raise ValueError(
                "Training setting configuration is missing in the provided config."
            )
        self.data_config = train_config.get("data")
        if self.data_config is None:
            raise ValueError(
                "Data configuration is missing in the provided config."
            )

        self.model_name = self.train_setting.get("model_name")
        self.task_type = self.train_setting.get("task_type")
        self.model = getattr(model, self.model_name)(
            model_config.get(self.model_name), input_size, seq_len
        )
        self.optimizer_class = getattr(optim, self.train_setting.get("optimizer"))
        self.loss_func = getattr(nn, self.train_setting.get("loss"))()

        self.scheduler_name = self.train_setting.get("scheduler").get("name")
        self.learning_rate = self.train_setting.get("learning_rate")

        if self.task_type != "regression":
            num_class = self.data_config.get("num_class")
            self.acc = Accuracy(
                task=self.task_type, num_classes=num_class, average="macro",
            )
            self.recall = Recall(
                task=self.task_type, num_classes=num_class, average="macro",
            )
            self.precision = Precision(
                task=self.task_type, num_classes=num_class, average="macro",
            )
            self.f1 = F1Score(
                task=self.task_type, num_classes=num_class, average="macro",
            )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)
        configure_dict = {"optimizer": optimizer}

        if self.scheduler_name is not None:
            scheduler_params = self.train_setting.get("scheduler").get("params")
            scheduler_class = getattr(
                lr, self.train_setting.get("scheduler").get("name")
            )
            scheduler = scheduler_class(optimizer, **scheduler_params)
            configure_dict["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
            }

        return configure_dict

    def training_step(self, batch: dict[str, Any]) -> torch.Tensor:
        inputs = batch.get("X")
        target = batch.get("y")
        output, _ = self.model(inputs)
        loss = self.loss_func(output, target)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )

        if self.task_type != "regression":
            self.log_dict(
                {
                    "train_acc": self.acc(output.argmax(1), target),
                    "train_recall": self.recall(output.argmax(1), target),
                    "train_precision": self.precision(output.argmax(1), target),
                    "train_f1": self.f1(output.argmax(1), target),
                },
                logger=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        if self.scheduler_name is not None:
            lr_schedulers = self.lr_schedulers()
            if lr_schedulers is not None:
                if isinstance(lr_schedulers, list):
                   lr = [torch.tensor(sched.get_last_lr()) for sched in lr_schedulers]
                else:
                   lr = [torch.tensor(lr_schedulers.get_last_lr())]
                if isinstance(lr, list):
                    self.log(
                        "learning_rate",
                        lr[0],
                        on_epoch=True,
                        logger=True,
                        prog_bar=True,
                    )
                else:
                    self.log(
                        "learning_rate",
                        lr,
                        on_epoch=True,
                        logger=True,
                        prog_bar=True,
                    )
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        inputs = batch[0]
        target = batch[1]
        output, _ = self.model(inputs)
        loss = self.loss_func(output, target)

        self.log(
            "valid_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )

        if self.task_type != "regression":
            self.log_dict(
                {
                    "valid_acc": self.acc(output.argmax(1), target),
                    "valid_recall": self.recall(output.argmax(1), target),
                    "valid_precision": self.precision(output.argmax(1), target),
                    "valid_f1": self.f1(output.argmax(1), target),
                },
                logger=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def test_step(self, batch: dict[str, Any]) -> torch.Tensor:
        inputs = batch.get("X")
        target = batch.get("y")
        output, _ = self.model(inputs)
        loss = self.loss_func(output, target)
        self.log("test_loss", loss, on_epoch=True, logger=True, prog_bar=True)

        return loss