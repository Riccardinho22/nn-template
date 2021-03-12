from typing import *

import hydra
from omegaconf import DictConfig
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import f1 as f1_score
from pytorch_lightning.metrics.functional import accuracy as accuracy_score


class LightningMNISTClassifier(pl.LightningModule):

    def __init__(self, cfg: DictConfig):
        super(LightningMNISTClassifier, self).__init__()
        self.cfg = cfg
        self.n_classes = 10  # TODO with hydra

        self.l1 = torch.nn.Linear(28 * 28, 20)
        self.l2 = torch.nn.Linear(20, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.l1(x))
        output = F.log_softmax(self.l2(x), dim=1)
        return output

    def _evaluate(self, x, y):
        logits = self(x)
        loss = F.nll_loss(logits, y)

        y_hat = F.log_softmax(logits, dim=-1)
        y_preds = torch.argmax(y_hat, dim=-1)
        f1 = f1_score(y_preds, y, num_classes=self.n_classes)
        accuracy = accuracy_score(preds=y_preds, target=y)

        return loss, f1, accuracy

    # ------ start step ------

    def _shared_step(self, x: torch.Tensor, y: torch.Tensor, label: str):
        '''

        :param x:
        :param y:
        :param label:
        :return:
        '''
        loss, f1, accuracy = self._evaluate(x, y)

        return {f'{label}loss': loss, f'{label}f1': f1, f'{label}accuracy': accuracy}

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        x, y = batch
        return self._shared_step(x, y, label='')

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        x, y = batch
        return self._shared_step(x, y, label='val_')

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        x, y = batch

        return self._shared_step(x, y, label='test_')

    # ------ end step ------

    # ------ start epoch_end ------
    def _shared_epoch_end(self, outputs: List[dict], label: str) -> None:
        '''

        :param outputs:
        :param label:
        :return:
        '''

        avg_loss: torch.Tensor = torch.stack([x[f'{label}loss'] for x in outputs]).mean()
        self.log(f'avg_{label}loss', avg_loss, prog_bar=True)

        avg_f1: torch.Tensor = torch.stack([x[f'{label}f1'] for x in outputs]).mean()
        self.log(f'avg_{label}f1', avg_f1, prog_bar=True)

        avg_accuracy: torch.Tensor = torch.stack([x[f'{label}accuracy'] for x in outputs]).mean()
        self.log(f'avg_{label}accuracy', avg_accuracy, prog_bar=True)

    def training_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, label='')

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, label='val_')

    def test_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, label='test_')

    # ------ end epoch_end ------

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(self.cfg.optim.optimizer, params=self.parameters())
        scheduler = hydra.utils.instantiate(self.cfg.optim.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]
