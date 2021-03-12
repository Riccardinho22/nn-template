import logging
import argparse
from collections import OrderedDict
import os
import shutil
from pathlib import Path
from typing import *

import hydra
import torch
import omegaconf
import numpy as np
import flwr as fl
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger

from src.common.utils import load_envs

# Set the cwd to the project root
os.chdir(Path(__file__).parent.parent)

# Load environment variables
load_envs()


class PyTorchLightningClient(fl.client.NumPyClient):
    def __init__(self, model: pl.LightningModule, datamodule: pl.LightningDataModule, callbacks: List[pl.Callback],
                 logger, cfg: DictConfig,
                 ):
        self.model: pl.LightningModule = model
        self.datamodule: pl.LightningDataModule = datamodule
        self.callbacks: List[pl.Callback] = callbacks
        self.logger = logger
        self.cfg: DictConfig = cfg
        self.trainer: Optional[pl.Trainer] = None

    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.trainer = pl.Trainer(
            # fast_dev_run=True,
            # default_root_dir=hydra_dir,
            #logger=self.logger,
            callbacks=self.callbacks,
            deterministic=self.cfg.train.deterministic,
            val_check_interval=self.cfg.logging.val_check_interval,
            progress_bar_refresh_rate=self.cfg.logging.progress_bar_refresh_rate,
            **self.cfg.train.pl_trainer,
        )
        self.trainer.fit(model=self.model, datamodule=self.datamodule)
        return self.get_parameters(), 55000

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        res: dict = self.trainer.test(datamodule=self.datamodule)[0]
        print(res)
        return len(self.datamodule.test_dataloader().dataset), res["avg_test_loss"], res["avg_test_accuracy"]


def run(cfg: DictConfig) -> None:
    """
    Generic train loop

    :param cfg: run configuration, defined by hydra in /conf
    """
    if cfg.train.deterministic:
        seed_everything(cfg.client.cid)
    hydra.utils.log.info(f"Client CID: {cfg.client.cid}")

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, cfg=cfg
    )

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model, cfg=cfg)

    callbacks = []
    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info(f"Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info(f"Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train.model_checkpoints:
        hydra.utils.log.info(f"Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
            )
        )

    wandb_logger = None
    if "wandb" in cfg.logging:
        hydra.utils.log.info(f"Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            project=wandb_config.project,
            entity=wandb_config.entity,
            tags=cfg.core.tags,
            log_model=True,
        )
        hydra.utils.log.info(f"W&B is now watching <{wandb_config.watch.log}>!")
        wandb_logger.watch(
            model, log=wandb_config.watch.log, log_freq=wandb_config.watch.log_freq
        )


    client = PyTorchLightningClient(model=model, datamodule=datamodule, callbacks=callbacks, logger=wandb_logger,
                                    cfg=cfg)
    fl.client.start_numpy_client("[::]:8080", client)
    '''

    hydra.utils.log.info(f"Instantiating the Trainer")
    trainer = pl.Trainer(
        #fast_dev_run=True,
        default_root_dir=hydra_dir,
        #logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        val_check_interval=cfg.logging.val_check_interval,
        progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
        **cfg.train.pl_trainer,
    )

    hydra.utils.log.info(f"Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    hydra.utils.log.info(f"Starting testing!")
    trainer.test(model=model, datamodule=datamodule)
    '''
    # shutil.copytree(".hydra", Path(wandb_logger.experiment.dir) / "hydra")


@hydra.main(config_path="../conf", config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
