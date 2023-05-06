import os
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models.resnet18.module import ResNet18


@hydra.main(version_base=None, config_path='configs')
def train(config):
    model = ResNet18(config)

    run_name = Path(config.config_name).name
    out_dir = Path(config.run_path) / run_name

    logger_name = 'lightning_logs'
    (out_dir / logger_name).mkdir(parents=True, exist_ok=True)

    tb_logger = TensorBoardLogger(
        save_dir=str(out_dir),
        name=logger_name,
        version=config.version)

    run_path = out_dir / logger_name / f'version_{tb_logger.version}'
    checkpoint_name_pattern = '{epoch}-{Val_metrics-Mean_f1:.4f}'
    checkpoints_path = str(run_path / 'checkpoints')

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_path,
        filename=checkpoint_name_pattern,
        save_top_k=config.save_top_k,
        verbose=True,
        mode=config.monitor_mode,
        monitor=config.best_model_metric
    )

    trainer = Trainer(
        max_epochs=config.max_epoch,
        logger=[tb_logger],
        gpus=list(config.gpus),
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        accelerator='gpu',
    )

    os.makedirs(run_path, exist_ok=True)
    OmegaConf.save(config, run_path / 'config.yaml', resolve=True)

    trainer.fit(model)


if __name__ == '__main__':
    train()
