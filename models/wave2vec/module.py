import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification

from models.transformer.dataset import FullStutteringDataset


class Wave2Vec(pl.LightningModule):
    def __init__(self, config) -> None:
        super(Wave2Vec, self).__init__()
        self.config = config

        self.train_ds = FullStutteringDataset(config, 'train')
        self.val_ds = FullStutteringDataset(config, 'val')

        self.model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base",
                                                                     num_labels=len(self.config.stutter_labels) + 1)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    def forward(self, x):
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        x, binary_y, labels_y = batch['features'], batch['binary_y'], batch['labels_y']

        y_hat = self.forward(x)
        y_hat_binary = y_hat[:, -1]
        y_hat_labels = y_hat[:, :5]

        binary_loss = F.binary_cross_entropy_with_logits(y_hat_binary.squeeze(), binary_y)

        labels_losses = []
        for i, l in zip(range(len(self.config.stutter_labels)), self.config.stutter_labels):
            loss_part = F.binary_cross_entropy_with_logits(y_hat_labels.squeeze()[:, i], labels_y.squeeze()[:, i])
            labels_losses.append(loss_part)
            self.log(f'Train_losses/{l}_loss', loss_part)

        loss = (binary_loss * self.config.loss.binary_coeff + sum(labels_losses)) / (
                len(labels_losses) + self.config.loss.binary_coeff)

        self.log('Train/Binary_loss', binary_loss)
        self.log('Train/loss', loss)

        return {'loss': binary_loss}

    def validation_step(self, batch, batch_idx):
        x, y_binary, y_labels = batch['features'], batch['binary_y'], batch['labels_y']

        y_hat_labels = self.forward(x)
        y_hat_binary = y_hat_labels.max(axis=1)[0]

        return {'y_hat_binary': y_hat_binary, 'binary_y': y_binary,
                'y_hat_labels': y_hat_labels, 'labels_y': y_labels}

    def validation_epoch_end(self, outputs):
        y_hat_binary = torch.cat([batch['y_hat_binary'] for batch in outputs])
        y_hat_labels = torch.cat([batch['y_hat_labels'] for batch in outputs])
        binary_y = torch.cat([batch['binary_y'] for batch in outputs])
        labels_y = torch.cat([batch['labels_y'] for batch in outputs])

        binary_loss = F.binary_cross_entropy_with_logits(y_hat_binary.squeeze(), binary_y)

        labels_losses = []
        for i, l in zip(range(len(self.config.stutter_labels)), self.config.stutter_labels):
            loss_part = F.binary_cross_entropy_with_logits(y_hat_labels.squeeze()[:, i], labels_y.squeeze()[:, i])
            labels_losses.append(loss_part)
            self.log(f'Val_losses/{l}_loss', loss_part)

        loss = (binary_loss * 1 + sum(labels_losses)) / (len(labels_losses) + 1)

        binary_y_hat = torch.sigmoid(y_hat_binary).cpu().numpy()
        label_y_hat = torch.sigmoid(y_hat_labels).cpu().numpy()

        binary_y = binary_y.cpu().detach()
        labels_y = labels_y.cpu().detach()

        binary_y_hat[binary_y_hat < self.config.threshold] = 0.
        binary_y_hat[binary_y_hat >= self.config.threshold] = 1.

        label_y_hat[label_y_hat < self.config.threshold] = 0.
        label_y_hat[label_y_hat >= self.config.threshold] = 1.

        bin_f1 = f1_score(binary_y, binary_y_hat)
        bin_acc = accuracy_score(binary_y, binary_y_hat)

        mean_f1 = [bin_f1]
        mean_acc = [bin_acc]
        for i, label in zip(range(5), self.config.stutter_labels):
            lab_f1 = f1_score(labels_y[:, i], label_y_hat[:, i])
            lab_acc = accuracy_score(labels_y[:, i], label_y_hat[:, i])
            mean_f1.append(lab_f1)
            mean_acc.append(lab_acc)
            self.log(f'Val_metrics/{label}_Accuracy', lab_acc)
            self.log(f'Val_metrics/{label}_F1', lab_f1)


        self.log('Val_losses/Binary_loss', binary_loss)
        self.log('Val_losses/Loss', loss)
        self.log(f'Val_metrics/Mean_f1', np.array(mean_f1).mean())
        self.log(f'Val_metrics/Mean_acc', np.array(mean_acc).mean())


    def configure_optimizers(self):
        parameters = [
            {'params': self.model.parameters(), 'lr': self.config.learning_rate}
        ]

        optimizer = torch.optim.Adam(parameters,
                                     lr=self.config.learning_rate)

        return optimizer

    def train_dataloader(self):
        return self.get_loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self.get_loader(self.val_ds, shuffle=False)

    def get_loader(self, dataset, shuffle):
        return DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=shuffle,
        )
