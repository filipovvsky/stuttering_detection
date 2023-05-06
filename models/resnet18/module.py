import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.dataset import FullStutteringDataset


class ResNet18(pl.LightningModule):
    def __init__(self, config):
        super(ResNet18, self).__init__()
        self.config = config

        self.train_ds = FullStutteringDataset(config, 'train')
        self.val_ds = FullStutteringDataset(config, 'val')

        self.resnet = ResNet18Arch(config.features.in_channels, ResBlock, 256)

        self.binary_clf_hid = torch.nn.Linear(in_features=config.model.linear.in_features,
                                              out_features=config.model.linear.hidden_size)
        self.binary_clf = torch.nn.Linear(in_features=config.model.linear.hidden_size,
                                          out_features=1)
        self.labels_clf_hid = torch.nn.Linear(in_features=config.model.linear.in_features,
                                              out_features=config.model.linear.hidden_size)
        self.labels_clf = torch.nn.Linear(in_features=config.model.linear.hidden_size,
                                          out_features=len(self.config.stutter_labels))

    def forward(self, x):
        if self.config.features.in_channels == 1:
            x = x[:, None, :, :]
        else:
            x = x.permute((0, 3, 1, 2))

        x = self.resnet(x)

        labels = self.labels_clf_hid(x)
        labels = self.labels_clf(labels)

        return labels

    def training_step(self, batch, batch_idx):
        x, binary_y, labels_y = batch['features'], batch['binary_y'], batch['labels_y']

        y_hat_labels = self.forward(x)
        y_hat_binary = y_hat_labels.max(axis=1)[0]

        binary_loss = F.binary_cross_entropy_with_logits(y_hat_binary.squeeze(), binary_y)
        labels_losses = []
        for i, l in zip(range(len(self.config.stutter_labels)), self.config.stutter_labels):
            loss_part = F.binary_cross_entropy_with_logits(y_hat_labels.squeeze()[:, i], labels_y.squeeze()[:, i])
            labels_losses.append(loss_part)
            self.log(f'Train_losses/{l}_loss', loss_part)

        loss = (binary_loss * 1 + sum(labels_losses)) / (len(labels_losses) + 1)

        self.log('Train/Binary_loss', binary_loss)
        self.log('Train/loss', loss)

        return {'loss': loss}

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

        self.log('Val_losses/Binary_loss', binary_loss)
        self.log('Val_losses/Loss', loss)

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

        self.log(f'Val_metrics/Mean_f1', np.array(mean_f1).mean())
        self.log(f'Val_metrics/Mean_acc', np.array(mean_acc).mean())

        self.log('Val_losses/Binary_loss', binary_loss)
        self.log('Val_losses/Loss', loss)

    def configure_optimizers(self):
        parameters = [
            {'params': self.resnet.parameters(), 'lr': self.config.learning_rate},
            {'params': self.binary_clf.weight, 'lr': self.config.learning_rate},
            {'params': self.labels_clf.weight, 'lr': self.config.learning_rate}
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


class ResNet18Arch(nn.Module):
    def __init__(self, in_channels, resblock, outputs=1000):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )

        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = torch.flatten(input, 1)
        input = self.fc(input)

        return input


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, downsample_stride=(2, 2)):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=downsample_stride, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=downsample_stride),
                nn.BatchNorm2d(out_channels)
            )

        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = nn.ReLU()(self.bn3(self.conv3(input)))
        input = input + shortcut
        return nn.ReLU()(input)
