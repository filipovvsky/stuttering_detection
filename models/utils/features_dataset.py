from pathlib import Path

import librosa.display
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from models.common.features import calculate_mfcc_features, calculate_chroma, calculate_spectrogram,\
    calculate_mel_spectrogram


class FullStutteringDataset(Dataset):
    def __init__(self, config, variant):
        assert variant in ['train', 'val', 'test']

        self.config = config
        self.folds_csv = pd.read_csv(self.config.folds_csv)

        if variant == 'train':
            self.folds = [f for f in range(20) if f not in config.valid_folds + config.test_folds]

        elif variant == 'val':
            self.folds = config.valid_folds

        else:
            self.folds = config.test_folds

        data = pd.read_csv(config.raw_csv_path)
        data = data.merge(self.folds_csv, on=['Show', 'EpId', 'ClipId']).reset_index(drop=True)
        self.data = data[data.fold.isin(self.folds)].reset_index(drop=True)

    def __getitem__(self, idx):
        sample = self.data.loc[idx]

        audio_path = Path(self.config.data_path) / str(
            sample.Show) / f'{str(sample.EpId):>03}' / f'{str(sample.Show)}_{str(sample.EpId):>03}_{str(sample.ClipId)}.wav'

        features = self.load_single_sample(audio_path)

        binary_y = sample[self.config.stutter_labels].sum().clip(max=1)
        labels_y = sample[self.config.stutter_labels].to_numpy(dtype=np.float32).clip(max=1)
        binary_y = torch.tensor(binary_y, dtype=torch.float32)
        labels_y = torch.tensor(labels_y, dtype=torch.float32)

        out = {'features': features,
               'binary_y': binary_y,
               'labels_y': labels_y}

        return out

    def load_single_sample(self, audio_path):
        features = []
        y, sr = librosa.load(audio_path, sr=16000)

        if self.config.features.mfcc:
            mfcc = calculate_mfcc_features(y, tabular=self.config.features.tabular_mfcc, sampling_rate=sr, n_mfcc=47,
                                           win_length=25)
            mfcc = torch.tensor(mfcc, dtype=torch.float32)

            features.append(mfcc)

        if self.config.features.spectrogram:
            spec = calculate_spectrogram(y, sr, n_fft=1024)
            spec = torch.tensor(spec, dtype=torch.float32)

            features.append(spec)

        if self.config.features.chroma:
            chroma = calculate_chroma(y, sr=sr)
            chroma = torch.tensor(chroma, dtype=torch.float32)

            features.append(chroma)

        if self.config.features.mel_spectrogram:
            spec = calculate_mel_spectrogram(y, sr=sr)
            spec = torch.tensor(spec, dtype=torch.float32)

            features.append(spec)

        features = torch.vstack(features).squeeze()
        return features

    def __len__(self):
        return len(self.data)
