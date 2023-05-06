import cv2
import librosa
import numpy as np
import scipy.stats


def calculate_spectrogram(y, hop_length=512, n_fft=4096, feature_len=94):
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length, n_fft=n_fft)), ref=np.max)

    if spec.shape[1] != feature_len:
        spec = cv2.resize(spec, dsize=(feature_len, spec.shape[0]))

    spec = (spec - spec.min()) / (spec.max() - spec.min())
    spec = spec * 2 - 1

    return spec


def calculate_mel_spectrogram(y, sr=16000):
    spec = librosa.feature.melspectrogram(y=y, sr=sr)
    return spec


def calculate_chroma(y, sr=16000, hop_length=512, feature_len=94):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

    if chroma.shape[1] != feature_len:
        chroma = cv2.resize(chroma, dsize=(feature_len, chroma.shape[0]))

    chroma = (chroma - chroma.min()) / (chroma.max() - chroma.min())
    chroma = chroma * 2 - 1

    return chroma


def calculate_mfcc_features(sample, tabular=True, sampling_rate=16000, n_mfcc=40, win_length=25, feature_len=94):
    mfcc = librosa.feature.mfcc(y=sample, sr=sampling_rate, n_mfcc=n_mfcc, win_length=win_length, dct_type=2)

    if tabular:
        mfcc = scipy.stats.zscore(mfcc, axis=1)
        mfcc = mfcc.flatten(order='C')
        return mfcc

    if mfcc.shape[1] != feature_len:
        mfcc = cv2.resize(mfcc, dsize=(feature_len, mfcc.shape[0]))

    mfcc = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min())
    mfcc = mfcc * 2 - 1

    return mfcc
