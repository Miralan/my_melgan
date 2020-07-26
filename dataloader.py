import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn

from librosa.core import load
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn

from pathlib import Path
import numpy as np
import random


def getfilepath(dir_path='wavs'):
    p = Path(dir_path)
    filelist = []
    print(p)
    for pth in p.iterdir():
        for fpth in pth.iterdir():
            filelist.append(fpth)
    return filelist

class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=16000,
        n_mel_channels=80,
        mel_fmin=90,
        mel_fmax=7600,
    ):
        super().__init__()

        """
            FFT parameters
        """

        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = audio.unsqueeze(1)
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        #log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        temp = 20 * torch.log10(torch.clamp(mel_output, min=1e-5)) -16
        log_mel_spec = torch.clamp((temp + 100.0)/100, min=0, max=1).squeeze(0)
        return log_mel_spec

class Audio2Mel_V(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=16000,
        n_mel_channels=80,
        mel_fmin=90,
        mel_fmax=7600,
    ):
        super().__init__()

        """
            FFT parameters
        """

        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        #log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        temp = 20 * torch.log10(torch.clamp(mel_output, min=1e-5)) -16
        log_mel_spec = torch.clamp((temp + 100.0)/100, min=0, max=1)
        return log_mel_spec

class AudioDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, train_path='wavs', segment_length=8192, sampling_rate=16000, augment=True):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.audio_files = getfilepath(train_path)
        random.seed(1234)
        random.shuffle(self.audio_files)
        self.augment = augment
        self.wav2mel = Audio2Mel()

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio, sampling_rate = self.load_wav_to_torch(filename)
        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data

        # audio = audio / 32768.0
        audio = audio.unsqueeze(0)
        melspec = self.wav2mel(audio)
        return audio.to(dtype=torch.float32), melspec.to(dtype=torch.float32)

    def __len__(self):
        return len(self.audio_files)

    def load_wav_to_torch(self, full_path):
        """
        Loads wavdata into torch array
        """
        data, sampling_rate = load(full_path, sr=self.sampling_rate)
        data = 0.95 * normalize(data)

        if self.augment:
            amplitude = np.random.uniform(low=0.3, high=1.0)
            data = data * amplitude

        return torch.from_numpy(data).to(dtype=torch.float32), sampling_rate


