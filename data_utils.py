import random
import numpy as np
import torch
import torch.utils.data
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import load_wav_to_torch
import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence
from layers import TacotronSTFT

from text import text_to_sequence  # Ensure you have a text preprocessing module

class TextMelUnitLoader(torch.utils.data.Dataset):
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.stft = TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

    def get_mel_text_unit_pair(self, audiopath_and_text):
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        unit = self.get_units(audiopath)  # Add logic to load unit data or return an empty tensor if not available
        return (text, mel, unit)

    def get_mel(self, filename):
        if self.load_mel_from_disk:
            melspec = torch.from_numpy(np.load(filename, allow_pickle=True))
        else:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError(f"{sampling_rate} SR doesn't match target {self.stft.sampling_rate} SR")
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def get_units(self, filename):
        # Logic to load unit data
        # If no unit data is available, return an empty tensor
        return torch.tensor([])  # Replace with actual logic for unit extraction

    def __getitem__(self, index):
        return self.get_mel_text_unit_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)

class TextUnitCollate():
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step

        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        # Handle unit sequences, allowing for the possibility of missing units
        unit_padded = torch.LongTensor(len(batch), max([len(x[2]) for x in batch if x[2].nelement() > 0]) if any(x[2].nelement() > 0 for x in batch) else 0)
        if unit_padded.size(1) > 0:
            unit_padded.zero_()
            for i in range(len(ids_sorted_decreasing)):
                if batch[ids_sorted_decreasing[i]][2].nelement() > 0:
                    unit = batch[ids_sorted_decreasing[i]][2]
                    unit_padded[i, :unit.size(0)] = unit

        return text_padded, input_lengths, mel_padded, unit_padded, output_lengths
