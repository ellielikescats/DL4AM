# %%
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as T
import tqdm
import librosa
from matplotlib import pyplot as plt

from dataModule import DataModule
from VAE import VAE
from AE import Convolutional_AE


# %%
mean = 8.7415
std = 50.6754


class dB_to_Amplitude(nn.Module):
    def __call__(self, features):
        return (torch.from_numpy(np.power(10.0, features.numpy() / 10.0)))


def get_waveform_from_logMel(features, n_fft=1024, hop_length=512, sr=22050):
    n_mels = features.shape[-2]
    inverse_transform = torch.nn.Sequential(
        # dB_to_Amplitude(),
        torchaudio.transforms.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sr),
        torchaudio.transforms.GriffinLim(n_fft=1024, hop_length=512)
    )
    waveform = inverse_transform(torch.squeeze(features))
    return torch.unsqueeze(waveform, 0)


# Function to plot mel-spectrogram
def plot_mel_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


def output_mel_spectrogram_to_wav(spec_normalised):
    # denormalize
    denorm = spec_normalised * std + mean

    # Plot mel spectrogram of denormalised signal

    plot_mel_spectrogram(denorm, title="MelSpectrogram - torchaudio", ylabel="mel freq")

    # griffin lim to get audio signal
    # signal = T.griffinlim(denorm, torch.hann_window(1024), 1024, 512, 1024, 2.0, 64, momentum=0.5, length=220500, rand_init=False)

    signal = get_waveform_from_logMel(denorm)

    return signal


if __name__ == "__main__":
    # Choose model
    model = Convolutional_AE()
    # model = VAE()

    datamodule = DataModule()
    datamodule.setup()

    test_dataloader = datamodule.test_dataloader()

    # Load model
    model.load_state_dict(torch.load('AE.pt'))
    model.eval()

    for i, batch in enumerate(test_dataloader):
        x, y = batch

        # Denormalise orignal signal as input
        original_signal = torch.squeeze(x[0])
        original_signal = output_mel_spectrogram_to_wav(original_signal)

        # Get output for AE model
        outs = model(x)

        # Get ouputs for VAE model
        # outs, mu, logvar = model(x)

        # take first of batch and remove channel dimension -> output dimension is w x h
        first = torch.squeeze(outs[0])

        first = first.detach()

        signal = output_mel_spectrogram_to_wav(first)

        # wav conversion
        torchaudio.save('asdf.wav', signal, 22050)

        a = 2

        break

# %%
