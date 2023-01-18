from torch.utils.data import Dataset
import torchaudio
import torch
import glob
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm


class SMDdataset(Dataset):

    # Load audio files
    # audio dir is the path to the directory of the dataset
    def __init__(self, audio_dir, transformation, target_sample_rate, num_samples):
        self.audio_dir = audio_dir
        self.audio_paths = glob.glob(self.audio_dir + "/*.wav")
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.x_signals =[]
        self.y_signals = []
        self.mean = 0
        self.std = 0
    
    # Function to call within the class
    def __call__(self):
        self.mean, self.std = self.get_means_and_stds()
        for i in range(len(self.x_signals)):
            signal = self.x_signals[i]
            signal = self.normalize(signal, self.mean, self.std)
            self.y_signals.append(signal)
            #signal = self.mask_middle(signal)
            signal = self.mask_end(signal)
            self.x_signals[i] = signal

        return self.x_signals, self.y_signals
    
    # Get item function
    def __getitem__(self, item):
        return self.x_signals[1], self.y_signals[1]


    # Function to get mean and standard deviations of each spectrogram
    def get_means_and_stds(self):
        means = []
        stds = []
        for index in tqdm(range(len(self.audio_paths))):
            audio_path = self.get_audio_sample_path(index)
            x_signal = self.get_melspectrogram(audio_path)
            self.x_signals.append(x_signal)
            means.append(torch.mean(x_signal))
            stds.append(torch.std(x_signal))

        return torch.mean(torch.stack(means)), torch.mean(torch.stack(stds))

    
    # Function to get mel spectrogram from audio data using pytoch audio transformation
    def get_melspectrogram(self, audio_sample_path):
        x_signal, sr = torchaudio.load(audio_sample_path, normalize=True)
        x_signal = self.mono_if_necessary(x_signal)
        x_signal = self.right_pad_if_necessary(x_signal)
        x_signal = self.resample_if_neccessary(x_signal, sr)
        x_signal = self.transformation(x_signal)
        return x_signal

    def normalize(self, signal, mean, std):
        return (signal - mean)/std

    # Get path to audio sample
    def get_audio_sample_path(self, index):
        return self.audio_paths[index]

    # Mix down to mono if needed
    def mono_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim =True)
        return signal

    # Resample to 22050 if needed
    def resample_if_neccessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resample = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resample(signal)
        return signal

    # Cut audio signal to 10s if needed
    def cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

   # Pad the right side of signal with zeros if signal is too short
    def right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    # Mask the middle of the signal (for x_signal only)
    def mask_middle(self, signal):
        mask = torch.ones((1, 64, 431))
        # Identify where to mask (10 time bins are being masked in the middle)
        mask[0,0:64,200:210] = 0
        signal = signal * mask
        return signal

    # Mask the end of the signal (for x_signal only)
    def mask_end(self, signal):
        # Create mask
        mask = torch.ones((1, 64, 431))
        # Identify where to mask (here I am masking 25 time bins)
        mask[0,0:64,406:431] = 0
        signal = signal * mask
        return signal


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



if __name__ == '__main__':

    AUDIO_DIR = '/Users/eleanorrow/Desktop/DL4AM_Project/dataset/sliced_data'
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 220500 #want 10s of audio

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device {device}.')


    # Instantiate Mel Spectrogram transformation
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    SMDObject = SMDdataset(AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES)
    x_signals, y_signals = SMDObject()

    torch.save(x_signals, 'x_signals_data.pt')
    torch.save(y_signals, 'y_signals_data.pt')


    #print(f'There are {len(SMDobject)} samples/audio files in the dataset audio directory')

    # Return the x and y signals as a tensor

    print(torch.stack(x_signals).shape)
    print(torch.stack(y_signals).shape)

    # Plot one spectrogram of signals to show that masking has worked
    x_signal, y_signal = SMDObject[1]

    plot_mel_spectrogram(x_signal[0], title="MelSpectrogram - torchaudio", ylabel="mel freq")
    plot_mel_spectrogram(y_signal[0], title="MelSpectrogram - torchaudio", ylabel="mel freq")

    print(f'The size of the image is {x_signal.shape}, where signal is [num_channels, height, width]')











