import numpy as np
import random
from scipy.signal import resample
import scipy.signal
try:
    import librosa
except ImportError:
    librosa = None


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


class Reshape(object):
    def __call__(self, seq):
        #print(seq.shape)
        return seq.transpose()


class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)

class Audio_Normalization(object):
    def __call__(self, seq):
        return seq / 32768.0


class AddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)

class AddGaussianBySNR(object):
    def __init__(self, SNR=3):
        self.snr = SNR

    def __call__(self, seq):
        if self.snr == 99:
            return seq
        else:
            signal_average_power = np.mean(seq**2)
            signal_average_db = 10 * np.log10(signal_average_power)
            noise_average_db = signal_average_db - self.snr
            noise_average_power = 10 ** (noise_average_db / 10)
            mean_noise = 0 
            noise = np.random.normal(mean_noise, np.sqrt(noise_average_power), seq.shape)
            return seq + noise


class RandomAddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class RandomScale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
            scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
            return seq*scale_matrix


class RandomStretch(object):
    def __init__(self, sigma=0.3):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            seq_aug = np.zeros(seq.shape)
            len = seq.shape[1]
            length = int(len * (1 + (random.random()-0.5)*self.sigma))
            for i in range(seq.shape[0]):
                y = resample(seq[i, :], length)
                if length < len:
                    if random.random() < 0.5:
                        seq_aug[i, :length] = y
                    else:
                        seq_aug[i, len-length:] = y
                else:
                    if random.random() < 0.5:
                        seq_aug[i, :] = y[:len]
                    else:
                        seq_aug[i, :] = y[length-len:]
            return seq_aug


class RandomCrop(object):
    def __init__(self, crop_len=20):
        self.crop_len = crop_len

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            max_index = seq.shape[1] - self.crop_len
            random_index = np.random.randint(max_index)
            seq[:, random_index:random_index+self.crop_len] = 0
            return seq

class Normalize(object):
    def __init__(self, type = "0-1"): # "0-1","-1-1","mean-std"
        self.type = type

    def __call__(self, seq):
        if  self.type == "0-1":
            seq =(seq-seq.min())/(seq.max()-seq.min())
        elif  self.type == "-1-1":
            seq = 2*(seq-seq.min())/(seq.max()-seq.min()) + -1
        elif self.type == "mean-std" :
            seq = (seq-seq.mean())/seq.std()
        else:
            raise NameError('This normalization is not included!')

        return seq

class RandomReverse(object):
    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            return seq[::-1]

class RandomLocalReverse(object):
    def __init__(self, rev_len=128):
        self.rev_len = rev_len
        
    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            max_index = seq.shape[1] - self.rev_len
            random_index = np.random.randint(1, max_index)
            seq[:, random_index:random_index+self.rev_len] = seq[:,random_index+self.rev_len-1:random_index-1:-1]
            return seq


class STFT_Transform(object):
    """
    Perform STFT on each channel of the input sequence and stack the results.
    Output: (3, freq_bins, time_frames)
    """
    def __init__(self, n_fft=256, hop_length=128, window='hann'):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window

    def __call__(self, seq):
        # seq: shape (channels, signal_length)
        stft_imgs = []
        for i in range(seq.shape[0]):
            f, t, Zxx = scipy.signal.stft(seq[i], nperseg=self.n_fft, noverlap=self.n_fft-self.hop_length, window=self.window)
            stft_img = np.abs(Zxx)
            stft_imgs.append(stft_img)
        stft_imgs = np.stack(stft_imgs, axis=0)  # (channels, freq_bins, time_frames)
        return stft_imgs

class MelSpectrogram_Transform(object):
    """
    Perform Mel-Spectrogram on each channel of the input sequence and stack the results.
    Output: (channels, n_mels, time_frames)
    Dependencies: librosa
    """
    def __init__(self, sr=44100, n_fft=512, hop_length=256, n_mels=64):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        if librosa is None:
            raise ImportError('librosa is required for MelSpectrogram_Transform')

    def __call__(self, seq):
        # seq: shape (num_samples, channels) or (channels, num_samples)
        if seq.shape[0] > seq.shape[1]:  # if (num_samples, channels) type，transpose it
            seq = seq.transpose()
        
        mel_imgs = []
        for i in range(seq.shape[0]):
            mel_spec = librosa.feature.melspectrogram(
                y=seq[i], 
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            mel_imgs.append(mel_spec)
        mel_imgs = np.stack(mel_imgs, axis=0)  # (channels, n_mels, time_frames)
        
        return mel_imgs.astype(np.float32)
