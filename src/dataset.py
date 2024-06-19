import glob
import soundfile as sf
from torch.utils.data import Dataset
import torch
import numpy as np
import librosa as lb
from IPython import embed

class AudioMNISTDataset(Dataset):
    def __init__(self, data_path, feature, test=False):
        self.data_path = data_path
        self.feature = feature
        self.test = test

    def __len__(self):
        if not self.test:
            return len(glob.glob(self.data_path+'/train/*'))
        else:
            return len(glob.glob(self.data_path+'/test/*'))

    def __getitem__(self, idx):
        # Get audio paths
        if not self.test:
            audio_paths = glob.glob(self.data_path + '/train/*')
        else:
            audio_paths = glob.glob(self.data_path + '/test/*')
        
        # Get audio data and labels
        audio, fs = sf.read(audio_paths[idx])
        label = audio_paths[idx].split('/')[-1].split('_')[0].split('\\')[1]
        # Extract features
        if self.feature == 'raw_waveform':
            # insertar código acá
            feat = torch.tensor(audio, dtype=torch.float)
        elif self.feature == 'audio_spectrum':
            feat = self.dft(audio,fs)
        elif self.feature == 'mfcc':
            feat = self.mfcc(audio, fs)
        
        feat =  feat.view(-1)
        feat = feat.type(torch.float)
        # print("label: ", label, "label_type, ", type(label))
        # embed()
        label = torch.tensor(int(label), dtype=torch.long)

        return feat, label

    @staticmethod
    def dft(audio: np.ndarray, fs: float) -> torch.Tensor:
        """
        Calculates the discrete Fourier transform of the audio data, normalizes the result and trims it, preserving only positive frequencies.
        Args:
            audio (Numpy array): audio file to process.
            fs (float): sampling frequency of the audio file.
        Returns:
            audio_f (Tensor): spectral representation of the audio data.
        """
        # insertar código
        audio_f = np.fft.fft(audio)
        # esto es porque la primera mmitad da negativo
        audio_f = np.abs(audio_f[:(len(audio_f)//2)+1])

        audio_f = audio_f / np.max(audio_f)
        # normalizamos dividiento por el valor maximo del audio
        # Ensure the tensor has the right shape
        audio_f = torch.tensor(audio_f, dtype=torch.float)
        return audio_f 

    @staticmethod
    def mfcc(audio, fs):
        """
        Calculates the Mel Frequency Cepstral Coefficients (MFCCs) of the audio data.
        Args:
            audio (Numpy array): audio file to process.
            fs (float): sampling frequency of the audio file.
            mfcc_params (dictionary): the keys are 'n_fft', the length of the FFTs, and 'window', the type of window to be used in the STFTs (see scipy.signal.get_window)
        Returns:
            mfcc (Tensor): MFCC of the input audio file.
        """
        # insertar código
        mfcc = lb.feature.mfcc(y=audio, sr=fs)
        return torch.tensor(mfcc, dtype=torch.float)
