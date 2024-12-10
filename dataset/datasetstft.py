import functools
import os
from itertools import cycle
import h5py
from numpy import argwhere
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, random_split, DataLoader
from torch import Tensor
from tqdm import tqdm
from utils.Data_augmentation import rotation_2d, Reverse, Flip
import pandas as pd
from scipy.signal import stft




class RMLgeneral(Dataset):
    def __init__(self, samples, labels, SNR):
        self.samples = samples
        self.SNR = SNR
        self.label = torch.tensor(labels, dtype=torch.long)
        self.STFTs = []
        with tqdm(total=len(self.samples)) as t:
            t.set_description('Generating STFT:')
            for _,IQ in enumerate(self.samples):
                _,_,stp = stft(IQ[0,:],200000,'blackman',31,30,128)
                self.STFTs.append(np.expand_dims(stp[:32,:],0))
                t.update(1)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        return Tensor(self.STFTs[idx]), self.SNR[idx], self.label[idx]

class RMLval(Dataset):
    def __init__(self, samples, labels, SNR):
        self.samples = samples
        self.SNR = SNR
        self.label = torch.tensor(labels, dtype=torch.long)
        self.STFTs = []
        with tqdm(total=len(self.samples)) as t:
            t.set_description('Generating STFT:')
            for _,IQ in enumerate(self.samples):
                _,_,stp = stft(IQ[0,:],200000,'blackman',31,30,128)
                self.STFTs.append(np.expand_dims(stp[:32,:],0))
                t.update(1)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        return Tensor(self.STFTs[idx]), self.SNR[idx], self.label[idx]
    
class RMLtest(Dataset):
    def __init__(self, samples, labels, SNR):
        self.samples = samples
        self.SNR = SNR
        self.label = torch.tensor(labels, dtype=torch.long)
        self.STFTs = []
        with tqdm(total=len(self.samples)) as t:
            t.set_description('Generating STFT:')
            for _,IQ in enumerate(self.samples):
                _,_,stp = stft(IQ[0,:],200000,'blackman',31,30,128)
                self.STFTs.append(np.expand_dims(stp[:32,:],0))
                t.update(1)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        return Tensor(self.STFTs[idx]), self.SNR[idx], self.label[idx]




