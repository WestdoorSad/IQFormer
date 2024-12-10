import os
import h5py
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch import Tensor
from tqdm import tqdm
from scipy.signal import stft

def zscore(X):
    """
      X (ndarray): Shape (m,n) input data, m examples, n features
      X_norm (ndarray): Shape (m,n)  input normalized by column
      mu (ndarray):     Shape (n,)   mean of each feature
      sigma (ndarray):  Shape (n,)   standard deviation of each feature
    """
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return X_norm

# Load the modulation classes. You can also copy and paste the content of classes-fixed.txt.
class RML2018_random(Dataset):
    def __init__(self, path: str, transform=None):
        self.path = path
        self.data = h5py.File(self.path, 'r')
        self.samples = self.data['samples']
        self.SNR = np.array(self.data['SNR'])
        self.label = self.data['label']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.Tensor(self.samples[idx]), self.SNR[idx], self.label[idx]


# class RMLgeneral(Dataset):
#     def __init__(self, samples, labels, SNR):
#         self.samples = samples
#         self.SNR = SNR
#         self.label = torch.tensor(labels, dtype=torch.long)
#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):

#         return torch.Tensor(self.samples[idx]), self.SNR[idx], self.label[idx]
# class RMLtest(Dataset):
#     def __init__(self, samples, labels, SNR):
#         self.samples = samples
#         self.SNR = SNR
#         self.label = torch.tensor(labels, dtype=torch.long)
#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):

#         return torch.Tensor(self.samples[idx]), self.SNR[idx], self.label[idx]
class RMLgeneral(Dataset):
    def __init__(self, samples, labels, SNR):
        self.samples = samples
        self.SNR = SNR
        self.label = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _,_,stp = stft(self.samples[idx][0,:],1.0,'blackman',31,30,128) 
        return torch.Tensor(self.samples[idx]),torch.Tensor(np.expand_dims(stp[:32,:],0)), self.SNR[idx], self.label[idx]
    
class RMLval(Dataset):
    def __init__(self, samples, labels, SNR):
        self.samples = samples
        self.SNR = SNR
        self.label = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _,_,stp = stft(self.samples[idx][0,:],1.0,'blackman',31,30,128) 
        return torch.Tensor(self.samples[idx]),torch.Tensor(np.expand_dims(stp[:32,:],0)), self.SNR[idx], self.label[idx]    
class RMLtest(Dataset): 
    def __init__(self, samples, labels, SNR):
        self.samples = samples
        self.SNR = SNR
        self.label = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _,_,stp = stft(self.samples[idx][0,:],1.0,'blackman',31,30,128) 
        return torch.Tensor(self.samples[idx]),torch.Tensor(np.expand_dims(stp[:32,:],0)), self.SNR[idx], self.label[idx]   

