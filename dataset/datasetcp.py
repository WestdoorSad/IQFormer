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
from utils.Data_augmentation import rotation_2d, Reverse, Flip
import pandas as pd




class RMLgeneral(Dataset):
    def __init__(self, samples, labels, SNR):
        self.samples = torch.tensor(samples,dtype=torch.float)
        self.SNR = SNR
        self.label = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        return self.samples[idx], self.SNR[idx], self.label[idx]

class RMLval(Dataset):
    def __init__(self, samples, labels, SNR):
        self.samples = torch.tensor(samples,dtype=torch.float)
        self.SNR = SNR
        self.label = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        return self.samples[idx], self.SNR[idx], self.label[idx]
    
class RMLtest(Dataset):
    def __init__(self, samples, labels, SNR):
        self.samples = torch.tensor(samples,dtype=torch.float)
        self.SNR = SNR
        self.label = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        return self.samples[idx], self.SNR[idx], self.label[idx]




