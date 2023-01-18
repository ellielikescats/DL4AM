import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchmetrics
import torchvision
from sklearn.model_selection import train_test_split

class SMD_Dataset(Dataset):
    def _init_(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.len = x_data.shape[0]

    def _getitem_(self, index):
        return self.x_data[index], self.y_data[index]

    def _len_(self):
        return self.len

def data_loader(data_X, data_y):
    data = SMD_Dataset(data_X, data_y)
    size = data.len
    loader = DataLoader(dataset=data,
                    batch_size=64,
                    shuffle=False,
                    num_workers=1)
    return loader

class DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.x_signals = torch.load("x_signals_data.pt")
        self.y_signals = torch.load('y_signals_data.pt')


    # This could work - the numpy array might not be taken to the model and might need to convert to tensor (torch.FloatTensor)
    def setup(self, stage=None):
        # download and prepare data from pickle?
        self.train_data_x, self.test_data_x, self.train_data_, self.test_data_ = train_test_split(
        self.x_signals, self.y_signals, shuffle=False, test_size = 0.2, random_state = 42)

    def train_dataloader(self):
        return data_loader(torch.from_numpy(self.train_data_x), torch.from_numpy(self.train_data_y))

    def test_dataloader(self):
        return data_loader(torch.from_numpy(self.test_data_x), torch.from_numpy(self.test_data_y))




