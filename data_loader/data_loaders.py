import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class KKstreamDataset(Dataset):
    """Dataset wrapping features and target labels for KKstream - KKStream Deep Learning Workshop.

    Arguments:
        A file path
    """

    def __init__(self, file_path, training=True):
    
        if training==True:
            self.dataset = np.load(file_path)
            self.X_train = self.dataset['train_eigens']
            self.y_train = self.dataset['train_labels']
        else:
            self.dataset = np.load(file_path)
            self.X_train = self.dataset['test_eigens']
            self.y_train = self.dataset['test_labels']
        
        # NOTE: a 896d feature vector for each user, the 28d vector in the end are
        #       labels
        #       896 = 32 (weeks) x 7 (days a week) x 4 (segments a day)
        #self.X_train = self.dataset[:, :-28].reshape(-1, 896)
        #self.y_train = self.dataset[:, -28:]
        
        #print(self.X_train.shape)

    def __getitem__(self, index):
        feature = torch.from_numpy(self.X_train[index]).float()
        label = torch.from_numpy(self.y_train[index]).float()
        return feature, label

    def __len__(self):
        return len(self.X_train)

class KKstreamDataLoader(BaseDataLoader):
    """
    KKstream data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = KKstreamDataset(self.data_dir, training=training)
        super(KKstreamDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
