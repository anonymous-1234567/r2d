import numpy as np
import os
from PIL import Image
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch

def convert_to_tensor(a):
    return(torch.tensor(a.to_numpy().astype(np.float64), dtype=torch.float32))

def convert_to_tensorlong(a):
    return(torch.tensor(a.to_numpy().astype(np.float64), dtype=torch.long))


class eICU(TensorDataset):

    def __init__(self, root):
        

        self.root = root
        self.targets = convert_to_tensorlong(pd.read_csv(os.path.join(self.root, 'labels.csv'))).squeeze(1)
        self.identities = np.array(pd.read_csv(os.path.join(self.root,  'identities.csv')).iloc[:, 0].to_list())
        self.indices=np.arange(len(self.targets))

        self.numerical_columns = [0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

        self.data = pd.read_csv(os.path.join(self.root, 'data.csv'))
        self.data = convert_to_tensor(self.data)

        super(eICU, self).__init__(self.data, self.targets)

        # The data is saved in BGR format. Convert to RGB.
        #self.data = self.data[...,::-1]

    def get_data_by_index(self, index): #for HF method, which handles the data loading in a completely different way
        img = self.data[index, :]
        target =  self.targets[index]
        identity = self.identities[index]
        return img, target, identity


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        index = self.indices[i]
        img = self.data[index, :]
        target =  self.targets[index]
        identity = self.identities[index]

        return img, target, identity



    def transform(self, transform=None):
        if transform is None:
            scaler = StandardScaler()
            scaler.fit_transform(self.data[self.indices][:, self.numerical_columns])

            self.data = self.data.numpy()
            self.data[:, self.numerical_columns] = scaler.transform(self.data[:, self.numerical_columns])
            
            self.data = torch.tensor(self.data)

            self.transform = scaler

        else:
            self.data = self.data.numpy()
            self.data[:, self.numerical_columns] = transform.transform(self.data[:, self.numerical_columns])

            self.data = torch.tensor(self.data)

    def reset(self):
        self.indices=np.arange(len(self.targets))