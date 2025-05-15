import numpy as np
import os
from PIL import Image
from torchvision.datasets import VisionDataset


class Lacuna100Large(VisionDataset):

    def __init__(self, root, transform=None):
        super(Lacuna100Large, self).__init__(root, transform=transform)

        self.targets = np.load(os.path.join(self.root, 'label.npy')) #avoid accessing this 
        self.identities = np.load(os.path.join(self.root,  'identities.npy')) #avoid accessing this
        self.indices=np.arange(len(self.targets)) #controls the data we actually have access to, indices of individual data samples

        # The data is saved in BGR format. Convert to RGB.
        #self.data = self.data[...,::-1]

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
        img = np.load(os.path.join(self.root, f'image{index}.npy'))

        target =  self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = Image.fromarray(img) #PIL image
        if self.transform is not None:
            img = self.transform(img)

            
        identity = self.identities[index]          

        return img, target, identity
    
    def reset(self):
        self.indices = np.arange(len(self.targets))
        
    def get_data_by_index(self, index):
        img = np.load(os.path.join(self.root, f'image{index}.npy'))

        target =  self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = Image.fromarray(img) #PIL image
        if self.transform is not None:
            img = self.transform(img)

            
        identity = self.identities[index]          

        return img, target, identity