import os
import copy
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from lacuna import Lacuna100Large
from sklearn.preprocessing import StandardScaler

from eicu import eICU

def manual_seed(seed):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

_DATASETS = {}

def _add_dataset(dataset_fn):
    _DATASETS[dataset_fn.__name__] = dataset_fn
    return dataset_fn

def _get_eicu_transforms(augment=False, train=True):
    return None

def _get_lacuna_transforms(augment=False):
    transform_augment = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.Resize(size=(32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.382, 0.420, 0.502), (0.276, 0.279, 0.302)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.382, 0.420, 0.502), (0.276, 0.279, 0.302)),
    ])

    if augment:
        return transform_augment
    else:
        return transform_test

@_add_dataset #this is what i added
def lacuna100binary128(root, augment=False):
    transform = _get_lacuna_transforms(augment=augment)
    dataset = Lacuna100Large(root=root, transform=transform)
    return dataset

@_add_dataset #this is what i added
def eicu(root, augment=False, transform=None):
    dataset = eICU(root=root)
    return dataset


def get_loader_simple(dataset_name, root, seed: int = 1, batch_size = 128, shuffle = True, **dataset_kwargs):
    manual_seed(seed)
    dataset = _DATASETS[dataset_name](root, **dataset_kwargs)
    dataset.targets = np.array(dataset.targets)

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                               worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    return(data_loader)


def remove_ids(dataset, ids, num_forget=None):
    final_indices = []
    for j in range(len(dataset.indices)):
        if dataset.identities[dataset.indices[j]] in ids: 
            pass
        else:
            final_indices.append(dataset.indices[j])
    dataset.indices = np.array(final_indices)


def keep_ids(dataset, ids, num_forget=None):
    final_indices = []
    for j in range(len(dataset.indices)):
        if dataset.identities[dataset.indices[j]] in ids: 
            final_indices.append(dataset.indices[j])
        else:
            pass
    dataset.indices = np.array(final_indices)

def get_loaders_large(dataset_name, num_ids_forget: int = None, forget_ids = None, seed: int = 1, root: str = None, batch_size=128, shuffle=True, ood=True, test=False, **dataset_kwargs):
    '''
    num_ids_forget: number of ids to forget 
    num_forget : number of each id to forget (currently doesn't work or do anything)
    forget_ids: specific ids to forget, if None then they are randomly chosen
    test: whether or not to get test loaders
    '''

    forget = False
    if (num_ids_forget is not None) or (forget_ids is not None):
        forget = True

    manual_seed(seed)
    if root is None:
        root = os.path.expanduser('~/data')

    train_root = root + '/in_distribution/train'
    test_root = root + '/in_distribution/test'

    train_set = _DATASETS[dataset_name](train_root, **dataset_kwargs)
    test_set = _DATASETS[dataset_name](test_root, **dataset_kwargs)

    if ood:
        ood_root = root + '/oo_distribution'
        ood_set = _DATASETS[dataset_name](ood_root, **dataset_kwargs)

    
    #train validation split
    valid_set = copy.deepcopy(train_set)
    

    rng = np.random.RandomState(seed)
    valid_idx=[]
    
    Nclasses = max(train_set.targets) + 1
    for i in range(Nclasses): # iterating through both classes (0 and 1)
        class_idx = np.where(train_set.targets==i)[0] #since np.where returns a tuple, the first element contains the actual indices
        valid_idx.append(rng.choice(class_idx,int(0.2*len(class_idx)),replace=False)) #makes sure valid set is balanced
    valid_idx = np.hstack(valid_idx)    
    train_idx = list(set(range(len(train_set)))-set(valid_idx))
    train_set.indices = train_idx
    valid_set.indices = valid_idx

    
    dset_list = [valid_set, test_set]
    #forget data
    if forget:
        forget_set = copy.deepcopy(train_set) 
        forget_set.reset() #forgotten samples from both training and validation

        train_forget_set = copy.deepcopy(train_set)

        all_ids = set(train_set.identities)
        
        if forget_ids is None:
            forget_ids = []
            
            if dataset_name != 'eicu':
                #yes this code only works for binary classification
                nclass0 = int(num_ids_forget/2.0)
                nclass1 = num_ids_forget - nclass0
                
                class0_idx = np.where(train_set.targets==0)[0] #since np.where returns a tuple, the first element contains the actual indices
                class0_idx = rng.choice(class0_idx, nclass0, replace=False) #makes sure forget set is balanced
                class0_identities = [train_set.identities[class0_i] for class0_i in class0_idx]

                class1_idx = np.where(train_set.targets==1)[0] #since np.where returns a tuple, the first element contains the actual indices
                class1_idx = rng.choice(class1_idx, nclass1, replace=False) #makes sure forget set is balanced
                class1_identities = [train_set.identities[class1_i] for class1_i in class1_idx]

                forget_ids = class0_identities + class1_identities
            else:
                forget_ids = rng.choice(sorted(list(all_ids)), num_ids_forget, replace=False) #sorted necessary to make sure seed makes consistent results
        
        retain_ids = list(all_ids - set(forget_ids))
        
        remove_ids(train_set, forget_ids)
        remove_ids(train_forget_set, retain_ids)
        dset_list.append(train_forget_set)

        remove_ids(forget_set, retain_ids)
        dset_list.append(forget_set)

        remove_ids(valid_set, forget_ids)

        if test:
            test_forget_set = copy.deepcopy(test_set)
            test_retain_set = copy.deepcopy(test_set)
            keep_ids(test_forget_set, forget_ids) #test forget set is only the forget ids
            remove_ids(test_retain_set, forget_ids) #test retain set does not have the forget ids (may have ids that are not in retain_ids if it is not lacuna data)
            dset_list.append(test_forget_set)
            dset_list.append(test_retain_set)

    

        print(f"Forgetting these IDs: {forget_ids}")

        #transform data based on train dataset
    if dataset_name=='eicu':
        train_set.transform()
        for dset in dset_list:
            dset.transform(train_set.transform)



    loader_args = {'num_workers': 0, 'pin_memory': False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    print(f"Number of training samples: {len(train_set)}")
    print(f"Number of validation samples: {len(valid_set)}")
    print(f"Number of test samples: {len(test_set)}")

    
    loaders = dict()


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                                               worker_init_fn=_init_fn if seed is not None else None, **loader_args) #shuffle
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                              worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle,
                                              worker_init_fn=_init_fn if seed is not None else None, **loader_args) #shuffle


    loaders['train_loader'] = train_loader
    loaders['valid_loader'] = valid_loader
    loaders['test_loader'] = test_loader

    if ood:
        print(f"Number of ood samples: {len(ood_set)}")
        ood_loader = torch.utils.data.DataLoader(ood_set, batch_size=batch_size, shuffle=False,
                                              worker_init_fn=_init_fn if seed is not None else None, **loader_args)
        loaders['ood_loader'] = ood_loader

    
    
    if forget:
        train_forget_loader = torch.utils.data.DataLoader(train_forget_set, batch_size=batch_size, shuffle=shuffle,
                                              worker_init_fn=_init_fn if seed is not None else None, **loader_args) #shuffle
        loaders['train_forget_loader'] = train_forget_loader

        forget_loader = torch.utils.data.DataLoader(forget_set, batch_size=batch_size, shuffle=shuffle,
                                              worker_init_fn=_init_fn if seed is not None else None, **loader_args) #shuffle
        loaders['forget_loader'] = forget_loader

        print(f"Number of train forget samples: {len(train_forget_set)}")
    
        if test:
            print(f"Number of test forget samples: {len(test_forget_set)}")
            print(f"Number of test retain samples: {len(test_retain_set)}")
            test_forget_loader = torch.utils.data.DataLoader(test_forget_set, batch_size=batch_size, shuffle=False,
                                                worker_init_fn=_init_fn if seed is not None else None, **loader_args)
            test_retain_loader = torch.utils.data.DataLoader(test_retain_set, batch_size=batch_size, shuffle=shuffle,
                                                worker_init_fn=_init_fn if seed is not None else None, **loader_args) #shuffle
            loaders['test_forget_loader'] = test_forget_loader
            loaders['test_retain_loader'] = test_retain_loader
    
    return loaders
        
