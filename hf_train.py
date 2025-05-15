#https://github.com/Anonymous202401/If-Recollecting-were-Forgetting/blob/main/main_proposed.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
import time
import copy
import numpy as np
import models
import datasets

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os

from hf_utils.options import args_parser
from hf_utils.Approximator import getapproximator
from hf_utils.Approximator_resnet import  getapproximator_resnet
from hf_utils.perturbation import NoisedNetReturn
#from models.Update import  train
import shutil
import joblib

from r2d import compute_test_error

class DatasetSplit(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label, identity = self.dataset[item]
        return image, label, self.dataset.indices[item] #returns the actual data index in the underlying dataset


def train(step, args, net, Dataset2recollect, lr,info):

    # Ensure reproducibility of results, which may lead to a slight decrease in performance as it disables some optimizations.
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #torch.cuda.manual_seed_all(args.seed)
    #torch.cuda.manual_seed(args.seed)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

    
    dataloader = DataLoader(Dataset2recollect, batch_size=args.batch_size, shuffle=True)
    
    loss=0
    for batch_idx, (images, labels, indices) in enumerate(dataloader):
        optimizer.zero_grad()
        net.eval()
        # save sample idx in batch
        state_dict_cpu = {key: value.cpu() for key, value in net.state_dict().items()}
        info.append({"batch_idx_list": indices.tolist(), "model_list": state_dict_cpu})  
        images, labels = images.to(args.device), labels.to(args.device)
        net.zero_grad()
        log_probs = net(images)

        loss = loss_func(log_probs, labels)
        for param in net.parameters():
            loss += 0.5 * args.regularization * (param * param).sum()
        net.train()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=args.clip, norm_type=2)
        optimizer.step()
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        del images, labels, log_probs
        #print("     Step {:3d}     Batch {:3d}, Batch Size: {:3d}, Trainning Loss: {:.2f}".format(step,batch_idx,dataloader.batch_size,loss))
        step +=1

    return net.state_dict(), loss, lr,step



if __name__ == '__main__':
    print('starting')
###############################################################################
#                               SETUP                                         #
###############################################################################
    pycache_folder = "__pycache__"
    if os.path.exists(pycache_folder):
        shutil.rmtree(pycache_folder)
    # parse args
    args = args_parser()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #torch.cuda.manual_seed_all(args.seed)
    #torch.cuda.manual_seed(args.seed)

    # path="./data"
    # if not os.path.exists(path):
    #     os.makedirs(path)

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    loaders = datasets.get_loaders_large(args.dataset, batch_size=args.batch_size, seed=args.seed, root=args.dataroot, ood=False) #no unlearning
    
    train_loader = loaders['train_loader']
    valid_loader = loaders['valid_loader']
    
    dataset_train = train_loader.dataset
    dataset_valid = valid_loader.dataset

    net = models.get_model(args.model, num_classes=args.num_classes, filters_percentage=1.0).to(args.device)
    w = net.state_dict()


###############################################################################
#                               LEARNING                                      #
###############################################################################
    # training
    step=0
    
    Dataset2recollect=DatasetSplit(dataset_train)
    
    if not args.dont_train:
        print('training!')
        for e in range(args.epochs):
            info=[]
            torch.cuda.synchronize()
            t_start = time.time()
            w, loss,lr, step = train(step, args=args, net=net, Dataset2recollect=Dataset2recollect, lr=args.lr, info=info)
            torch.cuda.synchronize()
            t_end = time.time()   
            net.load_state_dict(w)
            net.eval()
            acc_t = 1 - compute_test_error(net.to(args.device), valid_loader, device=args.device)
            print(" Epoch {:3d}, valid accuracy: {:.2f},Time Elapsed:  {:.7f}s \n".format(e, acc_t, t_end - t_start))
            step += 1

            path1 = "./Checkpoint/model_{}_checkpoints". format(args.model)
            file_name = "check_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}_iter_{}.dat". format(
                args.dataset,args.epochs, args.lr,args.lr_decay,args.clip,args.seed, e)
            file_path = os.path.join(path1, file_name)
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))

            print('saving info')
            info = joblib.dump(info,file_path); rho=0  

            del info 


        rootpath = './log/Original/Model/'
        if not os.path.exists(rootpath):
            os.makedirs(rootpath)   
        torch.save(net.state_dict(),  rootpath+ 'Original_model_{}_data_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'
                   .format(args.model,args.dataset, args.epochs,args.lr,args.lr_decay,args.clip,args.seed))
        

    loaders = datasets.get_loaders_large(args.dataset, num_ids_forget = args.num_ids_forget, batch_size=args.batch_size, seed=args.seed, root=args.dataroot, ood=False)

    forget_set = loaders['train_forget_loader'].dataset
    all_indices_train = dataset_train.indices
    indices_to_unlearn = forget_set.indices
    remaining_indices = list(set(all_indices_train) - set(indices_to_unlearn))
###############################################################################
#                              PRECOMPUTATION UNLEARNING                      #
###############################################################################
    
    print('generating approximators')
    Approximators, rho = getapproximator_resnet(args,Dataset2recollect=train_loader.dataset,indices_to_unlearn=indices_to_unlearn)
    

    save_path = './log/Approximators_all_model_{}_data_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'.format(
        args.model,args.dataset,args.epochs, args.lr,args.lr_decay,args.clip,args.seed)
 
    print("saving approximators")
    torch.save({'Approximators': Approximators, 'rho': rho}, save_path)

    
###############################################################################
#                               UNLEARNINTG                                   #
###############################################################################
    Approximator_proposed = {j: torch.zeros_like(param) for j, param in enumerate(net.parameters())}
    torch.cuda.synchronize()
    for idx in indices_to_unlearn:
        for j in range(len(Approximator_proposed)):
            Approximator_proposed[j] += Approximators[idx][j]

###############################################################################
#                               SAVE                                          #
###############################################################################
    # save approximator
    rootpath2 = './log/Proposed/Approximator/'
    if not os.path.exists(rootpath2):
        os.makedirs(rootpath2)    
    torch.save(Approximator_proposed,  rootpath2+ 'Proposed_Approximator_model_{}_data_{}_remove_{}_epoch_{}_lr_{}_lrdecay_{}_clip_{}_seed{}.pth'.format(
                        args.model, args.dataset, args.num_ids_forget, args.epochs, args.lr,args.lr_decay,args.clip,args.seed))

    print('all done!')
    
   
