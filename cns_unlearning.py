#https://github.com/zhangbinchi/certified-deep-unlearning/blob/main/unlearn.py

import time
import random
import numpy as np
import torch
import torch.nn as nn
#import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.autograd import grad
import argparse
import os
from tqdm import tqdm

import models
import datasets
from utils import *

from r2d import calibrateAnalyticGaussianMechanism


def params_to_vec(parameters, grad=False):
    vec = []
    for param in parameters:
        if grad:
            vec.append(param.grad.view(1, -1))
        else:
            vec.append(param.data.view(1, -1))
    return torch.cat(vec, dim=1).squeeze()


def vec_to_params(vec, parameters):
    param = []
    for p in parameters:
        size = p.view(1, -1).size(1)
        param.append(vec[:size].view(p.size()))
        vec = vec[size:]
    return param


def batch_grads_to_vec(parameters):
    vec = []
    for param in parameters:
        # vec.append(param.view(1, -1))
        vec.append(param.reshape(1, -1))
    return torch.cat(vec, dim=1).squeeze()


def batch_vec_to_grads(vec, parameters):
    grads = []
    for param in parameters:
        size = param.view(1, -1).size(1)
        grads.append(vec[:size].view(param.size()))
        vec = vec[size:]
    return grads


def grad_batch(batch_loader, lam, model, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    params = [p for p in model.parameters() if p.requires_grad]
    grad_batch = [torch.zeros_like(p).cpu() for p in params]
    num = 0
    for batch_idx, (data, targets, identities) in enumerate(batch_loader):
        num += targets.shape[0]
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)

        grad_mini = list(grad(criterion(outputs, targets), params))
        for i in range(len(grad_batch)):
            grad_batch[i] += grad_mini[i].cpu().detach()

    for i in range(len(grad_batch)):
        grad_batch[i] /= num

    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2)
    grad_reg = list(grad(lam * l2_reg, params))
    for i in range(len(grad_batch)):
        grad_batch[i] += grad_reg[i].cpu().detach()
    return [p.to(device) for p in grad_batch]


def grad_batch_approx(batch_loader, lam, model, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss = 0
    for batch_idx, (data, targets, identities) in enumerate(batch_loader):
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        loss += criterion(outputs, targets)

    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2)

    params = [p for p in model.parameters() if p.requires_grad]
    return list(grad(loss + lam * l2_reg, params))


def hvp(y, w, v):
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads


def inverse_hvp(y, w, v):

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)
    vec_first_grads = batch_grads_to_vec(first_grads)
    
    hessian_list = []
    for i in range(vec_first_grads.shape[0]):
        sec_grads = grad(vec_first_grads[i], w, retain_graph=True)
        hessian_list.append(batch_grads_to_vec(sec_grads).unsqueeze(0))
    
    hessian_mat = torch.cat(hessian_list, 0)
    return torch.linalg.solve(hessian_mat, v.view(-1, 1))

def compute_noise_cns(eps, delta, L,  G, d, C = 21, rho=0.1, M=1, lam = 1000, lam_min = 0):
    '''
    M: Lipschitz constant of the Hessian
    L: Lipschitz constant of gradient
    rho: probability that the bound does not hold  
    '''
    Delta = (2 * C * (M * C + lam) + G)/(lam + lam_min) + ((16 * np.sqrt(d/rho))/(lam + lam_min) + 1/16.0) * (2 * L * C + G)
    if eps <= 1:
        sigma = Delta * np.sqrt(2 * np.log(1.25/delta))/eps
    else:
        sigma = calibrateAnalyticGaussianMechanism(eps, delta, Delta, tol=1e-12)
    
    return sigma


def newton_update(g, batch_size, res_set, lam, gamma, model, s1, s2, scale, device): #gamma is the convex approximation coefficient, lam is the weight decay coefficeint
    model.eval()
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    H_res = [torch.zeros_like(p) for p in g]
    for i in tqdm(range(s1)):
        H = [p.clone() for p in g]
        sampler = RandomSampler(res_set, replacement=True, num_samples=batch_size * s2)
        # Create a data loader with the sampler
        res_loader = DataLoader(res_set, batch_size=batch_size, sampler=sampler)
        res_iter = iter(res_loader)
        for j in range(s2):
            data, target, identities = next(res_iter)
            data, target = data.to(device), target.to(device)
            z = model(data)
            loss = criterion(z, target)
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)
            # Add L2 regularization to the loss
            loss += (lam + gamma) * l2_reg
            H_s = hvp(loss, params, H)
            with torch.no_grad():
                for k in range(len(params)):
                    H[k] = H[k] + g[k] - H_s[k] / scale
                #if j % int(s2 / 10) == 0:
                    #print(f'Epoch: {j}, Sum: {sum([torch.norm(p, 2).item() for p in H])}')
        for k in range(len(params)):
            H_res[k] = H_res[k] + H[k] / scale
        
    return [p / s1 for p in H_res]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--dataset')
    parser.add_argument('--dataroot', type=str, default='data/')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--filters', type=float, default=1.0,
                        help='Percentage of filters')
    parser.add_argument('--model', default='resnetsmooth')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of Classes')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--unlearn-bs', type=int, default=10, metavar='N',
                        help='input batch size for unlearning (default: 10)')
    parser.add_argument('--num-ids-forget', type=int, default=2,
                        help='Number of IDs to forget')
    parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
    
    parser.add_argument('--C', type=float, default=21.0,
                        help='Norm constraint of parameters')
    parser.add_argument('--s1', type=int, default=10, help='Number of samples in Hessian approximation')
    parser.add_argument('--s2', type=int, default=1000, help='The order number of Taylor expansion in Hessian approximation')
    parser.add_argument('--std', type=float, default=0, help='The standard deviation of Gaussian noise')
    parser.add_argument('--gamma', type=float, default=200, help='The convex approximation coefficient')
    parser.add_argument('--scale', type=float, default=50000., help='The scale of Hessian')
    parser.add_argument('--gpu', type=int, default=0, metavar='S',
                        help='gpu')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:" + str(args.gpu) if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    # Set the random seed for CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    if not os.path.exists('./model'):
        os.mkdir('./model')
    
    PATH_load = './model/cns_'+f"clip_{str(args.C).replace('.','_')}_"+args.model+'_'+args.dataset+f"_lr_{str(args.lr).replace('.','_')}"+f"_bs_{str(args.batch_size)}"+f"_wd_{str(args.weight_decay).replace('.','_')}"+f"_epoch_{str(args.epochs)}"+f"_seed_{str(args.seed)}"+'.pth'
    PATH_save = './model/cns_unlearn_'+f"clip_{str(args.C).replace('.','_')}_"+args.model+'_'+args.dataset+f"_lr_{str(args.lr).replace('.','_')}"+f"_bs_{str(args.batch_size)}"+f"_wd_{str(args.weight_decay).replace('.','_')}"+f"_epoch_{str(args.epochs)}"+f"_seed_{str(args.seed)}"+f"_num_{str(args.num_ids_forget)}"+f"_unlearnbs_{str(args.unlearn_bs)}"+f"_s1_{str(args.s1)}"+f"_s2_{str(args.s2)}"+f"_std_{str(args.std).replace('.','_')}"+f"_gamma_{str(args.gamma).replace('.','_')}"+f"_scale_{str(args.scale).replace('.','_')}"+'.pth'

    loaders = datasets.get_loaders_large(args.dataset, num_ids_forget = args.num_ids_forget,
                                                        batch_size=args.batch_size, seed=args.seed, root=args.dataroot, ood=False)


    num_classes = 2
    args.num_classes = num_classes

    model = models.get_model(args.model, num_classes=num_classes, filters_percentage=args.filters).to(args.device)

    model.load_state_dict(torch.load(PATH_load, weights_only=True))

    

    res_loader = loaders['train_loader']

    start = time.time()
    
    g = grad_batch(res_loader, args.weight_decay, model, args.device)

    delta = newton_update(g, args.unlearn_bs, res_loader.dataset, args.weight_decay, args.gamma, model, args.s1, args.s2, args.scale, args.device)
    for i, param in enumerate(model.parameters()):
        param.data.add_(-delta[i] + args.std * torch.randn(param.data.size()).to(args.device))
    print(f'Time: {time.time() - start}')
    torch.save(model.state_dict(), PATH_save)
    