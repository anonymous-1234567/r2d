#!/usr/bin/env python3
import argparse
import os
import time
import copy
import random
import pickle

import numpy as np

import torch
import torch.optim as optim

from tqdm import tqdm
import models
import datasets
from utils import *

def get_Lipschitz(model1, grad_vector1, model2, grad_vector2):
    '''
    Estimates Lipschitz constant based on model1, model2
    '''
    params1 = []
    params2 = []

    for param in model1.parameters():
        params1.append(param.view(-1))
            
    for param in model2.parameters():
        params2.append(param.view(-1))

    param1_vector = torch.cat(params1)
    param2_vector = torch.cat(params2)

 
    numer = torch.norm(grad_vector1 - grad_vector2, p=2).item()   
    denom = torch.norm(param1_vector - param2_vector, p=2).item()
    L = numer/denom
    return L 

def compute_l2_norm(model): #computes l2 norm of gradients
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:  # Check if gradient exists
            param_norm = param.grad.data.norm(2)  # L2 norm of gradients
            total_norm += param_norm.item() ** 2  # Sum of squared norms

    total_norm = total_norm ** 0.5  # Square root of sum of squared norms (Euclidean norm)
    return(total_norm)

def copy_model_with_grads(model):
    new_model = copy.deepcopy(model)

    for (new_param, param) in zip(new_model.parameters(), model.parameters()):
        if param.grad is not None:
            new_param.grad = param.grad.clone()

    return new_model


def add_gaussian_noise_to_weights(args, model, sigma):
    # Iterate over all model parameters
    for param in model.parameters():
        # Check if the parameter has a gradient (usually it will if it's a learnable weight)
        if param.requires_grad:
            # Create Gaussian noise with mean 0 and standard deviation sigma
            noise = torch.normal(mean=0.0, std=sigma, size=param.size()).to(args.device)
            # Add the noise to the parameter
            param.data += noise


def compute_full_gradient(args, model, data_loader, criterion):
    model.eval()
    #model.train()
    model.zero_grad()
    
    total_gradient = None

    for batch_idx, (data, target, identity) in enumerate(data_loader):

        model.zero_grad(set_to_none=True)
        data, target = data.to(args.device), target.to(args.device)
                
        output = model(data)
        loss = criterion(output, target) 
        loss.backward()

        gradients = []
        params = []

        for param in model.parameters():
            if param.grad is not None:  # Check if gradient exists
                gradients.append(param.grad.view(-1))  # Flatten the gradient and add to list

        # Concatenate all gradients into a single vector
        grad_vector = torch.cat(gradients) 
        

        if total_gradient is None:
            total_gradient = grad_vector * args.batch_size
        else:
            total_gradient += grad_vector * args.batch_size

    total_gradient = total_gradient/len(data_loader.dataset)

    #print(f"gradnorm: {torch.linalg.vector_norm(total_gradient)}")
    #print(f"one batch gradnorm: {compute_l2_norm(model)}")

    return(total_gradient)

def estimate_Lipschitz(model, Nsamples = 100):
    L_list = []
    model1 = copy.deepcopy(model)
        
    grad_vector = compute_full_gradient(args, model, train_loader, criterion)

    Nsamples = 100 #number of samples for lipschitz estimate
    for i in tqdm(range(Nsamples)):
        model1 = copy.deepcopy(model)
        add_gaussian_noise_to_weights(args, model1, 0.01)
        grad_vector1 = compute_full_gradient(args, model1, train_loader, criterion)
            
        lip = get_Lipschitz(model1, grad_vector1, model, grad_vector)
        L_list.append(lip)

    del model1
    return(max(L_list))



    
def run_epoch(args, model, data_loader, criterion=torch.nn.CrossEntropyLoss(), optimizer=None, epoch=0, mode='train'):
    
    if mode == 'train':
        model.train()
    else:
        model.eval()
    
    
    metrics = AverageMeter() #reset after each epoch

    with torch.set_grad_enabled(mode == 'train'):

        for batch_idx, (data, target, identity) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)

                
            output = model(data)
            loss = criterion(output, target) 


            if not args.quiet:
                metrics.update(n=data.size(0), loss=loss.item(), error=get_error(output, target))
            
            if mode == 'train':

                optimizer.zero_grad() 
                model.zero_grad(set_to_none=True) #double assurance?
                loss.backward()
                if not args.no_gradient_estimation:
                    grad_norm = compute_l2_norm(model)
                    metrics.update(n=data.size(0), grad_norm=grad_norm)

                optimizer.step()
                

    log_metrics(mode, metrics, epoch)
    
    if mode == 'train':
        print('Learning Rate : {}'.format(optimizer.param_groups[0]['lr']))
    return metrics


    
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None)
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='something about suppressing logs')

    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--dataset', default='lacuna100binary128')
    parser.add_argument('--dataroot', type=str, default='data/lacuna100binary128')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--filters', type=float, default=1.0,
                        help='Percentage of filters')
    parser.add_argument('--num-ids-forget', type=int, default=None,
                        help='Number of IDs to forget')
    parser.add_argument('--lossfn', type=str, default='ce',
                        help='Cross Entropy: ce or mse')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--scheduler', type=float, default=0.9,
                        help='exponential scheduler')
    parser.add_argument('--model', default='resnetsmooth')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of Classes')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-checkpoints', action='store_true', default=False,
                        help='save checkpoints')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='plot training and validation loss')
    parser.add_argument('--model-selection', action='store_true', default=False,
                        help='store and log model with best validation error')
    
    parser.add_argument('--no-gradient-estimation', action='store_true', default=False, help='disables gradient norm estimation')

    parser.add_argument('--compute-lipschitz', action='store_true', default=False, help='estimate Lipschitz constant')

    args = parser.parse_args()
   
    manual_seed(args.seed)
        
    if args.name is None:
        args.name = f"{args.dataset}_{args.model}_{str(args.filters).replace('.','_')}"
        args.name += f"_forget_{args.num_ids_forget}"
        args.name+=f"_lr_{str(args.lr).replace('.','_')}"
        args.name+=f"_bs_{str(args.batch_size)}"
        args.name+=f"_ls_{args.lossfn}"
        args.name+=f"_seed_{str(args.seed)}"
    if args.scheduler is not None:
        args.name+=f"_scheduler_{str(args.scheduler).replace('.','_')}"

    print(f'Checkpoint name: {args.name}')
    
    os.makedirs('logs', exist_ok=True)

    #DEVICE MANAGEMENT
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    assert use_cuda

    os.makedirs('checkpoints', exist_ok=True)

    ood = True
    if args.dataset == 'eicu':
        ood=False

    #LOAD DATA!
    loaders = datasets.get_loaders_large(args.dataset, num_ids_forget = args.num_ids_forget, batch_size=args.batch_size, seed=args.seed, root=args.dataroot, augment=False, ood=ood, test=False)
    train_loader = loaders['train_loader']
    valid_loader = loaders['valid_loader']
    
    num_classes = max(train_loader.dataset.targets) + 1 if args.num_classes is None else args.num_classes
    args.num_classes = num_classes
    print(f"Number of Classes: {num_classes}")

    #GET MODEL
    model = models.get_model(args.model, num_classes=num_classes, filters_percentage=args.filters).to(args.device)
    
    if args.resume is not None:
        state = torch.load(args.resume,weights_only=True)
        model.load_state_dict(state)
        print(f"Loading state from: {args.resume}")
        args.name += f"_loadedfrom{args.resume.split('.')[0].split('_')[-1]}"
    
    torch.save(model.state_dict(), f"checkpoints/{args.name}_init.pt")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0, weight_decay=0)
    
    if args.scheduler is not None:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler)

    criterion = torch.nn.CrossEntropyLoss().to(args.device) if args.lossfn=='ce' else torch.nn.MSELoss().to(args.device)

    train_time = 0

    plot_loss = [] #training loss
    plot_valid_loss = []

    plot_train_error = []
    plot_valid_error = []
    
    plot_grad_norm = []
    plot_lr = []

    args_dict = vars(args)
    logs = dict()
    for key, value in args_dict.items():
        logs[key] = value

    valid_loss = 1000 #for model selection 

    for epoch in range(args.epochs):

        model1 = copy.deepcopy(model)
        t1 = time.time()

        if args.scheduler is not None:
            plot_lr.append(scheduler.get_last_lr())

        metrics = run_epoch(args, model, train_loader, criterion, optimizer, epoch, mode='train')
        valid_metrics = run_epoch(args, model, valid_loader, criterion, optimizer, epoch, mode='valid')

        if args.scheduler is not None:
            scheduler.step()

        if args.model_selection:
            if valid_metrics.avg['error'] < valid_loss:
                valid_epoch = epoch
                valid_name = f"checkpoints/{args.name}_selected.pt"
                torch.save(model.state_dict(), valid_name)
                valid_loss = valid_metrics.avg['error'] 
        
        plot_loss.append(metrics.avg['loss'])
        plot_train_error.append(metrics.avg['error'])
        plot_valid_loss.append(valid_metrics.avg['loss'])
        plot_valid_error.append(valid_metrics.avg['error'])
        if not args.no_gradient_estimation:
            plot_grad_norm.append(metrics.max['grad_norm'])

        t2 = time.time()
        train_time += np.round(t2-t1,2)
        
        if epoch % 5 == 0 and args.save_checkpoints:
            torch.save(model.state_dict(), f"checkpoints/{args.name}_{epoch}.pt")
        

        print(f'Epoch Time: {np.round(time.time()-t1,2)} sec')


    print (f'Pure training time: {train_time} sec')


    torch.save(model.state_dict(), f"checkpoints/{args.name}_{epoch}_final.pt") 


    final_it = -1

    if args.model_selection:
        state = torch.load(valid_name, weights_only=True)
        model.load_state_dict(state)
        print(f"Model selection: Epoch {valid_epoch} with valid error {valid_loss}")
        final_it = valid_epoch
        logs['selected epoch'] = final_it


    #log information for plotting
    logs['train loss over epochs'] = plot_loss
    logs['train error over epochs'] = plot_train_error
    logs['final train loss'] = plot_loss[final_it]
    logs['final train error'] = plot_train_error[final_it]

    logs['valid loss over epochs'] = plot_valid_loss
    logs['valid error over epochs'] = plot_valid_error
    logs['final valid loss'] = plot_valid_loss[final_it]
    logs['final valid error'] = plot_valid_error[final_it]

    if not args.no_gradient_estimation:
        logs['grad norm over epochs'] = plot_grad_norm
    
    logs['lr over epochs'] = plot_lr

    
    if args.dataset != "eicu":
        print("Testing on OOD data")
        ood_loader = loaders['ood_loader']
        ood_metrics = run_epoch(args, model, ood_loader, criterion, optimizer, epoch, mode='ood')
        logs['final ood loss'] = ood_metrics.avg['loss']
        logs['final ood error'] = ood_metrics.avg['error']

    if args.num_ids_forget is not None:
        train_forget_loader = loaders['train_forget_loader']
        train_forget_metrics = run_epoch(args, model, train_forget_loader, criterion, optimizer, epoch, mode='train forget')
        logs['final train forget loss'] = train_forget_metrics.avg['loss']
        logs['final train forget error'] = train_forget_metrics.avg['error']


    if args.compute_lipschitz:
        print("Computing Lipschitz constant...")
        L_list = []

        model1 = copy.deepcopy(model)
        
        grad_vector = compute_full_gradient(args, model, train_loader, criterion)

        Nsamples = 400 #number of samples for lipschitz estimate
        for i in tqdm(range(Nsamples)):
            model1 = copy.deepcopy(model)
            add_gaussian_noise_to_weights(args, model1, 0.01)
            grad_vector1 = compute_full_gradient(args, model1, train_loader, criterion)
            
            lip = get_Lipschitz(model1, grad_vector1, model, grad_vector)
            L_list.append(lip)

        del model1
        print(f"Lipschitz constant: {max(L_list)}")
        logs['Lipschitz'] = max(L_list)

    with open("logs/"+ args.name + '.pkl', 'wb') as f:
        pickle.dump(logs, f)


    #plotting! 

    if args.plot:
        import matplotlib.pyplot as plt

        # plotting loss over time 
        plt.figure()
        plt.plot(plot_loss, label="Loss")
        plt.plot(plot_valid_loss, label="Validation Loss")
        plt.plot(plot_train_error, label="Training Error")
        plt.plot(plot_valid_error, label="Validation Error")
        plt.legend()
        plt.xlabel("Epoch")
        plt.title(args.name)
        plt.savefig('plots/' + args.name + '.png')

        #plotting gradnorm over time
        if not args.no_gradient_estimation:
            plt.figure()
            
            plt.plot(plot_grad_norm, label="Gradient Norm")
            plt.legend()
            plt.xlabel("Epoch")
            plt.title(args.name)
            plt.savefig('plots/' + args.name + 'gradientnorm.png')

        print("plots saved successfully!")
