from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold
import random
import torch
import torch.nn as nn
import numpy as np


def evaluate_attack_model(sample_loss, members, n_splits = 5, random_state = None, scoring = 'roc_auc', n_repeats = 100):
  """Computes the cross-validation score of a membership inference attack.
  Args:
    sample_loss : array_like of shape (n,).
      objective function evaluated on n samples.
    members : array_like of shape (n,),
      whether a sample was used for training.
    n_splits: int
      number of splits to use in the cross-validation.
    random_state: int, RandomState instance or None, default=None
      random state to use in cross-validation splitting.
  Returns:
    score : array_like of size (n_splits,)
  """

  unique_members = np.unique(members)
  if not np.all(unique_members == np.array([0, 1])):
    raise ValueError("members should only have 0 and 1s")

  attack_model = LogisticRegression()
  cv = RepeatedStratifiedKFold(n_splits=n_splits, random_state=random_state, n_repeats=n_repeats) #automatically shuffles
  scores = cross_val_score(attack_model, sample_loss, members, cv=cv, scoring=scoring)
  return scores

def membership_inference_attack(model, t_loader, f_loader, device, n_splits = 5, seed=None, metric=None, normalize=None, n_repeats = 100, balance=True, scoring='roc_auc'):
  #adapted from SCRUB code to include both logits and loss in logistic regression model
#t_loader is test set, f_loader is forget set 

    if seed is not None:
      torch.manual_seed(seed)
      np.random.seed(seed)
    cr = nn.CrossEntropyLoss(reduction='none') #reduction none means we compute loss on each sample 
    test_losses = []
    forget_losses = []
    test_outputs = []
    forget_outputs = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target, _) in enumerate(t_loader):
            data, target = data.to(device), target.to(device)            
            output = model(data)
            loss = cr(output, target)
            output = torch.cat([output, loss.unsqueeze(1)], dim=1)
            #test_losses = test_losses + list(loss.cpu().detach().numpy())
            if metric == 'loss':
              test_outputs.append(loss)
            else:
              test_outputs.append(output)
        
        test_losses = torch.cat(test_outputs, dim=0)

        for batch_idx, (data, target, _) in enumerate(f_loader):
            data, target = data.to(device), target.to(device)            
            output = model(data)
            loss = cr(output, target)
            
            output = torch.cat([output, loss.unsqueeze(1)], dim=1)
            #forget_losses = forget_losses + list(loss.cpu().detach().numpy())
            if metric == 'loss':
              forget_outputs.append(loss)
            else:
              forget_outputs.append(output)
        
        forget_losses = torch.cat(forget_outputs, dim=0)

        if balance:
          if len(forget_losses) > len(test_losses):
              #forget_losses = list(random.sample(forget_losses, len(test_losses)))
              indices = torch.randperm(forget_losses.size(0))[:len(test_losses)]
              #indices = np.random.permutation(forget_losses.size(0))[:len(test_losses)]
              # Select the corresponding data points
              forget_losses = forget_losses[indices]

          elif len(test_losses) > len(forget_losses):
              #test_losses = list(random.sample(test_losses, len(forget_losses)))
              indices = torch.randperm(test_losses.size(0))[:len(forget_losses)]
              #indices = np.random.permutation(test_losses.size(0))[:len(forget_losses)]
              # Select the corresponding data points
              test_losses = test_losses[indices]
        

        
        features = torch.cat([test_losses, forget_losses], dim=0).cpu().numpy()

        #normalize
        if normalize=='zscore':
          mean_vals = features.mean(axis=0, keepdims=True)
          std_vals = features.std(axis=0, keepdims=True)

          features = (features - mean_vals) / std_vals
        elif normalize =='minmax':
        # Min-Max normalization
          min_vals = features.min(axis=0, keepdims=True)
          max_vals = features.max(axis=0, keepdims=True)
          features = (features - min_vals) / (max_vals - min_vals)

        elif normalize is None:
          features = features

        if metric == 'loss':
          features = features.reshape(-1, 1)
        #features = np.array(test_losses + forget_losses).reshape(-1,1)

        test_labels = [0]*len(test_losses)
        forget_labels = [1]*len(forget_losses)

        labels = np.array(test_labels + forget_labels).reshape(-1)
        #features = np.clip(features, -100, 100)

        permutation = np.random.permutation(len(labels))

        labels=labels[permutation]
        features = features[permutation]

        features = np.nan_to_num(features)
        score = evaluate_attack_model(features, labels, n_splits=n_splits, random_state=seed, n_repeats = n_repeats, scoring = scoring) # could change to scoring='accuracy'

        return score, features, labels


