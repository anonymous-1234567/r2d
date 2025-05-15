# #adapted from constrained newton step github repo https://github.com/zhangbinchi/certified-deep-unlearning/tree/main


    
    
import argparse
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold


from imblearn.under_sampling import RandomUnderSampler

import random

class DT:
    def __init__(self, balance=True):
        if balance:
            self.model = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0, class_weight='balanced')
        else:
            self.model = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
    def train_model(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        return accuracy_score(test_y, pred_y)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        return roc_auc_score(test_y, pred_y[:, 1])


class RF:
    def __init__(self, min_samples_leaf=30, balance=True):
        if balance:
            self.model = RandomForestClassifier(random_state=0, n_estimators=500, min_samples_leaf=min_samples_leaf, class_weight='balanced')
        else:
            self.model = RandomForestClassifier(random_state=0, n_estimators=500, min_samples_leaf=min_samples_leaf)

    def train_model(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        return accuracy_score(test_y, pred_y)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        return roc_auc_score(test_y, pred_y[:, 1])


class MLP_INF:
    def __init__(self):
        if balance:
            self.model = MLPClassifier(early_stopping=True, learning_rate_init=0.01)
        else:
            self.model = MLPClassifier(early_stopping=True, learning_rate_init=0.01)

    def scaler_data(self, data):
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        return data

    def train_model(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        return accuracy_score(test_y, pred_y)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        return roc_auc_score(test_y, pred_y[:, 1])


class LR:
    def __init__(self, balance = True):
        if balance:
            self.model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, n_jobs=1, class_weight='balanced')
        else:
            self.model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, n_jobs=1)

    def train_model(self, train_x, train_y):
        self.scaler = preprocessing.StandardScaler().fit(train_x)
        # temperature = 1
        # train_x /= temperature
        self.model.fit(self.scaler.transform(train_x), train_y)

    def predict_proba(self, test_x):
        self.scaler = preprocessing.StandardScaler().fit(test_x)
        return self.model.predict_proba(self.scaler.transform(test_x))

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(self.scaler.transform(test_x))

        return accuracy_score(test_y, pred_y)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(self.scaler.transform(test_x))
        return roc_auc_score(test_y, pred_y[:, 1])  # binary class classification AUC
        # return roc_auc_score(test_y, pred_y[:, 1], multi_class="ovr", average=None)  # multi-class AUC


def posterior(dataloader, model, device):
    posterior_list = []
    with torch.no_grad():
        for data, labels, _ in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            post = torch.softmax(outputs, 1)
            posterior_list.append(post)
    return torch.cat(posterior_list, 0)


def construct_feature(post_ori, post_unl, method):
    if method == "direct_diff":
        return post_ori - post_unl

    elif method == "sorted_diff":
        for index, posterior in enumerate(post_ori):
            sort_indices = np.argsort(posterior)
            post_ori[index] = posterior[sort_indices]
            post_unl[index] = post_unl[index][sort_indices]
        return post_ori - post_unl

    elif method == "l2_distance":
        feat = torch.ones(post_ori.shape[0])
        for index in range(post_ori.shape[0]):
            euclidean = distance.euclidean(post_ori[index], post_unl[index])
            feat[index] = euclidean
        return feat.unsqueeze(1)

    elif method == "direct_concat":
        return torch.cat([post_ori, post_unl], 1)

    elif method == "sorted_concat":
        for index, posterior in enumerate(post_ori):
            sort_indices = np.argsort(posterior)
            post_ori[index] = posterior[sort_indices]
            post_unl[index] = post_unl[index][sort_indices]
        return torch.cat([post_ori, post_unl], 1)

    
def mia_unlearning(res_loader, unl_loader, model, unlearn_model, device, attack_model='LR', method='l2_distance', seed=1, crossval=True, n_samples=1):

    '''
    res_loader: data that has never been seen by the model
    unl_loader: data learned and then unlearned
    model: og model
    unlearn_model: model after unlearning
    device: device
    n_samples: number of undersamples (for making a balanced dataset)
    
    '''
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Set the random seed for CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    unl_pos_post = posterior(unl_loader, unlearn_model, device).detach().cpu()
    unl_neg_post = posterior(res_loader, unlearn_model, device).detach().cpu()
    ori_pos_post = posterior(unl_loader, model, device).detach().cpu()
    ori_neg_post = posterior(res_loader, model, device).detach().cpu()

    feat_pos = construct_feature(ori_pos_post, unl_pos_post, method)
    feat_neg = construct_feature(ori_neg_post, unl_neg_post, method)

    feat = torch.cat([feat_pos, feat_neg], 0).numpy()
    label = torch.cat([torch.ones(feat_pos.shape[0]), torch.zeros(feat_neg.shape[0])], 0).numpy().astype('int')

    if attack_model == 'LR':
        attack_model = LR()
    elif attack_model == 'DT':
        attack_model = DT()
    elif attack_model == 'MLP':
        attack_model = MLP_INF()
    elif attack_model == 'RF':
        attack_model = RF()
    else:
        raise Exception("invalid attack name")
        
    total_acc = 0
    total_auc = 0
    all_values = []
    for i in range(n_samples):
        rus = RandomUnderSampler(random_state=seed+i)
        X, y = rus.fit_resample(feat, label)
        if not crossval: #plain train-test split

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,         # or whatever fraction you like
                random_state=seed+i,       # for reproducibility
                stratify=y             # ensures class balance in splits
                )
            attack_model.train_model(X_train, y_train)
            train_acc = attack_model.test_model_acc(feat, label)
            train_auc = attack_model.test_model_auc(feat, label)
            print(f"Attack Train Accuracy: {100 * train_acc:.4f}%, Attack Train AUC: {100 * train_auc:.4f}%")

            acc = attack_model.test_model_acc(X_test, y_test)
            auc = attack_model.test_model_auc(X_test, y_test)

            total_acc += acc
            total_auc += auc


        else:

            cv = RepeatedStratifiedKFold(n_splits=5, random_state=seed+i, n_repeats=10) #automatically shuffles
            auc = cross_val_score(attack_model.model, X, y, cv=cv, scoring='roc_auc')
            all_values += list(auc)

            acc = cross_val_score(attack_model.model, X, y, cv=cv, scoring='accuracy')
            total_acc += np.mean(acc)
            total_auc += np.mean(auc)
                
        
        auc_std = np.std(all_values)
        
        return total_acc/n_samples, total_auc/n_samples, auc_std

