#!/usr/bin/env python
# coding: utf-8

import sys
from math import sqrt

import torch
import torch.nn as nn
import models.group1 as models

import logging

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


def get_logger0(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a+")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def load_model(configs):   
    print("Using torchvision Pretrained Models")
    if configs.model in ('inception_v3', 'googlenet'):
        model = models.__dict__[configs.model](pretrained=True, aux_logits=False).cuda()
    else:
        model = models.__dict__[configs.model](pretrained=True).cuda()

    if configs.model in ['mobilenet_v2', 'mnasnet1_0']:
        fc_layer = model.classifier[1]
    elif configs.model in ['densenet121', 'densenet169', 'densenet201']:
        fc_layer = model.classifier
    elif configs.model in ['resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet', 'inception_v3']:
        fc_layer = model.fc
    else:
        # try your customized model
        raise NotImplementedError
    feature_dim = fc_layer.in_features
    return model, fc_layer, feature_dim


def forward_pass(score_loader, model, fc_layer, model_name='resnet50'):
    """
    a forward pass on target dataset
    :params score_loader: the dataloader for scoring transferability
    :params model: the model for scoring transferability
    :params fc_layer: the fc layer of the model, for registering hooks
    returns
        features: extracted features of model
        outputs: outputs of model
        targets: ground-truth labels of dataset
    """
    features = []
    outputs = []
    targets = []
    model = model.cuda()

    def hook_fn_forward(module, input, output):
        features.append(input[0].detach().cpu())
        # outputs.append(output.detach().cpu())

    forward_hook = fc_layer.register_forward_hook(hook_fn_forward)

    model.eval()
    with torch.no_grad():
        for _, (data, target) in enumerate(score_loader):
            targets.append(target)
            data = data.cuda()
            _ = model(data)

    forward_hook.remove()
    if model_name in ['pvt_tiny', 'pvt_small', 'pvt_medium', 'deit_small', 
                    'deit_tiny', 'deit_base', 'dino_base', 'dino_small', 
                    'mocov3_small']:
        features = torch.cat([x[:, 0] for x in features])

    elif model_name in ['pvtv2_b2', 'pvtv2_b3']:
        features = torch.cat([x.mean(dim=1) for x in features])
    
    elif model_name in ['swin_t', 'swin_s']:
        avgpool = nn.AdaptiveAvgPool1d(1).cuda()
        features = torch.cat([torch.flatten(avgpool(x.transpose(1, 2)), 1) for x in features])

    else:
        features = torch.cat([x for x in features])
    outputs = torch.cat([x for x in outputs])
    targets = torch.cat([x for x in targets])
    
    return features.cpu(), outputs, targets



def wpearson(vec_1, vec_2, weights=None, r=4):
    if weights is None:
        weights = [len(vec_1)-i for i in range(len(vec_1))]
    list_length = len(vec_1)
    weights = list(map(float, weights))
    vec_1 = list(map(float, vec_1))
    vec_2 = list(map(float, vec_2))
    if any(len(x) != list_length for x in [vec_2, weights]):
        print('Vector/Weight sizes not equal.')
        sys.exit(1)
    w_sum = sum(weights)

    # Calculate the weighted average relative value of vector 1 and vector 2.
    vec1_sum = 0.0
    vec2_sum = 0.0
    for x in range(len(vec_1)):
        vec1_sum += (weights[x] * vec_1[x])
        vec2_sum += (weights[x] * vec_2[x])	
    vec1_avg = (vec1_sum / w_sum)
    vec2_avg = (vec2_sum / w_sum)

    # Calculate wPCC
    sum_top = 0.0
    sum_bottom1 = 0.0
    sum_bottom2 = 0.0
    for x in range(len(vec_1)):
        dif_1 = (vec_1[x] - vec1_avg)
        dif_2 = (vec_2[x] - vec2_avg)
        sum_top += (weights[x] * dif_1 * dif_2)
        sum_bottom1 += (dif_1 ** 2 ) * (weights[x])
        sum_bottom2 += (dif_2 ** 2) * (weights[x])

    cor = sum_top / (sqrt(sum_bottom1 * sum_bottom2))
    return round(cor, r)



def to_torch(ndarray):
    from collections.abc import Sequence
    if ndarray is None: return None
    if isinstance(ndarray, Sequence):
        return [to_torch(ndarray_) for ndarray_ in ndarray if ndarray_ is not None]
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    if torch.is_tensor(ndarray): return ndarray
    raise ValueError('fail convert')


def softmax(X, copy=True):
    if copy:
        X = np.copy(X)
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X

def softmax_torch(X, copy=True):
    if copy:
        X = torch.copy(X)
    max_prob = torch.max(X, dim=1).reshape((-1, 1))
    X -= max_prob
    torch.exp(X, X)
    sum_prob = torch.sum(X, dim=1).reshape((-1, 1))
    X /= sum_prob
    return X

def _class_means(X, y):
    """Compute class means.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    Returns
    -------
    means : array-like of shape (n_classes, n_features)
        Class means.
    means ： array-like of shape (n_classes, n_features)
        Outer classes means.
    """
    classes, y = np.unique(y, return_inverse=True)
    cnt = np.bincount(y)
    means = np.zeros(shape=(len(classes), X.shape[1]))
    np.add.at(means, y, X)
    means /= cnt[:, None]

    means_ = np.zeros(shape=(len(classes), X.shape[1]))
    for i in range(len(classes)):
        means_[i] = (np.sum(means, axis=0) - means[i]) / (len(classes) - 1)
    return means, means_


def split_data(data: np.ndarray, percent_train: float):
    split = data.shape[0] - int(percent_train * data.shape[0])
    return data[:split], data[split:]


def feature_reduce(features: np.ndarray, f: int=None):
    """
        Use PCA to reduce the dimensionality of the features.
        If f is none, return the original features.
        If f < features.shape[0], default f to be the shape.
	"""
    if f is None:
        return features
    if f > features.shape[0]:
        f = features.shape[0]

    return sklearn.decomposition.PCA(
        n_components=f,
        svd_solver='randomized',
        random_state=1919,
        iterated_power=1).fit_transform(features)


def Ncc_score(X_features, y_labels):
    labelset = set(y_labels.tolist())
    labelset = [int(i) for i in labelset]

    N, dim = X_features.shape
    num_classes = len(labelset)

    probabality = np.zeros([N, num_classes])
    for label in labelset:
        class_features = X_features[y_labels == label]
        mean_c = np.mean(class_features, axis=0)
        Nc, _ = class_features.shape
        probabality[:, label] = multivariate_normal.logpdf(X_features, mean_c) + np.log(Nc/N)


    prob = softmax(probabality)  # N,C
    ncc_score = np.sum(prob[np.arange(N), y_labels]) / N

    return ncc_score



