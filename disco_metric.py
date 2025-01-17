#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint

import torch
import numpy as np
from utils import Ncc_score, feature_reduce
import json
import time

def save_score(score_dict, fpath):
    with open(fpath, "w") as f:
        # write dict
        json.dump(score_dict, f)


def exist_score(model_name, fpath):
    with open(fpath, "r") as f:
        result = json.load(f)
        if model_name in result.keys():
            return True
        else:
            return False


# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate transferability score.')
    parser.add_argument('-m', '--model', type=str, default='resnet50',
                        help='name of the pretrained model to load and evaluate (deepcluster-v2.pth | supervised)')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10',
                        help='name of the dataset to evaluate on')
    parser.add_argument('-me', '--metric', type=str, default='disco',
                        help='name of the method for measuring transferability')
    parser.add_argument('--pcadim', type=int, default=128,
                        help='PCA reduction dimension')
    parser.add_argument('--output-dir', type=str, default='./metircs',
                        help='dir of output score')

    args = parser.parse_args()



    score_dict = {}
    fpath = os.path.join(args.output_dir, 'group1', args.metric)

    if not os.path.exists(fpath):
        os.makedirs(fpath)
    fpath = os.path.join(fpath, f'{args.dataset}_metrics.json')

    if not os.path.exists(fpath):
        save_score(score_dict, fpath)
    else:
        with open(fpath, "r") as f:
            score_dict = json.load(f)

    models_hub = ['inception_v3', 'mobilenet_v2', 'mnasnet1_0', 'densenet121', 'densenet169', 'densenet201',
                    'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet']
    for model in models_hub:
        model = args.model

        model_npy_feature = os.path.join('./results_f/group1', f'{args.model}_{args.dataset}_feature.npy')
        model_npy_label = os.path.join('./results_f/group1', f'{args.model}_{args.dataset}_label.npy')
        model_npy_output = os.path.join('./results_f/group1', f'{args.model}_{args.dataset}_output.npy')
        X_features, y_labels = np.load(model_npy_feature), np.load(model_npy_label)
        X_features = feature_reduce(X_features, args.pcadim)
        print(f'X_features shape:{X_features.shape} and y_labels shape:{y_labels.shape}')

        X_features = torch.Tensor(X_features)
        X_features.cuda()

        start_time = time.time()

        # svd
        U, s, VT = torch.linalg.svd(X_features)
        U = U.cpu().numpy()
        s = s.cpu().numpy()
        VT = VT.cpu().numpy()


        nccscore_list = {}
        ratio_list = {}
        score_dict[model] = {}

        sum_s = np.sum(s)

        divide = 8
        partition_size = args.pcadim // divide

        for i in range(divide):
            sub_U = U[:, i * partition_size:(i + 1) * partition_size]
            sub_s = np.diag(s[i * partition_size:(i + 1) * partition_size])
            sub_VT = VT[i * partition_size:(i + 1) * partition_size, :]
            i_features = np.dot(sub_U, np.dot(sub_s, sub_VT))

            nccscore_list[i] = Ncc_score(i_features, y_labels)
            ratio = np.sum(s[i * partition_size:(i + 1) * partition_size]) / sum_s
            ratio_list[i] = ratio.astype(np.float64)

        end_time = time.time()
        score_dict[model]['time'] = end_time - start_time
        score_dict[model]['ncc'] = nccscore_list
        score_dict[model]['ratio'] = ratio_list
        save_score(score_dict, fpath)

    save_score(score_dict, fpath)



