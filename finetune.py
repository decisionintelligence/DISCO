#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import time
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import numpy as np

from sklearn.metrics import confusion_matrix, precision_recall_curve

from datasets.dtd import DTD
from datasets.pets import Pets
from datasets.cars import Cars
from datasets.food import Food
from datasets.sun397 import SUN397
from datasets.voc2007 import VOC2007
from datasets.flowers import Flowers
from datasets.aircraft import Aircraft
from datasets.caltech101 import Caltech101

from models.group1.initialize import initialize_model
from utils import get_logger0
import random

from get_dataloader import prepare_data, get_data
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def count_acc(pred, label, metric):
    if metric == 'accuracy':
        return pred.eq(label.view_as(pred)).to(torch.float32).mean().item()
    elif metric == 'mean per-class accuracy':
        # get the confusion matrix
        cm = confusion_matrix(label.cpu(), pred.detach().cpu())
        cm = cm.diagonal() / cm.sum(axis=1)
        return cm.mean()
    elif metric == 'mAP':
        aps = []
        for cls in range(label.size(1)):
            ap = voc_eval_cls(label[:, cls].cpu(), pred[:, cls].detach().cpu())
            aps.append(ap)
        mAP = np.mean(aps)
        return mAP


def voc_ap(rec, prec):
    """
    average precision calculations for PASCAL VOC 2007 metric, 11-recall-point based AP
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :return: average precision
    """
    ap = 0.
    for t in np.linspace(0, 1, 11):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap += p / 11.
    return ap


def voc_eval_cls(y_true, y_pred):
    # get precision and recall
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    # compute average precision
    ap = voc_ap(rec, prec)
    return ap


# Testing classes and functions
class FinetuneModel(nn.Module):
    def __init__(self, model, num_classes, steps, metric, device, feature_dim):
        super().__init__()
        self.num_classes = num_classes
        self.steps = steps
        self.metric = metric
        self.device = device
        self.model = model
        self.model = self.model.to(self.device)
        self.model.train()
        self.criterion = nn.BCEWithLogitsLoss() if self.metric == 'mAP' else nn.CrossEntropyLoss()

    def tune(self, train_loader, test_loader, lr, wd):
        # set up optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.steps)
        _logger.info(optimizer)
        # train the model with labels on the validation data
        self.model.train()
        train_loss = AverageMeter('loss', ':.4e')
        train_acc = AverageMeter('acc', ':6.2f')
        total_norm = AverageMeter('gnorm', ':2.4f')
        step = 0
        
        running = True
        while running:
            for data, targets in train_loader:
                if step >= self.steps:
                    running = False
                    break

                data, targets = data.to(self.device), targets.to(self.device)
                if self.metric == 'mAP':
                    targets = targets.to(torch.float32)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, targets)
                if self.metric == 'mAP':
                    output = (output >= 0).to(torch.float32)
                else:
                    output = output.argmax(dim=1)
                # during training we can always track traditional accuracy, it'll be easier
                acc = 100. * count_acc(output, targets, "accuracy")
                loss.backward()
                gnorm = 0.
                for p in self.model.parameters():
                    param_norm = p.grad.data.norm(2)
                    gnorm += param_norm.item() ** 2
                gnorm = gnorm ** (1. / 2)
                optimizer.step()

                total_norm.update(gnorm, data.size(0))
                train_loss.update(loss.item(), data.size(0))
                train_acc.update(acc, data.size(0))
                if step % 100 == 0:
                    _logger.info(
                        'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                        'LR: {lr:.6f}  '
                        'NORM: {norm.val:2.4f} ({norm.avg:>2.4f}) '
                        'Acc: {acc.val:>9.6f} ({acc.avg:>6.4f})  '.format(
                            self.steps,
                            step, self.steps,
                            100. * step / self.steps,
                            loss=train_loss,
                            lr=scheduler.optimizer.param_groups[0]['lr'],
                            norm=total_norm,
                            acc=train_acc
                        ))
                    global args
                    global output_dir

                    # checkpoint save
                    checkpoint_name = os.path.join(output_dir , '{}_{}_finetune_step{}_ckpt.pth'.format(
                                        args.model, args.dataset, step))
                    save_dict = {
                            'model': self.model.state_dict(),
                            'steps': step,
                            }
                    torch.save(save_dict, checkpoint_name)
                scheduler.step()

                step += 1
        #pbar.close()
        val_loss, val_acc = self.test_classifier(test_loader)
        return val_acc

    def test_classifier(self, data_loader):
        self.model.eval()
        test_loss, test_acc = 0, 0
        num_data_points = 0
        preds, labels = [], []
        with torch.no_grad():
            for i, (data, targets) in enumerate(data_loader):
                num_data_points += data.size(0)
                data, targets = data.to(self.device), targets.to(self.device)
                if self.metric == 'mAP':
                    targets = targets.to(torch.float32)
                output = self.model(data)
                tl = self.criterion(output, targets).item()
                tl *= data.size(0)
                test_loss += tl

                if self.metric in 'accuracy':
                    ta = 100. * count_acc(output.argmax(dim=1), targets, self.metric)
                    ta *= data.size(0)
                    test_acc += ta
                elif self.metric == 'mean per-class accuracy':
                    pred = output.argmax(dim=1).detach()
                    preds.append(pred)
                    labels.append(targets)
                elif self.metric == 'mAP':
                    #pred = (output >= 0).to(torch.float32)
                    pred = output.detach()
                    preds.append(pred)
                    labels.append(targets)

        if self.metric == 'accuracy':
            test_acc /= num_data_points
        elif self.metric == 'mean per-class accuracy':
            preds = torch.cat(preds)
            labels = torch.cat(labels)
            test_acc = 100. * count_acc(preds, labels, self.metric)
        elif self.metric == 'mAP':
            preds = torch.cat(preds)
            labels = torch.cat(labels)
            print(preds, labels)
            test_acc = 100. * count_acc(preds, labels, self.metric)
        test_loss /= num_data_points

        self.model.train()
        return test_loss, test_acc


class FinetuneTester():
    def __init__(self, model_name, train_loader, val_loader, trainval_loader, test_loader,
                 metric, device, num_classes, feature_dim=2048, grid=None, steps=5000):
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.trainval_loader = trainval_loader
        self.test_loader = test_loader
        self.metric = metric
        self.device = device
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.grid = grid
        self.steps = steps
        self.best_params = {}
        for (lr, wd) in grid:
            self.best_params['lr'] = lr
            self.best_params['wd'] = wd

    def validate(self):
        best_score = 0
        for i, (lr, wd) in enumerate(grid):
            _logger.info('Run {}'.format(i))
            _logger.info('lr={}, wd={}'.format(lr, wd))

            # load pretrained model
            self.model, _, dim, numsize = initialize_model(self.model_name, self.num_classes)
            self.model = self.model.to(self.device)
            self.finetuner = FinetuneModel(self.model, self.num_classes, self.steps,
                                           self.metric, self.device, self.feature_dim)
            val_acc = self.finetuner.tune(self.train_loader, self.val_loader, lr, wd)
            _logger.info('Finetuned val accuracy {:.2f}%'.format(val_acc))

            if val_acc > best_score:
                best_score = val_acc
                self.best_params['lr'] = lr
                self.best_params['wd'] = wd
                _logger.info("New best {}".format(self.best_params))

    def evaluate(self):
        _logger.info("Best params {}".format(self.best_params))

        # load pretrained model
        self.model, _, dim, numsize = initialize_model(self.model_name, self.num_classes)
        self.model = self.model.to(self.device)
        
        self.finetuner = FinetuneModel(self.model, self.num_classes, self.steps,
                                       self.metric, self.device, self.feature_dim)
        test_score = self.finetuner.tune(self.trainval_loader, self.test_loader, 
                                        self.best_params['lr'], self.best_params['wd'])
        _logger.info('Finetuned test accuracy {:.2f}%'.format(test_score))
        global args
        global output_dir
        checkpoint_name = os.path.join(output_dir, '{}_{}_finetune_ckpt.pth'.format(
                                        args.model, args.dataset))
        save_dict = {
            'model': self.model.state_dict(),
            'steps': 5000,
            'best_params': self.best_params,
            'acc': test_score,
        }
        torch.save(save_dict, checkpoint_name)

        return test_score


# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate pretrained self-supervised model via finetuning.')
    parser.add_argument('-m', '--model', type=str, default='resnet50',
                        help='name of the pretrained model to load and evaluate')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10',
                        help='name of the dataset to evaluate on')
    parser.add_argument('-b', '--batch-size', type=int, default=64, 
                        help='the size of the mini-batches when inferring features')
    parser.add_argument('-i', '--image-size', type=int, default=224, 
                        help='the size of the input images')
    parser.add_argument('-w', '--workers', type=int, default=8, 
                        help='the number of workers for loading the data')
    parser.add_argument('-g', '--grid-size', type=int, default=4, 
                        help='the number of learning rate values in the search grid')
    parser.add_argument('--steps', type=int, default=5000, 
                        help='the number of finetuning steps')
    parser.add_argument('--no-da', action='store_true', default=False, 
                        help='disables data augmentation during training')
    parser.add_argument('-n', '--no-norm', action='store_true', default=False,
                        help='whether to turn off data normalisation (based on ImageNet values)')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='CUDA or CPU training (cuda | cpu)')
    parser.add_argument('-blr', '--best-lr', type=float, default=0.01,
                        help='the best learning rate when inferring features')
    parser.add_argument('-bwd', '--best-wd', type=float, default=0.009,
                        help='the best weight decay when inferring features')
    parser.add_argument('-v', '--validate', action='store_true', default=False, 
                        help='whether validate model with best parameters')
    
    
    args = parser.parse_args()
    args.norm = not args.no_norm
    args.da = not args.no_da
    del args.no_norm
    del args.no_da

    #random seed
    seed = 10
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # load dataset
    output_dir = os.path.join('output', 'finetune_log_seed{}'.format(str(seed)))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logname = os.path.join(output_dir, '{}_{}_finetune.log'.format(args.model, args.dataset))
    _logger = get_logger0(filename=logname, name='Finetune')

    _logger.info(args)

    # name: {class, root, num_classes, metric}
    FINETUNE_DATASETS = {
        'aircraft': [Aircraft, 'data/Aircraft', 100, 'mean per-class accuracy'],
        'caltech101': [Caltech101, 'data/caltech101', 102, 'mean per-class accuracy'],
        'cars': [Cars, 'data/Cars', 196, 'accuracy'],
        'cifar10': [datasets.CIFAR10, 'data/CIFAR10', 10, 'accuracy'],
        'cifar100': [datasets.CIFAR100, 'data/CIFAR100', 100, 'accuracy'],
        'dtd': [DTD, 'data/DTD', 47, 'accuracy'],
        'flowers': [Flowers, 'data/Flowers', 102, 'mean per-class accuracy'],
        'food': [Food, 'data/Food', 101, 'accuracy'],
        'pets': [Pets, 'data/Pets', 37, 'mean per-class accuracy'],
        'sun397': [SUN397, 'data/SUN397', 397, 'accuracy'],
        'voc2007': [VOC2007, 'data/VOC2007', 20, 'mAP'],
    }
    dset, data_dir, num_classes, metric = FINETUNE_DATASETS[args.dataset]
    train_loader, val_loader, trainval_loader, test_loader, all_loader = prepare_data(
        dset, data_dir, args.batch_size, args.image_size, normalisation=args.norm,  seed=seed)

    # set up learning rate and weight decay ranges
    if args.validate:
        grid = [(args.best_lr, args.best_wd)]
        tester = FinetuneTester(args.model, train_loader, val_loader, trainval_loader, test_loader,
                                metric, args.device, num_classes, grid=grid, steps=args.steps)
        test_score = tester.evaluate()

    else:
        lr = torch.logspace(-4, -1, args.grid_size).flip(dims=(0,))
        wd = torch.cat([torch.zeros(1), torch.logspace(-6, -3, args.grid_size)])
        grid = [(l.item(), (w / l).item()) for l in lr for w in wd]

        # evaluate model on dataset by finetuning
        tester = FinetuneTester(args.model, train_loader, val_loader, trainval_loader, test_loader,
                                metric, args.device, num_classes, grid=grid, steps=args.steps)

        # tune hyperparameters
        tester.validate()
        # use best hyperparameters to finally evaluate the model
        test_score = tester.evaluate()
