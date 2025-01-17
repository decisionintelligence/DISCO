#!/usr/bin/env bash

dataset=( 'aircraft' 'flowers' 'dtd' 'voc2007' 'caltech101' 'pets' 'cars' 'cifar10' 'sun397' 'cifar100'  'food')
modelname=('resnet34' 'resnet50' 'resnet101' 'resnet152' 'densenet121' 'densenet169' 'densenet201' 'inception_v3' 'mobilenet_v2' 'mnasnet1_0' 'googlenet')

for data in "${dataset[@]}"
do
    for model in "${modelname[@]}"
    do
        echo $model, $data
        python finetune.py --model $model --dataset $data
    done
done

