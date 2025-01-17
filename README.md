
## Requirements
- Install `PyTorch==2.1.0` and `torchvision==0.16.0` with `CUDA==12.1`:
```bash
conda install pytorch==2.1.0 torchvision==0.16.0 cudatoolkit=12.1 -c pytorch
```

## Data Preparation
- Download the downstream datasets to ./data/*.

## Pipeline of DISCO
- Fine-tune pretrained models with hyper-paramters sweep to obtain ground-truth fine-tuning results
```
bash finetune.sh
```
- Extract features of target data using pretrained models
```
python forward_feature.py --dataset cifar10
```
- Compute DISCO
```
python disco_metric.py --dataset cifar10
```
- Assess the effectiveness of DISCO
```
python tw.py --dataset cifar10
```
