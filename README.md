# DSV: An Alignment Validation Loss for Self-supervised Outlier Model Selection

This is an official code repository for [DSV: An Alignment Validation Loss for
Self-supervised Outlier Model Selection](https://arxiv.org/abs/2307.06534),
which will be presented in ECML PKDD 2023. This repository is partially based on
https://github.com/Runinho/pytorch-cutpaste.

## Requirements

Python 3.8.12 is recommended. Please see `requirements.txt` for the required
packages.

## Datasets

You should first download the datasets to run our code. Change the `DATA_PATH`
variable in `utils.py` based on the location of your datasets.
- MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad
- MPDD: https://github.com/stepanje/MPDD

## Training

Run `train.py` to train anomaly detector models. The default configuration in
the file, including model hyperparameters, are set to those used in the paper.
The parameters of models, learned embeddings, and anomaly scores are stored as a
result of training. You can test the training script by typing the following
command:
```
cd ../src
python train.py \
  --type bottle \
  --gpu 0 \
  --epochs 100 \
  --test-epochs 100 \
  --augment cutdiff \
  --patch-size 0.05 0.10 \
  --patch-aspect 0.3 0.5 \
  --patch-angle 30 \
  --out ../out
```

## Evaluation

Run `eval.py` to select the model based on the values of validation losses. It
also generates useful figures on embeddings and anomaly scores.
