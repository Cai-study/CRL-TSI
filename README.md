# CRL-TSI
Pytorch code for paper: "Causality-inspired Unsupervised Domain Adaptation with Target Style Imitation for Medical Image Segmentation"
# Preparation
## Dataset
We followed the setting of [Unsupervised Bidirectional Cross-Modality Adaptation via Deeply Synergistic Image and Feature Alignment for Medical Image Segmentation](https://arxiv.org/abs/2002.02255).
We used two datasets in this paper: Multi-Modality Whole Heart Segmentation (MMWHS) Challenge 2017 dataset and Abdominal Multi-organ Dataset.
## Preprocessing
We followed the preprocessing of [Towards Generic Semi-Supervised Framework for Volumetric Medical Image Segmentation](https://arxiv.org/abs/2310.11320), you can find the preprocessing [code](https://github.com/xmed-lab/GenericSSL) here.
## Environments
It is recommended to create an anaconda virtual environment to run the code. The python version is python-3.9.0. The detailed version of some packages is available in requirements.txt. You can install all the required packages using the following command:
```
conda create -n uda python=3.9.0
conda activate uda
cd uda/
pip install -r requirements.txt
```
# Training
Run the following commands for training
```
bash train.sh -c 0 -e CRL-TSI -t <task> -i '' -l 1e-2 -w 10 -n 600 -d true 
```
Parameters:

-c: use which gpu to train

-e: use which training script

-t: switch to different tasks:
          For UDA on MMWHS dataset: mmwhs_ct2mr for labeled CT and unlabeled MR, mmwhs_mr2ct in opposite;
          For UDA on Abdominal dataset: abdominal_ct2mr for labeled CT and unlabeled MR, abdominal_mr2ct in opposite

-i: name of current experiment, can be whatever you like

-l: learning rate

-w: weight of unsupervised loss

-n: max epochs

-d: whether to train, if true, training -> testing -> evaluating; if false, testing -> evaluating
# Evaluate
All trained model weights can be downloaded from this [link](https://pan.baidu.com/s/1UTy3eX3ynW-5grJVs9gxTg) (password: yi78).
Put the logs_CRL_TSI directory under the root directory of this repo and set -d False, then you can test and evaluate the models.
