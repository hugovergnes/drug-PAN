# drug-graph-convolutional-kernel
This is an implementation of a Path Integral Based Convolution Graph Neural Network to solve the molhiv dataset on OGB.

## Prerequisites

## Getting Started
1. Clone the repository on your local machine.
2. Navigate to the relevant branch: 1) `baseline-colab` for the baseline model, 2) `edge-refactor` for the improved model
4. Navigate to the code folder (`cd code`)
5. Run your first training (`python lightning_train.py`)

## Features
This training was mainly ran on a laptop CPU. If you want to run it on a GPU for free using collab, we made a readily implemented notebook to run on colab (`train_model_colab.ipynb`).
You ll need to upload it on collab as well as Norm.py, model.py, lightning_model.py and parameters.json and the notebook will take care of installing all other dependencies.

## Acknowledgment
Special thanks to the writter of the paper: Path Integral Based Convolution and Pooling for Graph Neural Networks
Checkout their paper here: https://arxiv.org/abs/2006.16811

Thanks to the teaching staff of CS224W! 

