# Amortized Variational Filtering

Code to accompany the paper [A General Framework for Amortizing Variational Filtering](http://joelouismarino.github.io/research/avf_workshop_paper.pdf) by Marino et al., 2018.

## Installation & Set-Up

First, clone the repository by opening a terminal and running:
```
$ git clone https://github.com/joelouismarino/amortized-variational-filtering
```
The code uses PyTorch version `0.3.0.post4` and visdom version `0.1.7`. To avoid conflicts with more recent versions of these packages, you may wish to create a conda environment:
```
$ conda create --name avf python=2.7
```
To enter the environment, run:
```
$ source activate avf
```
Within the environment, install PyTorch by visiting the list of versions [here](https://pytorch.org/previous-versions/), and grabbing version `0.3.0.post4` for your version of CUDA (`8.0`, `9.0`, `9.1`, etc.). Note that the code requires CUDA. Be sure to also install torchvision.

To install vidsom, run
```
(avf) $ pip install visdom==0.1.7
```
Note: to run on the KTH Actions or BAIR datasets, you will need to install scipy, moviepy, and tensorflow.

To exit the environment, run:
```
(avf) $ source deactivate
```
