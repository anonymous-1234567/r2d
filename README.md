# R2D: Rewind-To-Delete


## Overview
This repository contains the code and scripts used for the experiments in our paper, "Rewind-To-Delete: Certified Machine Unlearning for Nonconvex Functions."


### Prerequisites
Package dependencies are included in `r2d.yml`

## Datasets
Download the MAAD-Face annotations here: https://github.com/pterhoer/MAAD-Face

MAAD-Face preprocessing procedure and Lacuna-100 generation from VGGFace2 are in `lacuna_maad_preprocessing_binary.ipynb`

To access the eICU dataset, follow the instructions here: https://eicu-crd.mit.edu/gettingstarted/access/

## Training the Model
To train the model from scratch, run:
```sh
python3 main.py  --dataset lacuna100binary128 --model resnetsmooth --dataroot data/lacuna100binary128/ --epochs 100 --lr 0.1 --batch-size 256 --compute-lipschitz --scheduler 0.9 --model-selection --save-checkpoints
```
