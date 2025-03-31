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

## Performance of Noised Retrained Model

### Lacuna-100
Below we also provide numerical results for $\epsilon = 40$ of the noised retrained model. These results will be added to Table 8 and 9.

| Algorithm    | Retrained Unlearned Error | Retrained Test Error | Retrained Train Error | Retrained MIA Score |
|---------------|----------------|------------|-------------|-----------|
| R2D, 26%     | 0.3780          | 0.3826     | 0.3784      | 0.5591    |
| R2D, 51%     | 0.1745          | 0.1969     | 0.1948      | 0.5735    |
| R2D, 75%     | 0.1002          | 0.1210     | 0.1103      | 0.5731    |
| R2D, 100%    | 0.0969          | 0.0912     | 0.0647      | 0.5958    |
