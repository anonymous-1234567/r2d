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
python3 main.py  --dataset eicu --model mlp --dataroot data/eicu/ --epochs 80 --lr 0.01 --batch-size 2048 --seed 1 --plot --model-selection --device 1 --compute-lipschitz --save-checkpoints
```

## R2D Experiments
To rerun the R2D checkpointing experiments, run the bash scripts `checkpointbash_eicu.sh` and `checkpointbash_lacuna.sh`.

## Baseline Methods
See the Jupyter notebooks `eicu_baseline_implementation.ipynb` and `lacuna_baseline_implementation.ipynb` for the non-certified baselines. See `unlearning_certified.ipynb` for the certified baselines.
