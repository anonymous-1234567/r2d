#!/bin/bash
#checkpointarray=( 25 50 100 150 200 250 )
#checkpointarray=( 20 50 150 200 250 )
#checkpointarray=( 40 70 120 170 220 )
checkpointarray=( 100 )
bs=512


python3 main.py  --dataset lacuna100binary128 --model resnetsmooth --dataroot data/lacuna100binary128/ --epochs 270 --lr 0.01 --batch-size ${bs} --num-ids-forget 2 --seed 1 --no-gradient-estimation --device 0 --resume checkpoints/lacuna100binary128_resnetsmooth_1_0_forget_None_lr_0_01_bs_${bs}_ls_ce_seed_1_scheduler_1_init.pt 

for checkpoint in ${checkpointarray[@]}
do
    E=$((270 - checkpoint))
    python3 -u main.py  --dataset lacuna100binary128 --model resnetsmooth --dataroot data/lacuna100binary128/ --epochs ${E} --lr 0.01 --batch-size ${bs} --num-ids-forget 2 --seed 1 --no-gradient-estimation --device 2 --resume checkpoints/lacuna100binary128_resnetsmooth_1_0_forget_None_lr_0_01_bs_${bs}_ls_ce_seed_1_scheduler_1_${checkpoint}.pt 
done

#sudo apt install bc
#python3 main.py  --dataset lacuna100binary128 --model resnetsmooth --dataroot data/lacuna100binary128/ --epochs 300 --lr 0.01 --batch-size 512 --model-selection --plot --compute-lipschitz --save-checkpoints --seed 1