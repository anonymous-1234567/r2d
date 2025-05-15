#!/bin/bash
checkpointarray=( 10 20 30 40 50 60 70 )


device=0


python3 main.py  --dataset eicu --model mlp --dataroot data/eicu/ --epochs 78 --lr 0.01 --batch-size 2048 --num-ids-forget 940 --seed 1 --device ${device} --resume checkpoints/eicu_mlp_1_0_forget_None_lr_0_01_bs_2048_ls_ce_seed_1_scheduler_1_init.pt --no-gradient-estimation 

for checkpoint in ${checkpointarray[@]}
do
    E=$((78 - checkpoint))
    python3 main.py  --dataset eicu --model mlp --dataroot data/eicu/ --epochs ${E} --lr 0.01 --batch-size 2048 --num-ids-forget 940 --seed 1 --device ${device} --resume checkpoints/eicu_mlp_1_0_forget_None_lr_0_01_bs_2048_ls_ce_seed_1_scheduler_1_${checkpoint}.pt --no-gradient-estimation 
done

#sudo apt install bc
#python3 main.py  --dataset eicu --model mlp --dataroot data/eicu/ --epochs 80 --lr 0.01 --batch-size 2048 --seed 1 --plot --model-selection --device 1 --compute-lipschitz --save-checkpoints