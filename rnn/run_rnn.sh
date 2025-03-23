#!/bin/bash

#SBATCH --job-name=fp_approx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --mem=16GB
#SBATCH --output=exp/test%a/slurm_%j.txt

python train.py --cuda\
        --save_checkpoint\
        --iters 1000\
        --epochs 25\
        --hidden_size 80\
        --num_areas 1\
        --eval_samples 30\
        --l2r 1e-2\
        --l2w 1e-5\
        --init_spectral 1.0\
        --balance_ei\
        --action_input\
        --learning_rate 1e-3\
        --exp_dir exp/test$SLURM_ARRAY_TASK_ID