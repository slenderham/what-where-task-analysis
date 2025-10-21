#!/bin/bash

#SBATCH --job-name=perturbation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --mem=16GB
#SBATCH --output=slurm_%j.txt

python test_perturbation.py