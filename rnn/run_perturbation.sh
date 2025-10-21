#!/bin/bash

#SBATCH --job-name=perturbation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --mem=64GB
#SBATCH --output=slurm_%j.txt

python -u test_perturbation.py