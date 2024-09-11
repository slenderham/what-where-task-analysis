#!/bin/bash

# Name of the job
#SBATCH --job-name=preprocessing

# Number of compute nodes
#SBATCH --nodes=1

# Number of tasks per node
#SBATCH --ntasks-per-node=1

# Number of CPUs per task
#SBATCH --cpus-per-task=16

# Request memory
#SBATCH --mem=32G

# Walltime (job duration)
#SBATCH --time=120:00:00

#SBATCH --output=slurm_%j.txt


python ./shuffle.py