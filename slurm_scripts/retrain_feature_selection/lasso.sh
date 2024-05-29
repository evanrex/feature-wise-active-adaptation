#!/bin/bash
#SBATCH -J lasso
#SBATCH --output=logs/out_%A.out
#SBATCH --error=logs/err_%A.err
#SBATCH -A COMPUTERLAB-SL2-CPU
#SBATCH --time=10:00:00
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

wandb agent evangeorgerex/fwal/5knef16b
