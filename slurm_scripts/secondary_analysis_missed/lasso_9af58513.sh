#!/bin/bash
#SBATCH -J lasso_9af58513
#SBATCH --output=logs/out_%A.out
#SBATCH --error=logs/err_%A.err
#SBATCH -A COMPUTERLAB-SL2-CPU
#SBATCH --time=10:00:00
#SBATCH -p icelake-himem 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=36000


. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

wandb agent evangeorgerex/fwal/0e94xqes
