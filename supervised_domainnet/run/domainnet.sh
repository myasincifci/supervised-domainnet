#!/bin/bash
#SBATCH --job-name=sup-domainnet
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=40gb:1

#SBATCH --ntasks-per-node=16
#SBATCH --mem=32G

#SBATCH --output=logs/job-%j.out

#SBATCH --array=0-0
declare -a lrs=(2e-3 5e-4 1e-4)

# 1. copy the squashed dataset to the nodes /tmp 
rsync -ah --progress /home/myasincifci/data/domainnet_v1.0.sqfs /tmp/

apptainer run --nv -B /tmp/domainnet_v1.0.sqfs:/data/domainnet_v1.0:image-src=/ /home/myasincifci/supervised-domainnet/apptainer/environment.sif \
    python supervised_domainnet/train.py \
        --config-name base \
        param.lr=${lrs[${SLURM_ARRAY_TASK_ID}]}