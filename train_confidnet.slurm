#!/bin/bash

#SBATCH --output=temp.out
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=3
#SBATCH -J confidnet_cifar10_vgg16_mixup_augm-confidnet
#SBATCH --partition A100,V100


export PATH="/home/ids/qbouniot/miniconda3/envs/uncertainty/bin:/home/ids/qbouniot/miniconda3/condabin:$PATH"
export PATH="$PATH:/usr/local/cuda-12.0/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-12.0/targets/x86_64-linux/lib/stubs/:/usr/local/cuda-12.0/lib64"


eval "$(conda shell.bash hook)"
conda activate uncertainty

EXP_DIR="exps/cifar10/vgg16-mixup_augm-confidnet_500ep"

mkdir $EXP_DIR

set -x
srun --kill-on-bad-exit=1 --output="$EXP_DIR/training.out" python ./confidnet/train.py -c configs/cifar10_vgg_confidnet.yaml

# srun --kill-on-bad-exit=1 --output="exps/cifar10/vgg16-confidnet_500ep/eval_mcp.out" python ./confidnet/test.py -c configs/cifar10_vgg_confidnet.yaml

# echo "DONE !"