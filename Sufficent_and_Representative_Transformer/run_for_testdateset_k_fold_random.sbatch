#!/bin/bash
#SBATCH -J TEST
#SBATCH -p normal 
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=100G
#SBATCH --cpus-per-task=7
#SBATCH --gres=dcu:4
#SBATCH -o logs_k_fold/test/%j.out
#SBATCH -e logs_k_fold/test/%j.err

echo "Start time: $(date)" #显示开始时间

source ~/.bashrc

module purge
module load compiler/devtoolset/7.3.1
module load mpi/openmpi/4.0.4/gcc-7.3.1
module load compiler/rocm/dtk/23.04

export MPICC=$(which mpicc)

conda activate pytorch
which python

#python test_k_fold_random_draw_test.py
python test_k_fold_random_draw.py
echo "End time: $(date)"
