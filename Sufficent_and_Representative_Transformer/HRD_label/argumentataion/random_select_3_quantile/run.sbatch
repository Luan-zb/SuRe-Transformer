#!/bin/bash
#SBATCH -J TEST
#SBATCH -p normal 
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --gres=dcu:0


echo "Start time: $(date)" #显示开始时间
echo "SLURM_JOB_ID: $SLURM_JOB_ID" #显示作业号
echo "SLURM_NNODES: $SLURM_NNODES" #显示节点数
echo "SLURM_NTASKS: $SLURM_NTASKS" #显示总任务数
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE" #显示每节点的任务数
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK" #显示每个任务使用的 CPU 数量
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION" #显示队列/分区名称
echo "SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR" #显示提交作业目录的路径
echo "SLURM_NODELIST:$SLURM_NODELIST" #显示执行节点列表名称

source ~/.bashrc

module purge
module load compiler/devtoolset/7.3.1
module load mpi/openmpi/4.0.4/gcc-7.3.1
module load compiler/rocm/dtk/23.04

conda activate pytorch
which python

python get_hrd_label.py  --tcga_breast_h5_dir /work1/lhj/project/HrdTransformer/get_feature_multi/features_argumentated_correct/200n_600s

echo "End time: $(date)"
