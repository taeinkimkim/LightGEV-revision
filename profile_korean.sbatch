#!/bin/bash
#SBATCH --job-name=profile_sota@taein
#SBATCH --output=logs/profile/slurm_%x_%j.log
#SBATCH --error=logs/profile/slurm_%x_%j.log
#SBATCH -p Korean_students
#SBATCH --gres=gpu:gpu0:1

echo "HOST=$(hostname)"
echo "JOBID=$SLURM_JOB_ID"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd ~/Workspace/taein/LightGEV/LightGEV-revision

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lightgev@taein

nvidia-smi -L

# LightGEVModify_4 module-wise runtime profiling
# sh V_experiments/profiling/stage_runtime_lightgev_modify_4.sh

# SOTA runtime profiling
sh V_experiments/profiling/model_runtime_sota_compare.sh
