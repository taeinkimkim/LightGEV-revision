#!/bin/bash
#SBATCH --job-name=plot_val_metrics_lightstereo@taein
#SBATCH --output=logs/plot/slurm_%x_%j.log
#SBATCH --error=logs/plot/slurm_%x_%j.log
#SBATCH -p Foreign_students
#SBATCH --gres=gpu:gpu1:1

echo "HOST=$(hostname)"
echo "JOBID=$SLURM_JOB_ID"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd ~/Workspace/taein/LightGEV/LightGEV-revision

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lightgev@taein

nvidia-smi -L

sh V_experiments/plotting/val_metrics.sh
# sh V_experiments/plotting/alpha_metrics.sh