#!/bin/bash
#SBATCH --job-name=deploy_sota@taein
#SBATCH --output=logs/deployment/slurm_%x_%j.log
#SBATCH --error=logs/deployment/slurm_%x_%j.log
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

# SOTA ONNX export for Jetson
sh V_experiments/deployment/export_onnx_jetson.sh
