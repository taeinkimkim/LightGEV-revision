#!/bin/bash
#SBATCH --job-name=lightgev_modify_4_full_iter6_r075_dil136_normT_biasF_bs12_lr0004_ep90@taein
#SBATCH --output=logs/train/slurm_%x_%j.log
#SBATCH --error=logs/train/slurm_%x_%j.log
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

sh V_experiments/ablation_studies/pretrain_sceneflow_ablation_lightgev_modify_4.sh
# sh V_experiments/training/pretrain_sceneflow.sh