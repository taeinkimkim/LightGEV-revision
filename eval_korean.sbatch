#!/bin/bash
#SBATCH --job-name=eval_lightgev_modify_4_refiner_only_iter6_r075_final_valid_GT_revised@taein
#SBATCH --output=logs/eval/slurm_%x_%j.log
#SBATCH --error=logs/eval/slurm_%x_%j.log
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

# SOTA
# sh V_experiments/evaluation/eval_pretrained_sceneflow_2D_SOTA.sh
# sh V_experiments/evaluation/eval_pretrained_sceneflow_3D_SOTA.sh

# LightGEV / LightStereo
# sh V_experiments/evaluation/eval_pretrained_sceneflow_by_sceneflow.sh
# sh V_experiments/evaluation/eval_pretrained_sceneflow_by_middlebury.sh
# sh V_experiments/evaluation/eval_pretrained_sceneflow_by_eth3d.sh
# sh V_experiments/evaluation/eval_pretrained_sceneflow_by_kitti.sh

# LightGEVModify
# sh V_experiments/evaluation/eval_pretrained_sceneflow_by_sceneflow_lightgev_modify.sh
# sh V_experiments/evaluation/eval_pretrained_sceneflow_by_middlebury_lightgev_modify.sh
# sh V_experiments/evaluation/eval_pretrained_sceneflow_by_eth3d_lightgev_modify.sh
# sh V_experiments/evaluation/eval_pretrained_sceneflow_by_kitti_lightgev_modify.sh

# LightGEVModify_3
# sh V_experiments/evaluation/eval_pretrained_sceneflow_by_sceneflow_lightgev_modify_3.sh
# sh V_experiments/evaluation/eval_pretrained_sceneflow_by_middlebury_lightgev_modify_3.sh
# sh V_experiments/evaluation/eval_pretrained_sceneflow_by_eth3d_lightgev_modify_3.sh
# sh V_experiments/evaluation/eval_pretrained_sceneflow_by_kitti_lightgev_modify_3.sh

# LightGEVModify_4
# sh V_experiments/evaluation/eval_pretrained_sceneflow_by_sceneflow_lightgev_modify_4.sh
# sh V_experiments/evaluation/eval_pretrained_sceneflow_by_middlebury_lightgev_modify_4.sh
# sh V_experiments/evaluation/eval_pretrained_sceneflow_by_eth3d_lightgev_modify_4.sh
# sh V_experiments/evaluation/eval_pretrained_sceneflow_by_kitti_lightgev_modify_4.sh

# LightGEVModify_4: Middlebury scale variation
sh V_experiments/evaluation/eval_pretrained_sceneflow_by_middlebury_lightgev_modify_4_scale.sh