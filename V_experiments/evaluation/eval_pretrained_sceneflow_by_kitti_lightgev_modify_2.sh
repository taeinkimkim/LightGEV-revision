### In-domain Evaluation  (LightGEVModify)
# ------------------------------------------------------------ [KITTI Dataset] ----------------------------------------------------
CKPT_ROOT=train_outputs/SceneFlowDataset/LightGEVModify_2
EXTRA_TAG=pretrain_trainBS4_valBS4_LR0.0004_EP30_modify_2
# EXTRA_TAG=pretrain_trainBS12_valBS12_LR0.0004_modify_2
OUT_ROOT=/media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/kitti_eval/lightgev_modify_2
EPOCH=29

# ---------------------------------------------------------------
# STEP 1 - gate 경로 검증
# ---------------------------------------------------------------
# # Aggregation_only  (gate ON, refiner OFF, r=0.5)
# python tools/eval_single.py \
#   --model_name lightgev_modify_2_pretrained_sceneflow_aggregation_only_bs12_lr0004 \
#   --cfg cfgs/lightgev_modify_2/lightgev_modify_2_eval_kitti_aggregation_only.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_2_train_sceneflow_aggregation_only/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 \
#   --no_npz \
#   --no_vis

# ---------------------------------------------------------------
# STEP 2 - 4칸 ablation 나머지
# ---------------------------------------------------------------
# # No_GEV
# python tools/eval_single.py \
#   --model_name lightgev_modify_2_pretrained_sceneflow_no_gev_bs12_lr0004 \
#   --cfg cfgs/lightgev_modify_2/lightgev_modify_2_eval_kitti_no_gev.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_2_train_sceneflow_no_gev/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 \
#   --no_npz \
#   --no_vis

# # Refiner_only  (gate OFF, refiner ON)
# python tools/eval_single.py \
#   --model_name lightgev_modify_2_pretrained_sceneflow_refiner_only_bs4_lr0004_ep30 \
#   --cfg cfgs/lightgev_modify_2/lightgev_modify_2_eval_kitti_refiner_only.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_2_train_sceneflow_refiner_only/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 \
#   --no_npz \
#   --no_vis

# Full_iter2  (gate ON, refiner ON)
python tools/eval_single.py \
  --model_name lightgev_modify_2_pretrained_sceneflow_full_iter_2_bs4_lr0004 \
  --cfg cfgs/lightgev_modify_2/lightgev_modify_2_eval_kitti_full_iter_2.yaml \
  --ckpt ${CKPT_ROOT}/lightgev_modify_2_train_sceneflow_full_iter_2/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
  --output_root ${OUT_ROOT} \
  --device cuda:0 \
  --no_npz \
  --no_vis

# # Full_iter4  (gate ON, refiner ON)
# python tools/eval_single.py \
#   --model_name lightgev_modify_2_pretrained_sceneflow_full_iter_4_bs12_lr0004 \
#   --cfg cfgs/lightgev_modify_2/lightgev_modify_2_eval_kitti_full_iter_4.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_2_train_sceneflow_full_iter_4/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 \
#   --no_npz \
#   --no_vis

# # Full_iter6  (gate ON, refiner ON)
# python tools/eval_single.py \
#   --model_name lightgev_modify_2_pretrained_sceneflow_full_iter_6_bs12_lr0004 \
#   --cfg cfgs/lightgev_modify_2/lightgev_modify_2_eval_kitti_full_iter_6.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_2_train_sceneflow_full_iter_6/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 \
#   --no_npz \
#   --no_vis

# ---------------------------------------------------------------
# STEP 3 - gate 변조 범위 r 스윕 (r=0.5 는 aggregation_only 와 동일)
# ---------------------------------------------------------------
# # r = 0.25
# python tools/eval_single.py \
#   --model_name lightgev_modify_2_pretrained_sceneflow_agg_only_r_025_bs12_lr0004 \
#   --cfg cfgs/lightgev_modify_2/lightgev_modify_2_eval_kitti_agg_only_r_025.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_2_train_sceneflow_agg_only_r_025/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 \
#   --no_npz \
#   --no_vis

# # r = 0.75
# python tools/eval_single.py \
#   --model_name lightgev_modify_2_pretrained_sceneflow_agg_only_r_075_bs12_lr0004 \
#   --cfg cfgs/lightgev_modify_2/lightgev_modify_2_eval_kitti_agg_only_r_075.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_2_train_sceneflow_agg_only_r_075/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 \
#   --no_npz \
#   --no_vis

## Comparison
# 같은 output_root 에 LightStereo / 원본 LightGEV 결과가 이미 있으므로 바로 비교 가능
# python tools/panel_compare.py \
#   --input_root ${OUT_ROOT} \
#   --output ${OUT_ROOT}/_compare/ \
#   --models lightstereo_pretrained_sceneflow_bs12_lr0004 \
#            lightgev_pretrained_sceneflow_full_iter_4_bs12_lr0004_fixed \
#            lightgev_modify_2_pretrained_sceneflow_full_iter_4_bs12_lr0004
