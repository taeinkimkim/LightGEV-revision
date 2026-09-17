### Zero-shot Evaluation  (LightGEVModify)
# ------------------------------------------------------------ [KITTI Dataset] ------------------------------------------------------
CKPT_ROOT=train_outputs/SceneFlowDataset/LightGEVModify
# EXTRA_TAG=pretrain_trainBS4_valBS4_LR0.0004_EP30_modify
# EXTRA_TAG=pretrain_trainBS12_valBS12_LR0.0004_modify
# EXTRA_TAG=pretrain_trainBS12_valBS12_LR0.0004_modify_gate_normT_biasF_same_loss
EXTRA_TAG=pretrain_trainBS12_valBS12_LR0.0004_modify_gate_normT_biasF
OUT_ROOT=/media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/kitti_eval/lightgev_modify
EPOCH=89

# ---------------------------------------------------------------
# STEP 1 - gate 경로 검증
# ---------------------------------------------------------------
# r = 0.25
python tools/eval_single.py \
  --model_name lightgev_modify_pretrained_sceneflow_agg_only_r_025_normT_biasF_bs12_lr0004_valid_GT_revised \
  --cfg cfgs/lightgev_modify/lightgev_modify_eval_kitti_agg_only_r_025.yaml \
  --ckpt ${CKPT_ROOT}/lightgev_modify_train_sceneflow_agg_only_r_025/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
  --output_root ${OUT_ROOT} \
  --device cuda:0

# r = 0.75
python tools/eval_single.py \
  --model_name lightgev_modify_pretrained_sceneflow_agg_only_r_075_normT_biasF_bs12_lr0004_valid_GT_revised \
  --cfg cfgs/lightgev_modify/lightgev_modify_eval_kitti_agg_only_r_075.yaml \
  --ckpt ${CKPT_ROOT}/lightgev_modify_train_sceneflow_agg_only_r_075/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
  --output_root ${OUT_ROOT} \
  --device cuda:0

# Aggregation_only  (gate ON, refiner OFF, r=0.5)
python tools/eval_single.py \
  --model_name lightgev_modify_pretrained_sceneflow_aggregation_only_normT_biasF_bs12_lr0004_valid_GT_revised \
  --cfg cfgs/lightgev_modify/lightgev_modify_eval_kitti_aggregation_only.yaml \
  --ckpt ${CKPT_ROOT}/lightgev_modify_train_sceneflow_aggregation_only/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
  --output_root ${OUT_ROOT} \
  --device cuda:0

# # Aggregation_only  (gate ON, refiner OFF, r=0.5, dilations=2)
# python tools/eval_single.py \
#   --model_name lightgev_modify_pretrained_sceneflow_aggregation_only_dil_2_normT_biasF_bs12_lr0004_same_loss_valid_GT_revised \
#   --cfg cfgs/lightgev_modify/lightgev_modify_eval_kitti_aggregation_only_dil_2.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_train_sceneflow_aggregation_only_dil_2/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0

# # Aggregation_only  (gate ON, refiner OFF, r=0.5, dilations=1)
# python tools/eval_single.py \
#   --model_name lightgev_modify_pretrained_sceneflow_aggregation_only_dil_1_normT_biasF_bs12_lr0004_same_loss_valid_GT_revised \
#   --cfg cfgs/lightgev_modify/lightgev_modify_eval_kitti_aggregation_only_dil_1.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_train_sceneflow_aggregation_only_dil_1/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0

# ---------------------------------------------------------------
# STEP 2 - 4칸 ablation 나머지
# ---------------------------------------------------------------
# # No_GEV
# python tools/eval_single.py \
#   --model_name lightgev_modify_pretrained_sceneflow_no_gev_bs12_lr0004 \
#   --cfg cfgs/lightgev_modify/lightgev_modify_eval_kitti_no_gev.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_train_sceneflow_no_gev/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0

# # Refiner_only  (gate OFF, refiner ON)
# python tools/eval_single.py \
#   --model_name lightgev_modify_pretrained_sceneflow_refiner_only_bs12_lr0004 \
#   --cfg cfgs/lightgev_modify/lightgev_modify_eval_kitti_refiner_only.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_train_sceneflow_refiner_only/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0

# # Full_iter2  (gate ON, refiner ON)
# python tools/eval_single.py \
#   --model_name lightgev_modify_pretrained_sceneflow_full_iter_2_bs12_lr0004 \
#   --cfg cfgs/lightgev_modify/lightgev_modify_eval_kitti_full_iter_2.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_train_sceneflow_full_iter_2/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0

# # Full_iter4  (gate ON, refiner ON)
# python tools/eval_single.py \
#   --model_name lightgev_modify_pretrained_sceneflow_full_iter_4_bs12_lr0004 \
#   --cfg cfgs/lightgev_modify/lightgev_modify_eval_kitti_full_iter_4.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_train_sceneflow_full_iter_4/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0

# # Full_iter6  (gate ON, refiner ON)
# python tools/eval_single.py \
#   --model_name lightgev_modify_pretrained_sceneflow_full_iter_6_bs12_lr0004 \
#   --cfg cfgs/lightgev_modify/lightgev_modify_eval_kitti_full_iter_6.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_train_sceneflow_full_iter_6/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0

##  Comparison
# LightStereo vs 원본 LightGEV vs LightGEVModify
# python tools/panel_compare.py \
#   --input_root ${OUT_ROOT} \
#   --output ${OUT_ROOT}/_compare/ \
#   --models lightstereo_pretrained_sceneflow_bs12_lr0004 \
#            lightgev_pretrained_sceneflow_iter4_bs12_lr0004_fixed \
#            lightgev_modify_pretrained_sceneflow_full_iter_4_bs12_lr0004
