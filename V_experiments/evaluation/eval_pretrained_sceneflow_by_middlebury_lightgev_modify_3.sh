# =================================================================================
# LightGEVModify_3 : lightgev_modify.py(gate 경로 1~3) 에서 refiner 입력만 바꾼 변형
#   - refiner 입력이 geo_feat.mean(dim=2) [B,16,H,W] 가 아니라
#     gate 를 통과한 cost volume gwc_enh [B,48,H,W] 다 (D축을 뭉개지 않음)
#   - gwc_enh 계산이 USE_GEO_COST_GATE 가 아니라 USE_GEV 로 옮겨져서,
#     refiner_only 에서도 refiner 는 gwc_enh 를 받는다.
#     USE_GEO_COST_GATE 는 "aggregation 에도 주입할지" 만 제어한다.
#   - 그 결과 refiner_only 와 full 은 파라미터 수가 같다 (7,771,887)
# =================================================================================
### In-domain Evaluation  (LightGEVModify_3)
# ------------------------------------------------------------ [Middlebury Dataset] ----------------------------------------------------
CKPT_ROOT=train_outputs/SceneFlowDataset/LightGEVModify_3
EXTRA_TAG=pretrain_trainBS12_valBS12_LR0.0004_modify_3_gate_normT_biasF_same_loss
OUT_ROOT=/media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/middlebury_eval/lightgev_modify_3
EPOCH=89

# ---------------------------------------------------------------
# STEP 1 - gate 경로 검증
# ---------------------------------------------------------------
# # Aggregation_only  (gate ON, refiner OFF, r=0.5)
# python tools/eval_single.py \
#   --model_name lightgev_modify_3_pretrained_sceneflow_aggregation_only_bs12_lr0004 \
#   --cfg cfgs/lightgev_modify_3/lightgev_modify_3_eval_middlebury_aggregation_only.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_3_train_sceneflow_aggregation_only/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 

# # Aggregation_only  (gate ON, refiner OFF, r=0.5, dilations_2)
# python tools/eval_single.py \
#   --model_name lightgev_modify_3_pretrained_sceneflow_aggregation_only_dil_2_normT_biasF_bs12_lr0004_same_loss_valid_GT_revised \
#   --cfg cfgs/lightgev_modify_3/lightgev_modify_3_eval_middlebury_aggregation_only_dil_2.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_3_train_sceneflow_aggregation_only_dil_2/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 \
#   --no_npz \
#   --no_vis

# # Aggregation_only  (gate ON, refiner OFF, r=0.5, dilations_1)
# python tools/eval_single.py \
#   --model_name lightgev_modify_3_pretrained_sceneflow_aggregation_only_dil_1_normT_biasF_bs12_lr0004_same_loss_valid_GT_revised \
#   --cfg cfgs/lightgev_modify_3/lightgev_modify_3_eval_middlebury_aggregation_only_dil_1.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_3_train_sceneflow_aggregation_only_dil_1/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 \
#   --no_npz \
#   --no_vis

# ---------------------------------------------------------------
# STEP 2 - 4칸 ablation 나머지
# ---------------------------------------------------------------
# # No_GEV
# python tools/eval_single.py \
#   --model_name lightgev_modify_3_pretrained_sceneflow_no_gev_bs12_lr0004 \
#   --cfg cfgs/lightgev_modify_3/lightgev_modify_3_eval_middlebury_no_gev.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_3_train_sceneflow_no_gev/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 

# # Refiner_only  (gate OFF, refiner ON, iter=2, window=5)
# python tools/eval_single.py \
#   --model_name lightgev_modify_3_pretrained_sceneflow_refiner_only_iter_2_window_5_bs12_lr0004_normT_biasF_same_loss_valid_GT_revised \
#   --cfg cfgs/lightgev_modify_3/lightgev_modify_3_eval_middlebury_refiner_only_iter_2_window_5.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_3_train_sceneflow_refiner_only_iter_2_window_5/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 

# # Refiner_only  (gate OFF, refiner ON, iter=4, window=5)
# python tools/eval_single.py \
#   --model_name lightgev_modify_3_pretrained_sceneflow_refiner_only_iter_4_window_5_bs12_lr0004_normT_biasF_same_loss_valid_GT_revised \
#   --cfg cfgs/lightgev_modify_3/lightgev_modify_3_eval_middlebury_refiner_only_iter_4_window_5.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_3_train_sceneflow_refiner_only_iter_4_window_5/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 

# # Refiner_only  (gate OFF, refiner ON, iter=6, window=9)
# python tools/eval_single.py \
#   --model_name lightgev_modify_3_pretrained_sceneflow_refiner_only_iter_6_window_9_bs12_lr0004_normT_biasF_same_loss_valid_GT_revised \
#   --cfg cfgs/lightgev_modify_3/lightgev_modify_3_eval_middlebury_refiner_only_iter_6_window_9.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_3_train_sceneflow_refiner_only_iter_6_window_9/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 

# # Full_iter2  (gate ON, refiner ON)
# python tools/eval_single.py \
#   --model_name lightgev_modify_3_pretrained_sceneflow_full_iter_2_bs4_lr0004 \
#   --cfg cfgs/lightgev_modify_3/lightgev_modify_3_eval_middlebury_full_iter_2.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_3_train_sceneflow_full_iter_2/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 

# # Full_iter2  (gate ON, refiner ON, window=5)
# python tools/eval_single.py \
#   --model_name lightgev_modify_3_pretrained_sceneflow_full_iter_2_window_5_bs12_lr0004_normT_biasF_same_loss_valid_GT_revised \
#   --cfg cfgs/lightgev_modify_3/lightgev_modify_3_eval_middlebury_full_iter_2_window_5.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_3_train_sceneflow_full_iter_2_window_5/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 

# # Full_iter4  (gate ON, refiner ON)
# python tools/eval_single.py \
#   --model_name lightgev_modify_3_pretrained_sceneflow_full_iter_4_window_9_bs12_lr0004_normT_biasF_same_loss_valid_GT_revised \
#   --cfg cfgs/lightgev_modify_3/lightgev_modify_3_eval_middlebury_full_iter_4.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_3_train_sceneflow_full_iter_4/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 

# # Full_iter4  (gate ON, refiner ON, window=5)
# python tools/eval_single.py \
#   --model_name lightgev_modify_3_pretrained_sceneflow_full_iter_4_window_5_bs12_lr0004_normT_biasF_same_loss_valid_GT_revised \
#   --cfg cfgs/lightgev_modify_3/lightgev_modify_3_eval_middlebury_full_iter_4_window_5.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_3_train_sceneflow_full_iter_4_window_5/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 

# # Full_iter4  (gate ON, refiner ON, window=3)
# python tools/eval_single.py \
#   --model_name lightgev_modify_3_pretrained_sceneflow_full_iter_4_window_3_bs12_lr0004_normT_biasF_same_loss_valid_GT_revised \
#   --cfg cfgs/lightgev_modify_3/lightgev_modify_3_eval_middlebury_full_iter_4_window_3.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_3_train_sceneflow_full_iter_4_window_3/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 

# # Full_iter4  (gate ON, refiner ON, window=1)
# python tools/eval_single.py \
#   --model_name lightgev_modify_3_pretrained_sceneflow_full_iter_4_window_1_bs12_lr0004_normT_biasF_same_loss_valid_GT_revised \
#   --cfg cfgs/lightgev_modify_3/lightgev_modify_3_eval_middlebury_full_iter_4_window_1.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_3_train_sceneflow_full_iter_4_window_1/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 

# Full_iter2  (gate ON, refiner ON, window=3)
python tools/eval_single.py \
  --model_name lightgev_modify_3_pretrained_sceneflow_full_iter_2_window_3_bs12_lr0004_normT_biasF_same_loss_valid_GT_revised \
  --cfg cfgs/lightgev_modify_3/lightgev_modify_3_eval_middlebury_full_iter_2_window_3.yaml \
  --ckpt ${CKPT_ROOT}/lightgev_modify_3_train_sceneflow_full_iter_2_window_3/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
  --output_root ${OUT_ROOT} \
  --device cuda:0

# Full_iter6  (gate ON, refiner ON, window=3)
python tools/eval_single.py \
  --model_name lightgev_modify_3_pretrained_sceneflow_full_iter_6_window_3_bs12_lr0004_normT_biasF_same_loss_valid_GT_revised \
  --cfg cfgs/lightgev_modify_3/lightgev_modify_3_eval_middlebury_full_iter_6_window_3.yaml \
  --ckpt ${CKPT_ROOT}/lightgev_modify_3_train_sceneflow_full_iter_6_window_3/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
  --output_root ${OUT_ROOT} \
  --device cuda:0

# # Full_iter4  (gate ON, refiner ON, window=5, disp_emb=False)
# python tools/eval_single.py \
#   --model_name lightgev_modify_3_pretrained_sceneflow_full_iter_4_window_5_disp_embF_bs12_lr0004_normT_biasF_same_loss_valid_GT_revised \
#   --cfg cfgs/lightgev_modify_3/lightgev_modify_3_eval_middlebury_full_iter_4_window_5_disp_embF.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_3_train_sceneflow_full_iter_4_window_5_disp_embF/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 

# # Full_iter6  (gate ON, refiner ON)
# python tools/eval_single.py \
#   --model_name lightgev_modify_3_pretrained_sceneflow_full_iter_6_bs12_lr0004 \
#   --cfg cfgs/lightgev_modify_3/lightgev_modify_3_eval_middlebury_full_iter_6.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_3_train_sceneflow_full_iter_6/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 

# # Full_iter6  (gate ON, refiner ON, window=5)
# python tools/eval_single.py \
#   --model_name lightgev_modify_3_pretrained_sceneflow_full_iter_6_window_5_bs12_lr0004_normT_biasF_same_loss_valid_GT_revised \
#   --cfg cfgs/lightgev_modify_3/lightgev_modify_3_eval_middlebury_full_iter_6_window_5.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_3_train_sceneflow_full_iter_6_window_5/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 

# ---------------------------------------------------------------
# STEP 3 - gate 변조 범위 r 스윕 (r=0.5 는 aggregation_only 와 동일)
# ---------------------------------------------------------------
# # r = 0.25
# python tools/eval_single.py \
#   --model_name lightgev_modify_3_pretrained_sceneflow_agg_only_r_025_bs12_lr0004 \
#   --cfg cfgs/lightgev_modify_3/lightgev_modify_3_eval_middlebury_agg_only_r_025.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_3_train_sceneflow_agg_only_r_025/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 

# # r = 0.75
# python tools/eval_single.py \
#   --model_name lightgev_modify_3_pretrained_sceneflow_agg_only_r_075_bs12_lr0004 \
#   --cfg cfgs/lightgev_modify_3/lightgev_modify_3_eval_middlebury_agg_only_r_075.yaml \
#   --ckpt ${CKPT_ROOT}/lightgev_modify_3_train_sceneflow_agg_only_r_075/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth \
#   --output_root ${OUT_ROOT} \
#   --device cuda:0 

## Comparison
# 같은 output_root 에 LightStereo / 원본 LightGEV 결과가 이미 있으므로 바로 비교 가능
# python tools/panel_compare.py \
#   --input_root ${OUT_ROOT} \
#   --output ${OUT_ROOT}/_compare/ \
#   --models lightstereo_pretrained_sceneflow_bs12_lr0004 \
#            lightgev_pretrained_sceneflow_full_iter_4_bs12_lr0004_fixed \
#            lightgev_modify_3_pretrained_sceneflow_full_iter_4_bs12_lr0004
