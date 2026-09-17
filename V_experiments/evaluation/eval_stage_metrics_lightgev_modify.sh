### Stage-wise Evaluation  (D0 / DT / final)
# ------------------------------------------------------------------------------------------------
# eval_single.py 는 최종 예측만 리포트한다. 이 스크립트는 tools/eval_stage_metrics.py 로
# 파이프라인을 단계별로 끊어서 잰다.
#
#   D0    aggregation 직후  -> gate 의 담당 영역.   D1(outlier) 로 판정한다.
#   iterN refiner 반복 단계별
#   DT    refiner 통과 후    -> refiner 의 담당 영역. EPE(서브픽셀) 로 판정한다.
#   final context upsample 까지 통과한 최종 예측
#
# D0/DT/iterN 은 전부 bilinear x4 로 올려 비교하므로 context upsampler 가 교란변수가 되지 않는다.
# cost_agg 에 hook 을 걸어 D0 를 계산하므로 LightStereo / LightGEV / LightGEVModify 가
# 동일 기준으로 비교된다.
#
# [aggregation_only 통과 기준]  D0 의 D1 이 LightStereo 아래로 내려가야 한다.
#   LightStereo   SceneFlow D0 D1  9.11 / KITTI D0 D1 22.11
#   LightGEV 원본  KITTI D0 EPE 6.2011, D0 D1 34.53  (LightStereo 3.1088 / 22.11 보다 크게 나쁨)
#
# NOTE: 절대값은 eval_single.py 와 다르다. eval_single 은 GT 를 패딩 캔버스에서 원본 크기로
#       cv2.INTER_LINEAR 리사이즈한 뒤 평가하고(single_helpers/disp_depth_ops.py:10),
#       이 도구는 로더가 준 텐서 위에서 그대로 평가한다. 스테이지 간 / 모델 간 비교에 쓰고,
#       논문 표의 절대값은 eval_single.py 쪽을 쓸 것.

CKPT_ROOT=train_outputs/SceneFlowDataset/LightGEVModify
EXTRA_TAG=pretrain_trainBS12_valBS12_LR0.0004_modify
OUT_KITTI=/media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/stage_eval_kitti
OUT_SF=/media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/stage_eval_sceneflow
EPOCH=89

stage_kitti () {  # $1=model_name  $2=cfg  $3=ckpt
    python tools/eval_stage_metrics.py \
        --model_name "$1" --cfg "$2" --ckpt "$3" \
        --output_root ${OUT_KITTI} --device cuda:0
}

stage_sf () {     # $1=model_name  $2=cfg  $3=ckpt
    # SceneFlow 는 --gt_mask in_range 를 쓴다.
    # gt>0 (기본값) 로 재면 disparity >= 192 인 픽셀(전체의 약 1.4%)이 포함되는데,
    # 모델은 192 이상을 예측할 수 없으므로 그 픽셀들의 거대한 오차가 EPE 를 지배해서
    # 모델 간 차이가 전부 묻혀버린다 (세 모델이 EPE 3.89~3.91 로 구분 불가).
    python tools/eval_stage_metrics.py \
        --model_name "$1" --cfg "$2" --ckpt "$3" --gt_mask in_range \
        --output_root ${OUT_SF} --device cuda:0
}

# ================================================================================================
# 0. 기준선 (KITTI 34장이라 금방 끝난다. 한 번만 돌려두면 된다)
# ================================================================================================
stage_kitti lightstereo_baseline \
    cfgs/lightstereo/lightstereo_m_eval_kitti.yaml \
    train_outputs/SceneFlowDataset/LightStereo/lightstereo_m_train_sceneflow/pretrain_trainBS12_valBS12_LR0.0004/ckpt/checkpoint_epoch_89.pth

stage_kitti lightgev_original_full_iter_4 \
    cfgs/lightgev/lightgev_eval_kitti_full_iter_4.yaml \
    train_outputs/SceneFlowDataset/LightGEV/lightgev_train_sceneflow_full_iter_4/pretrain_trainBS12_valBS12_LR0.0004_fixed/ckpt/checkpoint_epoch_89.pth

# # SceneFlow 기준선 (test 전체 4370장 - 오래 걸린다)
# stage_sf lightstereo_baseline \
#     cfgs/lightstereo/lightstereo_m_eval_sceneflow.yaml \
#     train_outputs/SceneFlowDataset/LightStereo/lightstereo_m_train_sceneflow/pretrain_trainBS12_valBS12_LR0.0004/ckpt/checkpoint_epoch_89.pth
#
# stage_sf lightgev_original_full_iter_4 \
#     cfgs/lightgev/lightgev_eval_sceneflow_full_iter_4.yaml \
#     train_outputs/SceneFlowDataset/LightGEV/lightgev_train_sceneflow_full_iter_4/pretrain_trainBS12_valBS12_LR0.0004_fixed/ckpt/checkpoint_epoch_89.pth

# ================================================================================================
# STEP 1 - gate 경로 검증 (aggregation_only)
# ================================================================================================
stage_kitti lightgev_modify_aggregation_only \
    cfgs/lightgev_modify/lightgev_modify_eval_kitti_aggregation_only.yaml \
    ${CKPT_ROOT}/lightgev_modify_train_sceneflow_aggregation_only/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth

stage_sf lightgev_modify_aggregation_only \
    cfgs/lightgev_modify/lightgev_modify_eval_sceneflow_aggregation_only.yaml \
    ${CKPT_ROOT}/lightgev_modify_train_sceneflow_aggregation_only/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth

# ================================================================================================
# STEP 2 - 4칸 ablation 나머지
# ================================================================================================
# for V in no_gev refiner_only full_iter_4; do
#     stage_kitti lightgev_modify_${V} \
#         cfgs/lightgev_modify/lightgev_modify_eval_kitti_${V}.yaml \
#         ${CKPT_ROOT}/lightgev_modify_train_sceneflow_${V}/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth
#     stage_sf lightgev_modify_${V} \
#         cfgs/lightgev_modify/lightgev_modify_eval_sceneflow_${V}.yaml \
#         ${CKPT_ROOT}/lightgev_modify_train_sceneflow_${V}/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth
# done

# ================================================================================================
# STEP 3 - gate 변조 범위 r 스윕 (r=0.5 는 aggregation_only 와 동일)
#          로그의 [geo gate] gate_dev_mean 이 "상한 r 대비 실제로 얼마나 썼는가" 를 보여준다.
#          r 에 붙어 포화하면 상한을 올리고 싶어한다는 뜻이다.
# ================================================================================================
# for V in agg_only_r_025 agg_only_r_075; do
#     stage_kitti lightgev_modify_${V} \
#         cfgs/lightgev_modify/lightgev_modify_eval_kitti_${V}.yaml \
#         ${CKPT_ROOT}/lightgev_modify_train_sceneflow_${V}/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth
#     stage_sf lightgev_modify_${V} \
#         cfgs/lightgev_modify/lightgev_modify_eval_sceneflow_${V}.yaml \
#         ${CKPT_ROOT}/lightgev_modify_train_sceneflow_${V}/${EXTRA_TAG}/ckpt/checkpoint_epoch_${EPOCH}.pth
# done
