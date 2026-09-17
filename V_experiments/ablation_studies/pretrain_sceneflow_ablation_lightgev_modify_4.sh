#!/usr/bin/env bash
# =================================================================================
# LightGEVModify_4 : GEV_ITERS ablation (full, iter 2 / 4 / 6)
#
# 모델은 lightgev_modify.py(gate 경로 1~3) 에 GeometryEncoding1D 의 branch_ch 고정만
# 더한 것이다. refiner 는 원본 그대로 (cost lookup / disp 정규화 / depthwise 없음,
# 입력은 geo_feat.mean(dim=2)).
#
# 주의: cross-domain 수치는 run 간 편차가 크다 (원본 LightGEV 실측 KITTI D0 EPE
#       1.28 차이). iter 2/4/6 의 차이가 그보다 작으면 seed 를 늘려야 한다.
# =================================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDNN_BENCHMARK=True
export TORCHAMP_USE_FP16_REDUCE=True
export PYTHONUNBUFFERED=1

CFG_DIR=cfgs/lightgev_modify_4
# EXTRA_TAG='pretrain_trainBS12_valBS12_LR0.0004_modify_4_gate_normT_biasF_same_loss'
EXTRA_TAG='pretrain_trainBS12_valBS12_LR0.0004_modify_4_gate_normT_biasF'
SEED=0

run () {   # $1 = variant
    python tools/train.py \
        --cfg_file="${CFG_DIR}/lightgev_modify_4_train_sceneflow_$1.yaml" \
        --fix_random_seed --seed "${SEED}" \
        --save_root_dir=train_outputs --extra_tag="${EXTRA_TAG}" \
        --cover_old_exp --workers 8 --pin_memory
}

# run full_iter_2
# run full_iter_2_r_075
# run full_iter_4_r_075
run full_iter_6_r_075
# run full_iter_8_r_075
# run full_iter_10_r_075
# run full_iter_6_r_075_dil_2
# run full_iter_6_r_075_dil_1
# run refiner_only_iter_6_r_075


# 다중 seed 로 돌리려면 EXTRA_TAG 에 seed 를 붙일 것:
#   for s in 0 1 2; do
#     for v in full_iter_2 full_iter_4 full_iter_6; do
#       python tools/train.py --cfg_file=${CFG_DIR}/lightgev_modify_4_train_sceneflow_${v}.yaml \
#         --fix_random_seed --seed $s --save_root_dir=train_outputs \
#         --extra_tag="${EXTRA_TAG}_seed${s}" --cover_old_exp --workers 8 --pin_memory
#     done
#   done
