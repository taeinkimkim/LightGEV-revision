#!/usr/bin/env bash
# =================================================================================
# LightGEVModify_3 : lightgev_modify.py(gate 경로 1~3) 에서 refiner 입력만 바꾼 변형
#   - refiner 입력이 geo_feat.mean(dim=2) [B,16,H,W] 가 아니라
#     gate 를 통과한 cost volume gwc_enh [B,48,H,W] 다 (D축을 뭉개지 않음)
#   - gwc_enh 계산이 USE_GEO_COST_GATE 가 아니라 USE_GEV 로 옮겨져서,
#     refiner_only 에서도 refiner 는 gwc_enh 를 받는다.
#     USE_GEO_COST_GATE 는 "aggregation 에도 주입할지" 만 제어한다.
#   - 그 결과 refiner_only 와 full 은 파라미터 수가 같다 (7,771,887)
# =================================================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDNN_BENCHMARK=True
export TORCHAMP_USE_FP16_REDUCE=True
export PYTHONUNBUFFERED=1

CFG_DIR=cfgs/lightgev_modify_3
EXTRA_TAG='pretrain_trainBS12_valBS12_LR0.0004_modify_3_gate_normT_biasF_same_loss'

run () {
    python tools/train.py \
        --cfg_file="${CFG_DIR}/lightgev_modify_3_train_sceneflow_$1.yaml" \
        --fix_random_seed \
        --save_root_dir=train_outputs --extra_tag="${EXTRA_TAG}" \
        --cover_old_exp --workers 8 --pin_memory
}

# ---------------------------------------------------------------
# STEP 1 - gate 변조 범위 r 스윕 (r=0.5 는 aggregation_only 와 동일)
#          alpha 와 달리 r 은 tanh 바깥에 있어 conv 가중치가 흡수할 수 없으므로
#          실제로 스윕이 의미를 갖는다.
# ---------------------------------------------------------------
# run aggregation_only
# run aggregation_only_dil_2
# run aggregation_only_dil_1
# run agg_only_r_025
# run agg_only_r_075


# ---------------------------------------------------------------
# STEP 2 - 4칸 ablation 나머지 (STEP 1 통과 후 주석 해제)
# ---------------------------------------------------------------
# run no_gev
# run refiner_only_iter_2_window_5
# run refiner_only_iter_4_window_5
# run refiner_only_iter_6_window_5
# run full_iter_2
# run full_iter_4_window_5
# run full_iter_4_window_3
# run full_iter_4_window_1
# run full_iter_2_window_3
run full_iter_6_window_3
# run full_iter_2_window_5
# run full_iter_6_window_5
# run full_iter_4_window_5_disp_embF
# run full_iter_6

