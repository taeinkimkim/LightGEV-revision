#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDNN_BENCHMARK=True
export TORCHAMP_USE_FP16_REDUCE=True
export PYTHONUNBUFFERED=1

CFG_DIR=cfgs/lightgev_modify
EXTRA_TAG='pretrain_trainBS12_valBS12_LR0.0004_modify_gate_normT_biasF_same_loss'

run () {
    python tools/train.py \
        --cfg_file="${CFG_DIR}/lightgev_modify_train_sceneflow_$1.yaml" \
        --fix_random_seed \
        --save_root_dir=train_outputs --extra_tag="${EXTRA_TAG}" \
        --cover_old_exp --workers 8 --pin_memory
}

# ---------------------------------------------------------------
# STEP 1 - gate 변조 범위 r 스윕 (r=0.5 는 aggregation_only 와 동일)
#          alpha 와 달리 r 은 tanh 바깥에 있어 conv 가중치가 흡수할 수 없으므로
#          실제로 스윕이 의미를 갖는다.
# ---------------------------------------------------------------
# run agg_only_r_025
# run agg_only_r_075
# run aggregation_only
# run aggregation_only_dil_2
run aggregation_only_dil_1

# ---------------------------------------------------------------
# STEP 2 - iteration 개수
# ---------------------------------------------------------------
# run no_gev
# run refiner_only
# run full_iter_2
# run full_iter_4
# run full_iter_6

# ---------------------------------------------------------------
# STEP 3 - dilation 개수
# ---------------------------------------------------------------
# run full_dil_1
# run full_dil_2