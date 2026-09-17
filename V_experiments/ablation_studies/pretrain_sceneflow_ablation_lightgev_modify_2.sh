#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDNN_BENCHMARK=True
export TORCHAMP_USE_FP16_REDUCE=True
export PYTHONUNBUFFERED=1

CFG_DIR=cfgs/lightgev_modify_2
EXTRA_TAG='pretrain_trainBS4_valBS4_LR0.0004_EP30_modify_2'

run () {
    python tools/train.py \
        --cfg_file="${CFG_DIR}/lightgev_modify_2_train_sceneflow_$1.yaml" \
        --fix_random_seed \
        --save_root_dir=train_outputs --extra_tag="${EXTRA_TAG}" \
        --cover_old_exp --workers 8 --pin_memory
}

# ---------------------------------------------------------------
# STEP 1 - gate 경로 검증 (이것만 먼저 돌린다)
# ---------------------------------------------------------------
# run aggregation_only

# ---------------------------------------------------------------
# STEP 2 - 4칸 ablation 나머지 (STEP 1 통과 후 주석 해제)
# ---------------------------------------------------------------
# run no_gev
# run refiner_only
run full_iter_2
# run full_iter_4
# run full_iter_6

# ---------------------------------------------------------------
# STEP 3 - gate 변조 범위 r 스윕 (r=0.5 는 aggregation_only 와 동일)
#          alpha 와 달리 r 은 tanh 바깥에 있어 conv 가중치가 흡수할 수 없으므로
#          실제로 스윕이 의미를 갖는다.
# ---------------------------------------------------------------
# run agg_only_r_025
# run agg_only_r_075
