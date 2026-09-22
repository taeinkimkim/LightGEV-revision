# =================================================================================
# 전체 inference runtime 비교 : LightGEVModify_4 (ours) vs 2D / 3D SOTA
#
#   2D : LightGEVModify_4 (full, GEV_ITERS 6, r=0.75), LightStereo-M,
#        BANet-2D, MobileStereoNet-2D
#   3D : GwcNet, ACVNet, IGEV++
#
# 구간별 분해는 하지 않는다 (그건 V_experiments/profiling/stage_runtime_lightgev_modify_4.sh).
# 여기서는 모델당 total latency / FPS / model size / learnable params /
# peak inference memory 만 본다.
#
# 주의 1: checkpoint 가 필요 없다. runtime 은 커널 shape 과 횟수로 정해지고 가중치
#         '값' 과는 무관하다 (실측: 학습본 로드 전후 9.873 vs 9.862 ms, 노이즈 수준).
#         cfg 의 MODEL 블록으로 구조만 만들어 랜덤 입력으로 잰다.
#
# 주의 2: IGEV++ 만 MAX_DISP=768 / VALID_ITERS=32 로, 나머지(MAX_DISP=192)와 조건이
#         다르다. 저자 공식 설정이지만 표에 나란히 실을 때는 각주로 밝힐 것.
#         스크립트가 각 모델의 해당 설정을 표 아래에 같이 찍는다.
#
# 주의 3: 다른 GPU 작업과 겹치면 수치가 흔들린다. 단독으로 돌릴 것.
# =================================================================================

set -u

OUT_ROOT=outputs/model_runtime
RESOLUTION=544x960          # SceneFlow 평가 해상도
PRECISION=fp16              # 평가 파이프라인이 amp=False 라 fp32 로 맞춘다
WARMUP=50
RUNS=200
DEVICE=cuda:0

mkdir -p ${OUT_ROOT}

python tools/profile_model_runtime.py \
    --models "lightgev (ours)" lightstereo_m banet_2d msnet_2d \
    --resolution ${RESOLUTION} \
    --precision ${PRECISION} \
    --warmup ${WARMUP} \
    --runs ${RUNS} \
    --device ${DEVICE} \
    --csv ${OUT_ROOT}/model_runtime_${RESOLUTION}_${PRECISION}.csv \
    2>&1 | tee ${OUT_ROOT}/model_runtime_${RESOLUTION}_${PRECISION}.log

# ---------------------------------------------------------------------------
# 필요할 때 주석 해제
# ---------------------------------------------------------------------------

# KITTI 해상도로도 재기
# python tools/profile_model_runtime.py \
#     --resolution 384x1248 --precision ${PRECISION} \
#     --warmup ${WARMUP} --runs ${RUNS} --device ${DEVICE} \
#     --csv ${OUT_ROOT}/model_runtime_384x1248_${PRECISION}.csv

# 정밀도 비교 (fp32 / amp / fp16 한 번에)
# python tools/profile_model_runtime.py \
#     --resolution ${RESOLUTION} --precision fp32 amp fp16 \
#     --warmup ${WARMUP} --runs ${RUNS} --device ${DEVICE} \
#     --csv ${OUT_ROOT}/model_runtime_precision_sweep.csv

# 일부 모델만
# python tools/profile_model_runtime.py --models gwcnet acvnet igev_pp

# ours 를 다른 설정으로 (iter 2 로 교체)
# python tools/profile_model_runtime.py \
#     --cfg "lightgev (ours)=cfgs/lightgev_modify_4/lightgev_modify_4_eval_sceneflow_full_iter_2_r_075.yaml"

echo ""
echo "saved: ${OUT_ROOT}/model_runtime_${RESOLUTION}_${PRECISION}.csv"
