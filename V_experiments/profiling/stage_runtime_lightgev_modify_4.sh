# =================================================================================
# LightGEVModify_4 : 구간별 runtime 측정 (aggregation_only, GEV_ITERS 2/4/6/8/10)
#
# 측정 구간 5개 (모델 forward 안의 CUDA event mark 와 1:1):
#   1. backbone_gwc            Backbone + gwc 생성
#   2. geo_encoding_gating     geometry encoding + gating
#   3. aggregation_regression  LightStereo-style Aggregation + disparity regression
#   4. gru_refinement          geometry-aware ConvGRU refinement
#   5. context_upsampling      LightStereo context-aware upsampling
#
# 함께 찍히는 것: input resolution / precision / model size / learnable params /
#                peak inference memory / total latency / FPS
#
# 주의: 이건 **구조만 보는 측정**이라 checkpoint 가 필요 없다. cfg 의 MODEL 블록으로
#       모델을 만들고 랜덤 입력으로 시간을 잰다. 그래서 학습본이 없는 설정
#       (agg_only_r_075) 도 그대로 측정된다. 반대로 말하면 이 수치는 가중치와
#       무관하므로, run 간 학습 편차 문제와도 무관하다.
#
# warm-up / 반복 횟수는 tools/profile_stage_runtime.py 의 기본값(50 / 200)을 쓴다.
# 실측상 warm-up 은 5회면 수렴하고(0회는 첫 호출 때문에 평균이 오염된다),
# 반복은 20회로도 평균이 ±0.2% 안에 들어온다. 기본값은 p95 까지 안정되게 잡은 값.
# =================================================================================

set -u

CFG_ROOT=cfgs/lightgev_modify_4
OUT_ROOT=/media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/stage_runtime/lightgev_modify_4
RESOLUTION=544x960          # SceneFlow 평가 해상도
PRECISION=fp32              # 평가 파이프라인이 amp=False 라 fp32 로 맞춘다
DEVICE=cuda:0

mkdir -p ${OUT_ROOT}

profile () {    # $1 = variant
    local variant=$1
    local cfg=${CFG_ROOT}/lightgev_modify_4_eval_sceneflow_${variant}.yaml

    if [ ! -f "${cfg}" ]; then echo "[skip] cfg 없음: ${cfg}"; return; fi

    echo ""
    echo "##############################################################"
    echo " ${variant}"
    echo "##############################################################"

    python tools/profile_stage_runtime.py \
      --cfg ${cfg} \
      --resolution ${RESOLUTION} \
      --precision ${PRECISION} \
      --device ${DEVICE} \
      --csv ${OUT_ROOT}/stage_runtime_${variant}.csv
}

profile aggregation_only_r_075
profile full_iter_2_r_075
profile full_iter_4_r_075
profile full_iter_6_r_075
profile full_iter_8_r_075
profile full_iter_10_r_075

# variant 별 csv 를 한 파일로 합친다 (헤더는 첫 파일 것만)
SUMMARY=${OUT_ROOT}/stage_runtime_all.csv
head -1 ${OUT_ROOT}/stage_runtime_aggregation_only_r_075.csv | sed 's/^/variant,/' > ${SUMMARY}
for f in ${OUT_ROOT}/stage_runtime_*.csv; do
    case "$f" in *stage_runtime_all.csv) continue;; esac
    v=$(basename "$f" .csv); v=${v#stage_runtime_}
    tail -n +2 "$f" | sed "s/^/${v},/" >> ${SUMMARY}
done
echo ""
echo "merged: ${SUMMARY}"
column -s, -t ${SUMMARY}
