# =================================================================================
# Jetson Orin Nano 에서 실행하는 스크립트 (데스크탑이 아니다)
#
#   1) ONNX -> TensorRT FP16 엔진 빌드
#   2) trtexec 로 순수 latency 측정
#   3) SceneFlow test 로 정확도 측정 + PyTorch FP32 대비 열화 확인
#
# 대상은 2D 계열 중 LightStereo-M 과 LightGEV (ours) 두 개다.
#
# --- 경로 (전부 환경변수로 덮어쓸 수 있다. 파일을 고칠 필요 없다) ---
#
#   외장 SSD 가 /mnt/ssd 에 마운트돼 있다면:
#
#     DATA_ROOT=/mnt/ssd/SceneFlow \
#     ONNX_DIR=/mnt/ssd/onnx \
#     ENGINE_DIR=/mnt/ssd/engines \
#     bash V_experiments/deployment/run_on_jetson.sh
#
#   자동 마운트면 보통 /media/$USER/<디스크이름> 이다. `lsblk -f` 로 확인할 것.
#   스크립트는 **레포 루트에서** 실행해야 한다 (--list 가 상대 경로다).
#
#   DATA_ROOT 는 FlyingThings3D_npz 를 **담고 있는** 디렉토리다:
#     ${DATA_ROOT}/FlyingThings3D_npz/test/000000.npz
#   목록 파일이 "./FlyingThings3D_npz/test/000000.npz" 로 적혀 있기 때문이다.
#
# --- 옮겨야 할 것 ---
#   * .onnx 파일      : 데스크탑 outputs/onnx_export/ 에서 복사
#   * 이 레포          : tools/ 와 data/SceneFlow/*.txt 가 필요
#   * SceneFlow npz   : **전체 4370 장 / 34GB**. 정확도를 기존 표와 같은 조건으로
#                       내려면 전체가 필요하다. Orin Nano 내장 스토리지로는 부족하니
#                       외장 SSD 를 붙이고 DATA_ROOT 를 거기로 잡을 것.
#                       파이프라인만 먼저 확인하려면 LIMIT 를 잠깐 걸면 된다.
#
# --- FP16 vs BF16 ---
# BF16 은 지수부가 넓은 대신 가수부가 8 비트로, FP16(11 비트)보다 **정밀도가 낮다**.
# 이 모델들은 192 단계 cost volume 에 softmax 를 걸어 soft-argmin 을 하므로 범위가
# 아니라 정밀도가 중요하다. 정규화된 활성값이 FP16 범위(±65504)를 넘을 일도 없다.
# 그래서 여기서는 FP16 을 기본으로 한다. BF16 을 꼭 보려면 --bf16 을 쓰되
# (TensorRT 10.x / JetPack 6.x 이상에서만 지원), 아래 3) 로 정확도를 반드시 비교할 것.
# =================================================================================

set -u

ONNX_DIR=${ONNX_DIR:-$HOME/onnx}
ENGINE_DIR=${ENGINE_DIR:-$HOME/engines}
DATA_ROOT=${DATA_ROOT:-$HOME/SceneFlow}
LIMIT=${LIMIT:-0}                        # 0 = 전체 4370 장 (기존 표와 같은 조건)
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
PRECISION_FLAG=${PRECISION_FLAG:---fp16} # BF16 을 보려면 --bf16 (TRT 10.x 이상)

MODELS="lightgev_ours_544x960 lightstereo_m_544x960"

# ---------------------------------------------------------------------------
# 0) 사전 점검. 경로가 틀린 채로 시작하면 34GB 를 읽다가 한참 뒤에 죽는다.
# ---------------------------------------------------------------------------
fail=0
note () { echo "  [FAIL] $*"; fail=1; }

echo "=============================================================="
echo " 경로 점검"
echo "=============================================================="
echo "  ONNX_DIR   = ${ONNX_DIR}"
echo "  ENGINE_DIR = ${ENGINE_DIR}"
echo "  DATA_ROOT  = ${DATA_ROOT}"
echo "  TRTEXEC    = ${TRTEXEC}"
echo ""

[ -x "${TRTEXEC}" ] || note "trtexec 가 없거나 실행 불가: ${TRTEXEC}  (TRTEXEC=... 로 지정)"

for M in ${MODELS}; do
    [ -f "${ONNX_DIR}/${M}.onnx" ] || note "ONNX 없음: ${ONNX_DIR}/${M}.onnx"
done

if [ ! -d "${DATA_ROOT}/FlyingThings3D_npz/test" ]; then
    note "DATA_ROOT 아래에 FlyingThings3D_npz/test 가 없다: ${DATA_ROOT}"
    echo "         DATA_ROOT 는 FlyingThings3D_npz 를 '담고 있는' 디렉토리여야 한다."
else
    n=$(ls "${DATA_ROOT}/FlyingThings3D_npz/test" | wc -l)
    echo "  npz 파일 ${n} 개 발견 (전체 평가에는 4370 개 필요)"
    [ "${n}" -lt 4370 ] && [ "${LIMIT}" -eq 0 ] && \
        echo "  [경고] 4370 개 미만인데 LIMIT=0 이다. 파일이 모자라면 중간에 멈춘다."
fi

[ -f "data/SceneFlow/sceneflow_test_npz.txt" ] || \
    note "목록 파일이 없다. 레포 루트에서 실행하고 있는가?"

mkdir -p "${ENGINE_DIR}" 2>/dev/null || note "ENGINE_DIR 을 만들 수 없다: ${ENGINE_DIR}"

if [ "${fail}" -ne 0 ]; then
    echo ""
    echo "위 문제를 고치고 다시 실행할 것."
    exit 1
fi
echo "  점검 통과"

# ---------------------------------------------------------------------------
# 1) 엔진 빌드 + 2) 순수 latency
#    latency 는 trtexec 가 이미 잘 재 준다 (median/mean/percentile). 별도 도구를
#    만들 필요가 없어서 여기서는 trtexec 출력을 그대로 쓴다.
# ---------------------------------------------------------------------------
for M in ${MODELS}; do
    echo ""
    echo "=============================================================="
    echo " ${M} : 엔진 빌드 (${PRECISION_FLAG})"
    echo "=============================================================="
    ${TRTEXEC} \
        --onnx=${ONNX_DIR}/${M}.onnx \
        --saveEngine=${ENGINE_DIR}/${M}${PRECISION_FLAG}.engine \
        ${PRECISION_FLAG} \
        --iterations=200 --warmUp=2000 --avgRuns=200 \
        2>&1 | tee ${ENGINE_DIR}/${M}${PRECISION_FLAG}_build.log

    echo ""
    echo "--- latency 요약 ---"
    grep -E "Latency:|Throughput:|GPU Compute Time" \
        ${ENGINE_DIR}/${M}${PRECISION_FLAG}_build.log || true
done

# ---------------------------------------------------------------------------
# 3) 정확도: SceneFlow test 전체로 region-wise EPE
#
#    tools/eval_single.py 와 같은 DepthThicknessEvaluator 경로로 재므로, 나오는
#    REGION_SUMMARY 는 기존 results.csv 와 같은 정의다 (30 장 대조에서 npix 완전
#    일치, EPE 소수 4자리 일치 확인).
#
#    --compare torch 를 주면 같은 프레임을 두 백엔드로 돌려 region 별 차이를 직접
#    찍는다. Jetson 에 PyTorch 가 없거나 너무 느리면 --compare 를 빼고, 데스크탑의
#    기존 results.csv REGION_SUMMARY 와 비교하면 된다 (같은 전체 4370 장이므로
#    그대로 비교 가능하다).
# ---------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo " SceneFlow region-wise 정확도 ($([ ${LIMIT} -eq 0 ] && echo '전체 4370' || echo ${LIMIT}) 장)"
echo "=============================================================="

python3 tools/eval_sceneflow_engine.py \
    --backend trt \
    --engine ${ENGINE_DIR}/lightgev_ours_544x960${PRECISION_FLAG}.engine \
    --data-root ${DATA_ROOT} \
    --limit ${LIMIT} \
    --csv ${ENGINE_DIR}/eval_lightgev_ours${PRECISION_FLAG}.csv \
    2>&1 | tee ${ENGINE_DIR}/eval_lightgev_ours${PRECISION_FLAG}.log

python3 tools/eval_sceneflow_engine.py \
    --backend trt \
    --engine ${ENGINE_DIR}/lightstereo_m_544x960${PRECISION_FLAG}.engine \
    --data-root ${DATA_ROOT} \
    --limit ${LIMIT} \
    --csv ${ENGINE_DIR}/eval_lightstereo_m${PRECISION_FLAG}.csv \
    2>&1 | tee ${ENGINE_DIR}/eval_lightstereo_m${PRECISION_FLAG}.log

# ---------------------------------------------------------------------------
# 필요할 때 주석 해제
# ---------------------------------------------------------------------------

# PyTorch FP32 기준값과 나란히 (Jetson 에 torch 가 있고 시간이 있을 때)
# python3 tools/eval_sceneflow_engine.py \
#     --backend trt --engine ${ENGINE_DIR}/lightgev_ours_544x960--fp16.engine \
#     --compare torch \
#     --cfg cfgs/lightgev_modify_4/lightgev_modify_4_eval_sceneflow_full_iter_6_r_075.yaml \
#     --ckpt <checkpoint_epoch_89.pth 경로> \
#     --data-root ${DATA_ROOT} --limit ${LIMIT}

# FP16 vs BF16 정확도 비교 (둘 다 엔진을 구워 둔 뒤)
# python3 tools/eval_sceneflow_engine.py \
#     --backend trt --engine ${ENGINE_DIR}/lightgev_ours_544x960--bf16.engine \
#     --compare trt --compare-engine ${ENGINE_DIR}/lightgev_ours_544x960--fp16.engine \
#     --data-root ${DATA_ROOT} --limit ${LIMIT}

# 전력/발열까지 보려면 별도 터미널에서
#   sudo tegrastats --interval 1000
