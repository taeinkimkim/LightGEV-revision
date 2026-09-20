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
#   기본값은 이 Jetson 의 실제 경로다(외장 SSD 자동 마운트):
#     SSD=/media/mines/PortableSSD_taein/TaeinKim/LightGEV
#   마운트 위치가 바뀌면 SSD 하나만 덮어쓰면 된다:
#
#     SSD=/mnt/ssd/LightGEV bash V_experiments/deployment/run_on_jetson.sh
#
#   `lsblk -f` 나 `findmnt -T <경로>` 로 확인할 것.
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

SSD=${SSD:-/media/mines/PortableSSD_taein/TaeinKim/LightGEV}

ONNX_DIR=${ONNX_DIR:-${SSD}/outputs/onnx_export}
DATA_ROOT=${DATA_ROOT:-${SSD}/SceneFlow}
# 주의: 엔진을 외장 SSD 에 두면 exFAT 이 ro 로 마운트됐을 때 21분 걸린 빌드를
# 저장하지 못하고 날린다(실제로 한 번 겪었다). 아래 사전 점검이 실제 쓰기 시험을
# 하므로 그 경우 시작 전에 멈춘다. 안전하게 가려면 ENGINE_DIR=$HOME/engines.
ENGINE_DIR=${ENGINE_DIR:-${SSD}/outputs/engines}
LIMIT=${LIMIT:-0}                        # 0 = 전체 4370 장 (기존 표와 같은 조건)
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
PRECISION_FLAG=${PRECISION_FLAG:---fp16} # BF16 을 보려면 --bf16 (TRT 10.x 이상)
PREC_TAG=$(echo "${PRECISION_FLAG}" | tr -d '-')   # 파일 이름용: --fp16 -> fp16

# trtexec 플래그. 빌드와 측정이 분리됐으므로 각각에 맞는 것만 준다.
#   --monitorMemory 는 빌드 중 메모리 추적용이라 build 쪽에.
#   warmUp/iterations/avgRuns/duration 은 측정용이라 bench 쪽에.
# 측정은 엔진을 불러오기만 하면 되어 빠르니, 수치를 안정시키려면 늘려도 부담이 적다
# (10 회는 꼬리 분위수를 보기엔 적다. 200 정도면 충분하다).
BUILD_FLAGS=${BUILD_FLAGS:---monitorMemory}

# latency 측정 조건. 데스크탑 model_runtime_sota_compare.sh 와 맞춘다.
WARMUP=${WARMUP:-50}     # 횟수
RUNS=${RUNS:-200}        # 횟수

# trtexec 로도 교차 확인한다. 주의: --warmUp 은 **밀리초**라 "50 회" 를 직접 줄 수
# 없다. 1 회가 대략 WARMUP_MS/50 ms 보다 빠르면 50 회 이상 돌아간다는 뜻으로 넉넉히 잡는다.
# --duration=0 을 줘야 --iterations 횟수만큼만 돈다(기본은 최소 3 초를 채운다).
WARMUP_MS=${WARMUP_MS:-5000}
# avgRuns 는 "N 회 평균" 보고 단위라 iterations 와 같게 두면 요약이 한 줄뿐이다. 10 으로 둔다.
BENCH_FLAGS=${BENCH_FLAGS:---warmUp=${WARMUP_MS} --iterations=${RUNS} --avgRuns=10 --duration=0}

MODELS="lightgev_ours_544x960 lightstereo_m_544x960"

# 단계를 나눠서 실행한다. 엔진 빌드가 Orin Nano 에서 모델당 20분 넘게 걸리므로,
# 뒷단(측정/평가)에서 실패했을 때 빌드부터 다시 하지 않도록 분리했다.
#
#   STEPS=build        엔진만 굽는다
#   STEPS=bench        이미 구운 엔진으로 latency 만 잰다 (--loadEngine)
#   STEPS=eval         이미 구운 엔진으로 정확도만 잰다
#   STEPS=build,bench,eval   (기본) 전부
#
# 이미 엔진 파일이 있으면 build 는 건너뛴다. 다시 굽고 싶으면 FORCE_BUILD=1.
STEPS=${STEPS:-build,bench,eval}
FORCE_BUILD=${FORCE_BUILD:-0}

has_step () { case ",${STEPS}," in *",$1,"*) return 0;; *) return 1;; esac; }

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

mkdir -p "${ENGINE_DIR}" 2>/dev/null
if ! touch "${ENGINE_DIR}/.write_test" 2>/dev/null; then
    note "ENGINE_DIR 에 쓸 수 없다: ${ENGINE_DIR}"
    echo "         (mkdir -p 는 디렉토리가 이미 있으면 read-only 에서도 성공하므로"
    echo "          여기서 실제로 파일을 만들어 본다. 21분짜리 빌드를 날리지 않으려면 필수)"
    echo "         확인:  findmnt -T \"${ENGINE_DIR}\"     <- OPTIONS 에 ro 가 있는지"
    echo "         외장이 ro 로 잡혔다면 내장 스토리지를 쓸 것:  ENGINE_DIR=\$HOME/engines"
else
    rm -f "${ENGINE_DIR}/.write_test"
fi

if [ "${fail}" -ne 0 ]; then
    echo ""
    echo "위 문제를 고치고 다시 실행할 것."
    exit 1
fi
echo "  점검 통과"

# ---------------------------------------------------------------------------
# 1) 엔진 빌드 (--buildOnly). 측정은 여기서 하지 않는다.
#
#    --timingCacheFile 을 주면 두 번째 빌드부터 크게 빨라진다. 첫 빌드가
#    1255 초(21분) 걸렸으므로 다시 구울 일이 있으면 반드시 쓸 것.
# ---------------------------------------------------------------------------
if has_step build; then
    for M in ${MODELS}; do
        ENG=${ENGINE_DIR}/${M}_${PREC_TAG}.engine
        if [ -f "${ENG}" ] && [ "${FORCE_BUILD}" != "1" ]; then
            echo ""
            echo "[skip] 엔진이 이미 있다: ${ENG}   (다시 구우려면 FORCE_BUILD=1)"
            continue
        fi
        echo ""
        echo "=============================================================="
        echo " ${M} : 엔진 빌드 (${PRECISION_FLAG})  - 모델당 20분 이상 걸린다"
        echo "=============================================================="
        ${TRTEXEC} \
            --onnx=${ONNX_DIR}/${M}.onnx \
            --saveEngine=${ENG} \
            --timingCacheFile=${ENGINE_DIR}/timing_${PREC_TAG}.cache \
            ${PRECISION_FLAG} \
            --buildOnly ${BUILD_FLAGS} \
            2>&1 | tee ${ENGINE_DIR}/${M}_${PREC_TAG}_build.log

        if [ ! -f "${ENG}" ]; then
            echo "[FAIL] 엔진이 저장되지 않았다: ${ENG}"
            echo "       빌드 로그 끝부분을 확인할 것 (쓰기 권한 문제면 위 점검에서 걸렸어야 한다)."
            exit 1
        fi
        echo "[OK] 저장됨: ${ENG}  ($(du -h "${ENG}" | cut -f1))"
    done
fi

# ---------------------------------------------------------------------------
# 2) 순수 latency. 이미 구운 엔진을 불러 재기만 한다 (빌드를 다시 하지 않는다).
#
#    두 가지로 잰다:
#      (a) tools/profile_trt_runtime.py - 데스크탑과 **같은 조건**.
#          warm-up/측정을 횟수로 지정하고, H2D/D2H 복사를 빼고 실행만 잰다.
#          peak GPU 메모리와 FPS 도 같이 낸다. 논문 표에 쓸 숫자는 이쪽.
#      (b) trtexec - 교차 확인용. --warmUp 이 ms 단위라 "50 회" 를 못 주지만,
#          "GPU Compute Time" 이 (a) 의 mean 과 비슷해야 한다.
# ---------------------------------------------------------------------------
if has_step bench; then
    for M in ${MODELS}; do
        ENG=${ENGINE_DIR}/${M}_${PREC_TAG}.engine
        if [ ! -f "${ENG}" ]; then
            echo "[skip] 엔진이 없다: ${ENG}   (STEPS=build 로 먼저 구울 것)"
            continue
        fi
        echo ""
        echo "=============================================================="
        echo " ${M} : latency 측정 (warm-up ${WARMUP} 회 / 측정 ${RUNS} 회)"
        echo "=============================================================="

        # (a) 데스크탑과 같은 조건
        python3 tools/profile_trt_runtime.py \
            --engine ${ENG} \
            --warmup ${WARMUP} --runs ${RUNS} \
            --csv ${ENGINE_DIR}/runtime_${M}_${PREC_TAG}.csv \
            2>&1 | tee ${ENGINE_DIR}/${M}_${PREC_TAG}_bench.log

        # (b) trtexec 교차 확인
        echo ""
        echo "--- trtexec 교차 확인 ---"
        ${TRTEXEC} \
            --loadEngine=${ENG} \
            ${BENCH_FLAGS} \
            2>&1 | tee ${ENGINE_DIR}/${M}_${PREC_TAG}_trtexec.log \
            | grep -E "Latency:|Throughput:|GPU Compute Time" || true
    done
fi

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
if has_step eval; then
    echo ""
    echo "=============================================================="
    echo " SceneFlow region-wise 정확도 ($([ ${LIMIT} -eq 0 ] && echo '전체 4370' || echo ${LIMIT}) 장)"
    echo "=============================================================="

    for M in ${MODELS}; do
        ENG=${ENGINE_DIR}/${M}_${PREC_TAG}.engine
        if [ ! -f "${ENG}" ]; then
            echo "[skip] 엔진이 없다: ${ENG}   (STEPS=build 로 먼저 구울 것)"
            continue
        fi
        echo ""
        echo "--- ${M} ---"
        python3 tools/eval_sceneflow_engine.py \
            --backend trt \
            --engine ${ENG} \
            --data-root ${DATA_ROOT} \
            --limit ${LIMIT} \
            --csv ${ENGINE_DIR}/eval_${M}_${PREC_TAG}.csv \
            2>&1 | tee ${ENGINE_DIR}/eval_${M}_${PREC_TAG}.log
    done
fi

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
