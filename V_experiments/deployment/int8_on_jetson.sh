# =================================================================================
# Jetson Orin Nano : INT8 엔진 빌드 + 측정 + FP16 대비 정확도 비교
#
#   1) build : train split 으로 calibration 해서 INT8 엔진을 만든다
#   2) bench : latency / FPS / 메모리
#   3) eval  : SceneFlow test 로 region-wise EPE, **FP16 엔진과 직접 비교**
#
# run_on_jetson.sh 와 나눠 둔 이유: INT8 은 trtexec 만으로 안 된다. trtexec 에
# --int8 만 주면 dynamic range 를 몰라 양자화 못 한 레이어가 FP32 로 남아
# "FP32+INT8" 혼합이 된다. calibration cache 는 파이썬 API 로만 만들 수 있다.
#
# --- 먼저 알아 두실 것 ---
#
#  * calibration 에는 **train split** 을 쓴다. test 로 하면 test 정보가 양자화
#    파라미터에 새어 들어가 수치가 낙관적으로 나온다. 스크립트가 막아 둔다.
#
#  * "INT8 만" 은 안 된다. TensorRT 가 레이어마다 가장 빠른 정밀도를 고르므로
#    INT8 커널이 없는 레이어는 FP16/FP32 로 남는다. 목표는 최대한 INT8 이다.
#
#  * 이 모델은 192 단계 cost volume 에 softmax 를 걸어 soft-argmin 을 하므로
#    INT8 의 좁은 동적 범위가 disparity 오차로 바로 증폭된다. **속도가 빨라져도
#    정확도를 반드시 확인할 것.** 기준선은 아래와 같다 (데스크탑, 전체 4370 장):
#
#      PyTorch FP32 == ONNX FP32 : global 1.1041  texless 0.9877  edge 1.8651
#                                  thinQ 2.5238   thickQ 0.7083
#
#    FP16 은 데스크탑에서 변환 손실 0 이었다. INT8 이 global 1.2 를 넘으면
#    속도 이득과 무관하게 쓰기 어려운 수준으로 본다.
#
# --- 경로 ---
#   run_on_jetson.sh 와 같은 SSD 변수를 쓴다. 필요하면 덮어쓸 것:
#     SSD=/mnt/ssd/LightGEV bash V_experiments/deployment/int8_on_jetson.sh
#
#   주의: calibration 에 **train npz 가 필요하다**. test 만 옮기셨다면
#   train 앞 몇백 장을 추가로 복사하거나 CALIB_LIST 를 val 로 바꿀 것.
# =================================================================================

set -u

SSD=${SSD:-/media/mines/PortableSSD_taein/TaeinKim/LightGEV}

ONNX_DIR=${ONNX_DIR:-${SSD}/outputs/onnx_export}
DATA_ROOT=${DATA_ROOT:-${SSD}/SceneFlow}
ENGINE_DIR=${ENGINE_DIR:-${SSD}/outputs/engines}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}

CALIB_LIST=${CALIB_LIST:-data/SceneFlow/sceneflow_train_npz.txt}
NUM_CALIB=${NUM_CALIB:-200}
# INT8 이 안 되는 레이어를 FP32 대신 FP16 으로. 거의 항상 이게 낫다.
INT8_EXTRA=${INT8_EXTRA:---fp16}

LIMIT=${LIMIT:-0}        # 0 = 전체 4370 장
WARMUP=${WARMUP:-50}
RUNS=${RUNS:-200}

STEPS=${STEPS:-build,bench,eval}
FORCE_BUILD=${FORCE_BUILD:-0}
has_step () { case ",${STEPS}," in *",$1,"*) return 0;; *) return 1;; esac; }

# MODELS="lightgev_ours_544x960 lightstereo_m_544x960 banet_2d_544x960"
MODELS="lightstereo_m_544x960"

# ---------------------------------------------------------------------------
# 0) 사전 점검
# ---------------------------------------------------------------------------
fail=0
note () { echo "  [FAIL] $*"; fail=1; }

echo "=============================================================="
echo " 경로 점검 (INT8)"
echo "=============================================================="
echo "  ONNX_DIR   = ${ONNX_DIR}"
echo "  ENGINE_DIR = ${ENGINE_DIR}"
echo "  DATA_ROOT  = ${DATA_ROOT}"
echo "  CALIB_LIST = ${CALIB_LIST}  (앞 ${NUM_CALIB} 장)"
echo ""

for M in ${MODELS}; do
    [ -f "${ONNX_DIR}/${M}.onnx" ] || note "ONNX 없음: ${ONNX_DIR}/${M}.onnx"
done

[ -f "${CALIB_LIST}" ] || note "calibration 목록이 없다: ${CALIB_LIST} (레포 루트에서 실행 중인가?)"

case "$(basename "${CALIB_LIST}")" in
    *test*) note "calibration 에 test split 을 쓰려 한다: ${CALIB_LIST}"
            echo "         train 을 쓸 것: CALIB_LIST=data/SceneFlow/sceneflow_train_npz.txt";;
esac

# calibration 용 데이터가 실제로 있는지 (test 만 옮긴 경우를 잡는다)
if [ -f "${CALIB_LIST}" ]; then
    FIRST=$(head -1 "${CALIB_LIST}" | awk '{print $1}')
    if [ ! -f "${DATA_ROOT}/${FIRST}" ]; then
        note "calibration 데이터가 없다: ${DATA_ROOT}/${FIRST}"
        echo "         test 만 SSD 에 옮기셨다면 train 도 일부 복사하거나"
        echo "         CALIB_LIST=data/SceneFlow/sceneflow_val_npz.txt 로 바꿀 것."
    fi
fi

[ -d "${DATA_ROOT}/FlyingThings3D_npz/test" ] || \
    note "평가용 test 데이터가 없다: ${DATA_ROOT}/FlyingThings3D_npz/test"

mkdir -p "${ENGINE_DIR}" 2>/dev/null
if ! touch "${ENGINE_DIR}/.write_test" 2>/dev/null; then
    note "ENGINE_DIR 에 쓸 수 없다: ${ENGINE_DIR}"
    echo "         확인:  findmnt -T \"${ENGINE_DIR}\"     <- OPTIONS 에 ro 가 있는지"
    echo "         외장이 ro 로 잡혔다면:  ENGINE_DIR=\$HOME/engines"
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
# 1) INT8 빌드 (calibration 포함). trtexec 가 아니라 파이썬 API 를 쓴다.
# ---------------------------------------------------------------------------
if has_step build; then
    for M in ${MODELS}; do
        ENG=${ENGINE_DIR}/${M}_int8.engine
        if [ -f "${ENG}" ] && [ "${FORCE_BUILD}" != "1" ]; then
            echo ""
            echo "[skip] 엔진이 이미 있다: ${ENG}   (다시 구우려면 FORCE_BUILD=1)"
            continue
        fi
        echo ""
        echo "=============================================================="
        echo " ${M} : INT8 빌드 (calibration ${NUM_CALIB} 장) - 20 분 이상"
        echo "=============================================================="
        python3 tools/build_trt_int8.py \
            --onnx ${ONNX_DIR}/${M}.onnx \
            --engine ${ENG} \
            --cache ${ENGINE_DIR}/${M}_int8.calib \
            --data-root ${DATA_ROOT} \
            --list ${CALIB_LIST} \
            --num-calib ${NUM_CALIB} \
            ${INT8_EXTRA} \
            2>&1 | tee ${ENGINE_DIR}/${M}_int8_build.log

        if [ ! -f "${ENG}" ]; then
            echo "[FAIL] 엔진이 저장되지 않았다: ${ENG}  (위 로그 확인)"
            exit 1
        fi
        echo "[OK] 저장됨: ${ENG}  ($(du -h "${ENG}" | cut -f1))"
    done
fi

# ---------------------------------------------------------------------------
# 2) latency. FP16 과 같은 조건(warm-up ${WARMUP} 회 / 측정 ${RUNS} 회)으로 잰다.
# ---------------------------------------------------------------------------
if has_step bench; then
    for M in ${MODELS}; do
        ENG=${ENGINE_DIR}/${M}_int8.engine
        [ -f "${ENG}" ] || { echo "[skip] 엔진 없음: ${ENG}"; continue; }
        echo ""
        echo "=============================================================="
        echo " ${M} : INT8 latency (warm-up ${WARMUP} / 측정 ${RUNS})"
        echo "=============================================================="
        python3 tools/profile_trt_runtime.py \
            --engine ${ENG} --warmup ${WARMUP} --runs ${RUNS} \
            --csv ${ENGINE_DIR}/runtime_${M}_int8.csv \
            2>&1 | tee ${ENGINE_DIR}/${M}_int8_bench.log
    done
fi

# ---------------------------------------------------------------------------
# 3) 정확도. INT8 을 FP16 엔진과 **같은 프레임에서** 직접 비교한다.
#    FP16 엔진이 없으면 INT8 단독 수치만 내고, 데스크탑 기준선과 비교하면 된다.
# ---------------------------------------------------------------------------
if has_step eval; then
    for M in ${MODELS}; do
        ENG=${ENGINE_DIR}/${M}_int8.engine
        FP16=${ENGINE_DIR}/${M}_fp16.engine
        [ -f "${ENG}" ] || { echo "[skip] 엔진 없음: ${ENG}"; continue; }

        echo ""
        echo "=============================================================="
        echo " ${M} : INT8 정확도 ($([ ${LIMIT} -eq 0 ] && echo '전체 4370' || echo ${LIMIT}) 장)"
        echo "=============================================================="
        if [ -f "${FP16}" ]; then
            echo " FP16 엔진과 직접 비교한다."
            python3 tools/eval_sceneflow_engine.py \
                --backend trt --engine ${ENG} \
                --compare trt --compare-engine ${FP16} \
                --data-root ${DATA_ROOT} --limit ${LIMIT} \
                --csv ${ENGINE_DIR}/eval_${M}_int8_vs_fp16.csv \
                2>&1 | tee ${ENGINE_DIR}/eval_${M}_int8.log
        else
            echo " [참고] FP16 엔진이 없어 INT8 단독으로만 낸다: ${FP16}"
            echo "        데스크탑 기준선(global 1.1041)과 비교할 것."
            python3 tools/eval_sceneflow_engine.py \
                --backend trt --engine ${ENG} \
                --data-root ${DATA_ROOT} --limit ${LIMIT} \
                --csv ${ENGINE_DIR}/eval_${M}_int8.csv \
                2>&1 | tee ${ENGINE_DIR}/eval_${M}_int8.log
        fi
    done

    echo ""
    echo "=============================================================="
    echo " 판단 기준 (데스크탑 FP32, 전체 4370 장)"
    echo "=============================================================="
    echo "   global 1.1041 | texless 0.9877 | edge 1.8651 | thinQ 2.5238 | thickQ 0.7083"
    echo "   INT8 global 이 1.2 를 넘으면 속도 이득과 무관하게 쓰기 어렵다고 본다."
fi

# ---------------------------------------------------------------------------
# 참고
# ---------------------------------------------------------------------------
# calibration cache 를 trtexec 에서 재사용 (다시 calibration 하지 않는다)
#   ${TRTEXEC} --onnx=${ONNX_DIR}/lightgev_ours_544x960.onnx \
#       --saveEngine=/tmp/a.engine --int8 --fp16 \
#       --calib=${ENGINE_DIR}/lightgev_ours_544x960_int8.calib
#
# 전력/발열
#   sudo tegrastats --interval 1000
