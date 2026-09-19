# =================================================================================
# ONNX export : Jetson Orin Nano 에서 TensorRT FP16 엔진을 굽기 위한 앞단
#
#   기본 대상 (Orin Nano 에 실제로 올릴 만한 2D 계열)
#     lightgev (ours) / lightstereo_m / banet_2d
#
#   3D 계열(msnet_2d, gwcnet, acvnet, igev_pp)은 아래에 주석으로 남겨 두었다.
#   ONNX 파일이 360~650 MB 이고 GMac 이 130~3145 라, 메모리 8GB 를 CPU 와 공유하는
#   Orin Nano 에서는 엔진 빌드가 실패하거나 초당 1프레임을 밑돌 가능성이 크다.
#
# 주의 1: FP32 로 내보낸다. FP16 변환은 Jetson 에서 trtexec --fp16 이 한다.
# 주의 2: .onnx 는 기기 독립이지만 TensorRT 엔진은 GPU 아키텍처·TRT 버전에 묶인다.
#         엔진 빌드는 반드시 Jetson 에서 할 것. 여기서는 .onnx 만 만든다.
# 주의 3: igev_pp 는 export 는 되지만 **검증에 실패한다** (PyTorch 대비 상대오차 0.76).
#         32회 GRU 루프가 펼쳐지면서 그래프가 깨진다. 이 상태로 엔진을 구우면
#         오류 없이 틀린 disparity 가 나오므로 쓰지 말 것.
# =================================================================================

set -u

OUT_ROOT=/media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/onnx_export
RESOLUTION=544x960      # SceneFlow 평가 해상도. Jetson 에서 쓸 해상도로 맞출 것
OPSET=17                # TensorRT 버전이 낮아 거부하면 16 또는 13 으로 낮춘다
DEVICE=cuda:0

mkdir -p ${OUT_ROOT}

python tools/export_onnx.py \
    --models "lightgev (ours)" lightstereo_m banet_2d msnet_2d \
    --resolution ${RESOLUTION} \
    --opset ${OPSET} \
    --device ${DEVICE} \
    --out-dir ${OUT_ROOT} \
    2>&1 | tee ${OUT_ROOT}/export_${RESOLUTION}_opset${OPSET}.log

echo ""
echo "=================================================================="
echo " 저장 위치: ${OUT_ROOT}"
echo "=================================================================="
ls -lh ${OUT_ROOT}/*.onnx 2>/dev/null

# ---------------------------------------------------------------------------
# 필요할 때 주석 해제
# ---------------------------------------------------------------------------

# 3D 계열까지 전부 (검증에 CPU forward 가 필요해 수십 분 걸린다)
# python tools/export_onnx.py \
#     --resolution ${RESOLUTION} --opset ${OPSET} --device ${DEVICE} \
#     --out-dir ${OUT_ROOT}

# 검증 없이 빠르게 (그래프가 맞는지 확인 안 함 - 배포 전에는 반드시 검증할 것)
# python tools/export_onnx.py \
#     --models "lightgev (ours)" lightstereo_m banet_2d \
#     --resolution ${RESOLUTION} --out-dir ${OUT_ROOT} --no-verify

# KITTI 해상도로
# python tools/export_onnx.py \
#     --models "lightgev (ours)" --resolution 384x1248 --out-dir ${OUT_ROOT}

# 동적 H/W (Jetson 에서 optimization profile 을 따로 줘야 하고 보통 더 느리다)
# python tools/export_onnx.py \
#     --models "lightgev (ours)" --resolution ${RESOLUTION} \
#     --dynamic --out-dir ${OUT_ROOT}

# ---------------------------------------------------------------------------
# 다음 단계 (Jetson Orin Nano 에서)
#
#   1) .onnx 만 기기로 복사
#        scp ${OUT_ROOT}/*.onnx  <jetson>:~/onnx/
#
#   2) 기기에서 FP16 엔진 빌드 (위 python 출력이 모델별 명령을 찍어 준다)
#        /usr/src/tensorrt/bin/trtexec --onnx=lightgev_ours_544x960.onnx \
#            --saveEngine=lightgev_ours_544x960_fp16.engine --fp16
#
#   3) FP16 정확도 확인이 필수다. cost volume 에 softmax 를 거는 구조라
#      FP16 에서 오차가 증폭될 수 있다. 엔진 출력 disparity 를 PyTorch FP32 와
#      비교해 EPE 열화를 반드시 재 볼 것.
# ---------------------------------------------------------------------------
