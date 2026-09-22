"""ONNX -> TensorRT INT8 엔진. 실제 SceneFlow 영상으로 calibration 한다 (PTQ).

trtexec 에 `--int8` 만 주면 각 레이어의 dynamic range 를 모르기 때문에 TensorRT 가
양자화하지 못한 레이어를 FP32 로 되돌린다. 그래서 "FP32+INT8" 혼합이 된다.
trtexec 는 calibration cache 를 **읽기만**(`--calib=`) 하고 만들지는 못한다.
이 스크립트가 그 cache 를 만든다.

--- 알아 두실 것 ---

1. **"INT8 만" 은 사실상 불가능하고 바람직하지도 않다.** TensorRT 는 레이어마다
   가장 빠른 정밀도를 고르고, INT8 커널이 없거나 더 느린 레이어는 FP16/FP32 로
   둔다. 목표는 "전부 INT8" 이 아니라 "INT8 로 돌 수 있는 레이어를 최대한" 이다.
   굳이 강제하려면 --strict 를 주지만(--precisionConstraints=obey 에 해당),
   보통 빌드가 실패하거나 오히려 느려진다.

2. **이 모델에서 INT8 은 정확도 위험이 크다.** 192 단계 cost volume 에 softmax 를
   걸어 soft-argmin 을 하는 구조라, INT8 의 좁은 동적 범위가 disparity 로 바로
   증폭된다. FP16 은 데스크탑 대비 손실이 거의 없었지만 INT8 은 그렇지 않을 수
   있다. **반드시** tools/eval_sceneflow_engine.py 로 region-wise EPE 를 재서
   FP16 대비 얼마나 나빠졌는지 확인할 것.

3. calibration 에는 **train split 을 쓴다** (기본값). test 로 calibration 하고
   test 로 평가하면 test 정보가 양자화 파라미터에 새어 들어가 수치가 낙관적으로
   나온다. SceneFlow train 19028 장 중 앞 몇백 장이면 충분하다.
   val 을 써도 되지만, val 로 epoch/체크포인트를 고르셨다면 train 이 더 안전하다.

4. calibration 데이터는 **추론과 같은 전처리** 여야 한다. 그래서 평가 코드의
   load_frame 을 그대로 쓴다 (RightTopPad 544x960 + ImageNet 정규화).

Usage:
    python3 tools/build_trt_int8.py \\
        --onnx  ~/onnx/lightgev_ours_544x960.onnx \\
        --engine ~/engines/lightgev_ours_544x960_int8.engine \\
        --data-root /media/mines/.../SceneFlow --num-calib 200
        # --list 기본값이 train 이다. test 를 쓰면 안 된다.

    # INT8 + FP16 혼합 (권장). INT8 이 안 되는 레이어가 FP32 대신 FP16 으로 간다
    python3 tools/build_trt_int8.py ... --fp16

    # 만들어진 cache 를 trtexec 에서 재사용
    trtexec --onnx=a.onnx --saveEngine=a_int8.engine --int8 --calib=a.calib
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_sceneflow_engine import load_frame
from infer_backends import _CudaMem


def make_calibrator(trt, files, root, args):
    """IInt8EntropyCalibrator2. 한 번에 1 장씩 GPU 로 올려 준다.

    TRT 10 에서 이 API 는 deprecated 지만 여전히 동작한다. 앞으로는 ONNX 에 Q/DQ
    노드를 넣는 explicit quantization(QAT)이 권장 경로다 - 그건 재학습이 필요하다.
    """

    class Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.files = files
            self.idx = 0
            self.mem = _CudaMem()
            self.dev = {}
            self.cache_path = args.cache
            self.failed = False

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            # 주의: TRT 는 이 콜백에서 난 예외를 **조용히 삼키고** calibration 이
            # 끝난 것으로 처리한다. 그러면 scale 이 하나도 안 만들어져서 빌드 때
            # "Missing scale and zero-point" 경고가 텐서 수만큼 쏟아진다.
            # 그래서 여기서 직접 잡아 크게 찍는다.
            try:
                if self.idx >= len(self.files):
                    print("  calibration 완료: {} 장 공급".format(self.idx), flush=True)
                    return None                  # None 이면 정상 종료
                left, right, _gt, _bgr = load_frame(
                    os.path.join(root, self.files[self.idx]),
                    args.cleanpass, (args.pad_h, args.pad_w))
                self.idx += 1

                if self.idx == 1:
                    print("  TRT 가 요구한 입력 이름: {}".format(list(names)), flush=True)
                    print("  공급 배열: left {} {}, right {} {}".format(
                        left.shape, left.dtype, right.shape, right.dtype), flush=True)
                if args.progress and self.idx % args.progress == 0:
                    print("  calibration {}/{}".format(self.idx, len(self.files)), flush=True)

                # 위치가 아니라 **이름**으로 맞춘다 (TRT 가 순서를 바꿔 물어볼 수 있다)
                supply = {"left": left, "right": right}
                out = []
                for i, name in enumerate(names):
                    arr = supply.get(name)
                    if arr is None:
                        arr = (left, right)[i] if i < 2 else left
                    arr = np.ascontiguousarray(arr, dtype=np.float32)
                    if name not in self.dev:
                        self.dev[name] = self.mem.alloc(arr.nbytes)
                    self.mem.htod(self.dev[name], arr)
                    out.append(self.mem.ptr(self.dev[name]))
                return out
            except Exception:
                import traceback
                print("\n  [치명] calibration get_batch 에서 예외 발생. "
                      "이대로 두면 TRT 가 조용히 calibration 을 끝내고", flush=True)
                print("         scale 없는 엔진을 만든다. 아래 원인을 고칠 것.", flush=True)
                traceback.print_exc()
                self.failed = True
                raise

        def read_calibration_cache(self):
            # cache 가 있으면 데이터를 다시 돌리지 않는다. 다시 만들려면 파일을 지울 것.
            if self.cache_path and os.path.isfile(self.cache_path):
                print("  기존 calibration cache 사용: {}".format(self.cache_path), flush=True)
                with open(self.cache_path, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            if self.cache_path:
                with open(self.cache_path, "wb") as f:
                    f.write(cache)
                print("  calibration cache 저장: {}".format(self.cache_path), flush=True)

    return Calibrator()


def build(args):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO if args.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # TRT 10 에서 EXPLICIT_BATCH 플래그가 사라졌다. 있으면 쓰고 없으면 0.
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)

    parser = trt.OnnxParser(network, logger)
    with open(args.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  [ONNX] {}".format(parser.get_error(i)), flush=True)
            raise SystemExit("ONNX 파싱 실패: {}".format(args.onnx))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb * 1024 * 1024)

    # 레이어별 정밀도를 나중에 들여다보려면 **빌드 시점에** 이걸 켜야 한다.
    # 기본값(LAYER_NAMES_ONLY)으로 구우면 EngineInspector 가 Myelin 오류를 내고
    # 아무것도 못 읽는다. 로드 시점에는 바꿀 수 없다. 엔진 크기가 조금 커질 뿐
    # 속도에는 영향이 없다.
    if not args.no_detailed_layers:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    # timing cache. INT8 빌드는 레이어마다 INT8/FP16/FP32 커널을 다 재 보므로
    # FP16 보다 훨씬 오래 걸린다(Orin Nano 에서 40 분 이상도 흔하다). 캐시를 두면
    # 다시 구울 때 그 탐색을 건너뛴다.
    timing_path = args.timing_cache or (os.path.splitext(args.engine)[0] + ".timing")
    timing_blob = b""
    if os.path.isfile(timing_path):
        with open(timing_path, "rb") as f:
            timing_blob = f.read()
        print("  timing cache 사용: {} ({:.1f} MB)".format(
            timing_path, len(timing_blob) / 1024 ** 2), flush=True)
    timing_cache = config.create_timing_cache(timing_blob)
    config.set_timing_cache(timing_cache, ignore_mismatch=False)
    config.set_flag(trt.BuilderFlag.INT8)
    if args.fp16:
        # INT8 커널이 없는 레이어가 FP32 대신 FP16 으로 간다. 보통 이게 낫다.
        config.set_flag(trt.BuilderFlag.FP16)
    if args.strict:
        # 지정한 정밀도를 강제한다. 빌드가 실패하거나 느려질 수 있다.
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            try:
                layer.precision = trt.int8
            except Exception:
                pass

    files = [l.split()[0] for l in open(args.list) if l.strip()][:args.num_calib]
    if not files:
        raise SystemExit("calibration 목록이 비었다: {}".format(args.list))
    missing = [f for f in files[:5] if not os.path.isfile(os.path.join(args.data_root, f))]
    if missing:
        raise SystemExit("calibration 데이터가 없다: {}".format(
            os.path.join(args.data_root, missing[0])))

    print("calibration: {} 장  목록={}  첫 항목={}".format(
        len(files), os.path.basename(args.list), files[0]), flush=True)
    config.int8_calibrator = make_calibrator(trt, files, args.data_root, args)

    print("빌드 시작. INT8 은 커널 탐색 공간이 넓어 Orin Nano 에서 40~90 분도 걸린다.",
          flush=True)
    print("  (TensorRT 는 빌드 중 진행률을 내지 않는다. --verbose 로 레이어별 로그를 볼 수 있다.)",
          flush=True)

    import threading
    t0 = time.time()
    done = threading.Event()

    def heartbeat():
        while not done.wait(300):               # 5 분마다 살아 있음을 알린다
            print("  ... 빌드 중 {:.0f} 분 경과".format((time.time() - t0) / 60), flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    plan = builder.build_serialized_network(network, config)
    done.set()
    print("  빌드 소요 {:.1f} 분".format((time.time() - t0) / 60), flush=True)

    if getattr(config.int8_calibrator, "failed", False):
        raise SystemExit("calibration 이 예외로 중단됐다. 위 traceback 을 고치고 "
                         ".calib 파일을 지운 뒤 다시 실행할 것.")
    if plan is None:
        raise SystemExit("엔진 빌드 실패 (위 로그 확인). --verbose 로 더 볼 수 있다.")

    os.makedirs(os.path.dirname(os.path.abspath(args.engine)) or ".", exist_ok=True)
    with open(args.engine, "wb") as f:
        f.write(plan)

    try:
        with open(timing_path, "wb") as f:
            f.write(memoryview(config.get_timing_cache().serialize()))
        print("  timing cache 저장: {}".format(timing_path), flush=True)
    except Exception as e:
        print("  timing cache 저장 실패(무해): {}".format(e), flush=True)
    print("저장: {}  ({:.1f} MB)".format(
        args.engine, os.path.getsize(args.engine) / 1024 ** 2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--cache", default=None,
                   help="calibration cache 경로 (기본: <engine>.calib). "
                        "trtexec --calib= 으로 재사용할 수 있다")
    p.add_argument("--data-root", dest="data_root", required=True,
                   help="FlyingThings3D_npz 를 담고 있는 디렉토리")
    p.add_argument("--list", default="data/SceneFlow/sceneflow_train_npz.txt",
                   help="calibration 에 쓸 목록. **train split 이 기본값이다.** "
                        "평가에 쓰는 test 목록을 여기 넣으면 test 정보가 새어 "
                        "들어가므로 쓰지 말 것")
    p.add_argument("--num-calib", dest="num_calib", type=int, default=200,
                   help="calibration 에 쓸 프레임 수 (기본 200). 수백 장이면 충분하다")
    p.add_argument("--fp16", action="store_true",
                   help="INT8 이 안 되는 레이어를 FP32 대신 FP16 으로 (권장)")
    p.add_argument("--strict", action="store_true",
                   help="모든 레이어에 INT8 을 강제 시도 (실패하거나 느려질 수 있다)")
    p.add_argument("--workspace-mb", dest="workspace_mb", type=int, default=2048)
    p.add_argument("--timing-cache", dest="timing_cache", default=None,
                   help="커널 탐색 캐시 경로 (기본: <engine>.timing). 재빌드가 크게 빨라진다")
    p.add_argument("--cleanpass", action="store_true")
    p.add_argument("--pad-h", dest="pad_h", type=int, default=544)
    p.add_argument("--pad-w", dest="pad_w", type=int, default=960)
    p.add_argument("--progress", type=int, default=50)
    p.add_argument("--allow-test-calib", dest="allow_test_calib", action="store_true",
                   help="test split 으로 calibration 하는 것을 허용 (권장하지 않는다)")
    p.add_argument("--no-detailed-layers", dest="no_detailed_layers", action="store_true",
                   help="레이어 상세 정보를 엔진에 넣지 않는다 (기본은 넣는다. "
                        "넣어야 tools/inspect_trt_engine.py 로 INT8 비중을 볼 수 있다)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.cache is None:
        args.cache = os.path.splitext(args.engine)[0] + ".calib"

    # tensorrt import 전에 검사한다. 잘못된 split 이면 즉시 멈추는 게 낫다.
    if "test" in os.path.basename(args.list).lower() and not args.allow_test_calib:
        raise SystemExit(
            "calibration 목록이 test split 이다: {}\n"
            "  test 로 calibration 하고 test 로 평가하면 test 정보가 양자화\n"
            "  파라미터에 새어 들어간다. train 목록을 쓸 것:\n"
            "    --list data/SceneFlow/sceneflow_train_npz.txt\n"
            "  그래도 강행하려면 --allow-test-calib".format(args.list))

    build(args)


if __name__ == "__main__":
    main()
