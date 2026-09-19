"""TensorRT 엔진(또는 ONNX / PyTorch)으로 SceneFlow test 의 region-wise 정확도를 잰다.

Jetson 에 올린 FP16/BF16 엔진이 **PyTorch FP32 대비 얼마나 나빠졌는지**를 재는 게
목적이다. --compare 로 두 백엔드를 같은 프레임에서 돌려 region 별로 직접 비교한다.

    # Jetson: FP16 엔진 정확도 (전체 4370 장)
    python3 tools/eval_sceneflow_engine.py --backend trt \\
        --engine lightgev_ours_544x960_fp16.engine --data-root <SceneFlow>

    # 데스크탑: 파이프라인 확인 (TensorRT 없이 ONNX 로)
    python tools/eval_sceneflow_engine.py --backend onnx \\
        --onnx onnx_export/lightgev_ours_544x960.onnx --limit 20

    # FP16 엔진 vs PyTorch FP32 를 나란히
    python3 tools/eval_sceneflow_engine.py --backend trt --engine x.engine \\
        --compare torch --cfg cfgs/... --ckpt ...

--- torch 없이 돈다 ---

Jetson 에 PyTorch 가 없어도 trt / onnx 백엔드는 그대로 쓸 수 있다. 추론은 엔진이,
region 마스크는 numpy+cv2 가, 지표는 numpy 가 한다. torch 는 `--backend torch` 나
`--compare torch` (FP32 기준값)를 쓸 때만 import 된다.

GPU 메모리는 cuda-python 이나 pycuda 로 잡는다 (`pip install cuda-python`).

--- 지표 ---

`tools/eval_single.py` 와 **같은 정의**로 잰다: `RegionBinner` 로
texless / edge / thick:thinQ / thick:thickQ / global 마스크를 만들고 region 마다
EPE / D1 / t1 / t3 를 낸다. 마스크 생성은 원본 코드를 그대로 import 하고, 지표
산술만 torch -> numpy 로 옮겼다 (`tools/region_metrics_np.py`). 전처리(RightTopPad 544x960, ImageNet 정규화),
region 마스크의 입력(정규화 전 BGR 좌영상), GT clip(0..MAX_DISP) 까지 맞춰 두었으므로
결과는 기존 results.csv 의 REGION_SUMMARY 행과 같은 정의다.

주의: 여기서는 REGION_SUMMARY 만 낸다. 기존 파이프라인의 SUMMARY 행은 region 루프
안에서 누적돼 모든 region 을 겹쳐 세므로(global 포함) 전체 평균이 아니다. 그 값은
재현하지 않는다.

주의: region 마스크는 예측값과 무관하다(좌영상과 GT 로만 만든다). 그래서 백엔드가
달라도 마스크와 npix 는 동일하고, region 별 EPE 차이는 순수하게 예측 차이다.

--- 데이터 ---

SceneFlow test 는 4370 장 / 34 GB 다. 정확도를 기존 표와 같은 조건으로 보려면 전체를
써야 하므로 --limit 은 기본값 0(전부)이다. --limit 은 Jetson 으로 옮기기 전에
파이프라인만 빠르게 확인할 때 쓴다.
"""

import argparse
import os
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infer_backends import make_runner
# torch 를 쓰지 않는 region 평가기. 마스크 코드는 원본(RegionBinner)을 그대로 쓰고
# 지표 산술만 numpy 로 옮긴 것이다. 데스크탑에서 --check-metrics 로 torch 구현과
# 같은 값이 나오는지 대조할 수 있다.
from region_metrics_np import (
    NumpyRegionEvaluator, denorm_disparity, resize_disparity_with_scale_x,
    sanitize_and_clip,
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

REGION_ORDER = ["texless", "edge", "thick:thinQ", "thick:thickQ", "global"]


def load_frame(npz_path, cleanpass, pad_to):
    """eval 파이프라인의 TESTING 전처리를 그대로 재현한다.

    RightTopPad([544,960]) -> NormalizeImage. SceneFlow 원본이 540x960 이라 위쪽에
    4 행이 붙고 그 영역 GT 는 0 이라 유효 마스크에서 빠진다. 544/960 은 이미 32 의
    배수라 InputPadder 는 아무 일도 하지 않는다.

    region 마스크는 **정규화 전** 좌영상으로 만든다. NormalizeImage 가 정규화 직전에
    left_ori 를 복사해 두고 single_model_eval 이 그걸 BGR 로 바꿔 쓰기 때문이다.
    """
    d = np.load(npz_path)
    key_l, key_r = ("left_clean", "right_clean") if cleanpass else ("left_final", "right_final")
    left, right, disp = d[key_l], d[key_r], d["disp"]

    th, tw = pad_to
    h, w = left.shape[:2]
    pad_top, pad_right = max(0, th - h), max(0, tw - w)
    if pad_top or pad_right:
        pw = ((pad_top, 0), (0, pad_right), (0, 0))
        left = np.pad(left, pw, "edge")
        right = np.pad(right, pw, "edge")
        disp = np.pad(disp, ((pad_top, 0), (0, pad_right)), "constant", constant_values=0)

    left_ori_bgr = cv2.cvtColor(left.astype(np.float32), cv2.COLOR_RGB2BGR)

    def prep(img):
        x = ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
        return np.ascontiguousarray(x, dtype=np.float32)[None]

    return prep(left), prep(right), disp.astype(np.float32), left_ori_bgr


def run_backend(runner, files, root, args, evaluator, label):
    """(region 별 집계, 프레임별 예측, latency).

    집계는 single_model_eval 과 같은 npix 가중 평균이다.
    """
    stats = {}
    lat, preds = [], []

    for i, rel in enumerate(files):
        left, right, gt, left_bgr = load_frame(os.path.join(root, rel),
                                               args.cleanpass, (args.pad_h, args.pad_w))
        H, W = gt.shape

        t0 = time.perf_counter()
        pred_np = runner(left, right)          # 백엔드가 자체적으로 동기화한다
        lat.append((time.perf_counter() - t0) * 1000.0)

        # single_model_eval 과 같은 후처리
        pred_np = np.asarray(pred_np, dtype=np.float32)
        while pred_np.ndim > 2 and pred_np.shape[0] == 1:
            pred_np = pred_np[0]
        pred_np = resize_disparity_with_scale_x(pred_np, (H, W))
        pred_np = denorm_disparity(pred_np, args.max_disp, args.pred_is_normalized)
        pred_np = sanitize_and_clip(pred_np, 0.0, args.max_disp)
        gt_np = sanitize_and_clip(gt, 0.0, args.max_disp)

        results, _ = evaluator.evaluate(left_bgr, pred_np, gt_np)
        for region, m in results.items():
            s = stats.setdefault(region, {"npix": 0, "epe": 0.0, "d1": 0.0, "t1": 0.0, "t3": 0.0})
            n = int(m["npix"])
            s["npix"] += n
            for k in ("epe", "d1", "t1", "t3"):
                s[k] += float(m[k]) * n

        if args.keep_preds:
            preds.append(pred_np)

        if args.progress and (i + 1) % args.progress == 0:
            g = stats.get("global")
            cur = g["epe"] / g["npix"] if g and g["npix"] else float("nan")
            print("  [{}] {}/{}  누적 global EPE {:.4f}".format(
                label, i + 1, len(files), cur), flush=True)

    if not stats:
        raise SystemExit("region 결과가 비었다. 데이터 경로와 --max_disp 를 확인할 것.")

    agg = {r: {"npix": s["npix"],
               **{k: s[k] / s["npix"] for k in ("epe", "d1", "t1", "t3")}}
           for r, s in stats.items() if s["npix"] > 0}
    return agg, preds, lat


def check_metrics(files, root, args):
    """numpy 지표가 torch 원본(DepthThicknessEvaluator)과 같은 값을 내는지 대조한다.

    torch 가 있는 데스크탑에서만 돈다. region 정의를 고친 뒤 Jetson 경로가 여전히
    같은 값을 내는지 확인하는 용도.
    """
    from stereo.evaluation.depth_thickness_evaluator import DepthThicknessEvaluator
    ref = DepthThicknessEvaluator(device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "0") else "cpu")
    mine = NumpyRegionEvaluator()
    rng = np.random.default_rng(0)
    worst = {}
    for rel in files:
        left, right, gt, left_bgr = load_frame(os.path.join(root, rel),
                                               args.cleanpass, (args.pad_h, args.pad_w))
        # 예측은 아무거나 좋다 - 지표 산술이 같은지만 보면 된다
        pred = sanitize_and_clip(gt + rng.normal(0, 2.0, gt.shape).astype(np.float32),
                                 0.0, args.max_disp)
        gt_c = sanitize_and_clip(gt, 0.0, args.max_disp)
        a, _ = ref.evaluate(left_bgr, pred, gt_c)
        b, _ = mine.evaluate(left_bgr, pred, gt_c)
        assert set(a) == set(b), (sorted(a), sorted(b))
        for r in a:
            for k in ("npix", "epe", "d1", "t1", "t3"):
                d = abs(float(a[r][k]) - float(b[r][k]))
                worst[k] = max(worst.get(k, 0.0), d)
    print("\n=== 지표 구현 대조 (torch 원본 vs numpy), {} 장 ===".format(len(files)), flush=True)
    for k, v in worst.items():
        flag = "OK" if v <= (0 if k == "npix" else 1e-3) else "[불일치]"
        print("  {:<6} 최대 차이 {:.3e}  {}".format(k, v, flag), flush=True)


def build_torch_model(args):
    import yaml
    from easydict import EasyDict
    from profile_model_runtime import build_model
    from export_onnx import load_checkpoint
    if not args.cfg:
        raise SystemExit("torch 백엔드에는 --cfg 가 필요하다")
    cfg = EasyDict(yaml.safe_load(open(args.cfg)))
    model = build_model(cfg)
    if args.ckpt:
        load_checkpoint(model, args.ckpt, lambda s: print(s, flush=True))
    else:
        print("  [경고] --ckpt 없음 -> 랜덤 가중치. 정확도 수치는 의미 없다.", flush=True)
    # TorchRunner 가 dict 를 넣고 dict 에서 disp_pred 를 꺼내므로 원본 모델을 그대로 준다
    # (StereoWrapper 는 ONNX export 전용이라 (left, right) 시그니처가 다르다).
    return model


def report(label, agg, lat):
    print("\n  === {} ===".format(label), flush=True)
    print("  {:<14}{:>14}{:>10}{:>10}{:>10}{:>10}".format(
        "region", "npix", "EPE", "D1", "t1", "t3"), flush=True)
    for r in [x for x in REGION_ORDER if x in agg] + [x for x in agg if x not in REGION_ORDER]:
        m = agg[r]
        print("  {:<14}{:>14d}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}".format(
            r, m["npix"], m["epe"], m["d1"], m["t1"], m["t3"]), flush=True)
    s = sorted(lat)
    print("  latency: mean {:.2f} ms  median {:.2f}  p95 {:.2f}  ({:.1f} FPS)".format(
        statistics.fmean(s), statistics.median(s),
        s[min(len(s) - 1, int(0.95 * (len(s) - 1)))], 1000.0 / statistics.fmean(s)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="trt", choices=["trt", "onnx", "torch"])
    p.add_argument("--engine", help="TensorRT 엔진 (.engine)")
    p.add_argument("--onnx", help="ONNX 파일")
    p.add_argument("--cfg", help="torch 백엔드용 모델 yaml")
    p.add_argument("--ckpt", help="torch 백엔드용 checkpoint")
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"],
                   help="torch 백엔드의 정밀도 (기준값은 fp32)")
    p.add_argument("--compare", choices=["trt", "onnx", "torch"],
                   help="두 번째 백엔드를 같이 돌려 region 별로 비교한다")
    p.add_argument("--compare-engine", dest="compare_engine")
    p.add_argument("--compare-onnx", dest="compare_onnx")

    p.add_argument("--data-root", dest="data_root",
                   default="/media/user/PortableSSD_taein/TaeinKim/LightGEV/SceneFlow")
    p.add_argument("--list", default="data/SceneFlow/sceneflow_test_npz.txt")
    p.add_argument("--limit", type=int, default=0,
                   help="앞에서 N 장만 (기본 0 = 전체 4370장). 파이프라인 확인용")
    p.add_argument("--cleanpass", action="store_true",
                   help="clean pass 사용 (기본은 final pass = eval cfg 의 CLEANPASS false)")
    p.add_argument("--max_disp", type=float, default=192.0)
    p.add_argument("--pred-is-normalized", dest="pred_is_normalized", action="store_true")
    p.add_argument("--pad-h", dest="pad_h", type=int, default=544)
    p.add_argument("--pad-w", dest="pad_w", type=int, default=960)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=None, help="onnxruntime 스레드 수")
    p.add_argument("--progress", type=int, default=200, help="N 장마다 진행 출력 (0=끔)")
    p.add_argument("--keep-preds", dest="keep_preds", action="store_true",
                   help="--compare 시 프레임별 |Δdisparity| 도 본다 (메모리를 많이 쓴다)")
    p.add_argument("--check-metrics", dest="check_metrics", action="store_true",
                   help="numpy 지표를 torch 원본과 대조하고 끝낸다 (데스크탑 전용)")
    p.add_argument("--csv", help="결과 csv 경로 (results.csv 의 REGION_SUMMARY 형식)")
    args = p.parse_args()

    files = [l.split()[0] for l in open(args.list) if l.strip()]
    if args.limit:
        files = files[:args.limit]
    print("데이터  : {} 장  ({})".format(
        len(files), "clean pass" if args.cleanpass else "final pass"), flush=True)
    print("해상도  : {}x{}  (RightTopPad)".format(args.pad_h, args.pad_w), flush=True)
    print("max_disp: {}".format(args.max_disp), flush=True)

    if args.check_metrics:
        check_metrics(files, args.data_root, args)
        return

    evaluator = NumpyRegionEvaluator()

    torch_model = build_torch_model(args) if "torch" in (args.backend, args.compare) else None

    runner = make_runner(args.backend, engine=args.engine, onnx=args.onnx,
                         model=torch_model, device=args.device,
                         precision=args.precision, threads=args.threads)
    print("백엔드  : {}".format(runner.name), flush=True)
    agg1, preds1, lat1 = run_backend(runner, files, args.data_root, args, evaluator, args.backend)
    report(args.backend, agg1, lat1)
    rows = [(args.backend, runner.name, agg1, lat1)]

    if args.compare:
        r2 = make_runner(args.compare, engine=args.compare_engine or args.engine,
                         onnx=args.compare_onnx or args.onnx, model=torch_model,
                         device=args.device, precision="fp32", threads=args.threads)
        print("\n비교 백엔드: {}".format(r2.name), flush=True)
        agg2, preds2, lat2 = run_backend(r2, files, args.data_root, args, evaluator, args.compare)
        report(args.compare, agg2, lat2)
        rows.append((args.compare, r2.name, agg2, lat2))

        print("\n" + "=" * 74, flush=True)
        print(" region 별 EPE : {} 기준 -> {}".format(args.compare, args.backend), flush=True)
        print("=" * 74, flush=True)
        print("  {:<14}{:>12}{:>12}{:>12}{:>10}".format(
            "region", args.compare, args.backend, "Δ", "%"), flush=True)
        for r in [x for x in REGION_ORDER if x in agg1 and x in agg2]:
            a, b = agg2[r]["epe"], agg1[r]["epe"]
            print("  {:<14}{:>12.4f}{:>12.4f}{:>+12.4f}{:>9.2f}%".format(
                r, a, b, b - a, 100.0 * (b - a) / a if a else 0.0), flush=True)
        m1, m2 = statistics.fmean(lat1), statistics.fmean(lat2)
        print("\n  속도   {:.2f} ms -> {:.2f} ms   {:.2f}x".format(m2, m1, m2 / m1), flush=True)

        if args.keep_preds and preds1 and preds2:
            dm = np.array([np.abs(a - b).max() for a, b in zip(preds1, preds2)])
            print("  프레임별 max |Δdisparity|: 중앙값 {:.4f} px, 최대 {:.4f} px".format(
                float(np.median(dm)), float(dm.max())), flush=True)

    if args.csv:
        import csv as _csv
        parent = os.path.dirname(os.path.abspath(args.csv))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["backend", "detail", "frames", "name", "metric", "npix",
                        "epe", "d1", "t1", "t3", "latency_mean_ms", "fps"])
            for tag, name, agg, lat in rows:
                mean = statistics.fmean(lat)
                for r in [x for x in REGION_ORDER if x in agg]:
                    m = agg[r]
                    w.writerow([tag, name, len(files), "REGION_SUMMARY", r, m["npix"],
                                round(m["epe"], 6), round(m["d1"], 6), round(m["t1"], 6),
                                round(m["t3"], 6), round(mean, 3), round(1000.0 / mean, 2)])
        print("\nsaved: {}".format(args.csv), flush=True)


if __name__ == "__main__":
    main()
