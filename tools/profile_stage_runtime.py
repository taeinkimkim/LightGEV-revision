"""LightGEVModify_4 의 구간별 runtime 과 모델 제원을 측정한다.

측정 대상 5 구간 (모델 forward 안의 CUDA event mark 와 1:1):

    1. backbone_gwc            Backbone + gwc 생성
    2. geo_encoding_gating     geometry encoding + gating
    3. aggregation_regression  LightStereo-style Aggregation + disparity regression
    4. gru_refinement          geometry-aware ConvGRU refinement
    5. context_upsampling      LightStereo context-aware upsampling

함께 보고하는 것: input resolution / numerical precision / model size(MB) /
learnable parameter 수 / peak inference memory.

--- 측정 방법에 대해 ---

전체 latency 와 구간별 분해를 **따로** 잰다. CUDA event 의 elapsed_time 은
synchronize 를 강제해서 커널 파이프라이닝을 끊기 때문에, 계측을 켠 채로 잰
전체 시간은 실제보다 느리게 나온다. 그래서

  * 전체 latency : 계측 OFF 로 --runs 회 반복, 시작/끝에만 event
  * 구간별       : 계측 ON 으로 --runs 회 반복 (합이 전체보다 조금 크게 나온다)

두 번을 돌리고, 리포트에 그 차이(계측 오버헤드)를 같이 찍는다.

warm-up 과 반복 횟수: 첫 수십 회는 cuDNN algo 선택, 메모리 할당자 캐시,
GPU 클럭 상승 때문에 느리다. 기본값 warmup 50 / runs 200 은 이 모델(~10ms)
기준 warm-up 0.5s, 측정 2s 로, p95 가 안정되는 선에서 잡은 값이다.
`--runs` 를 늘려도 평균은 거의 안 변하고 꼬리 분위수만 촘촘해진다.

Usage:
    python tools/profile_stage_runtime.py --cfg cfgs/lightgev_modify_4/lightgev_modify_4_eval_sceneflow_full_iter_6_r_075.yaml
    python tools/profile_stage_runtime.py --cfg <cfg> --resolution 544x960 384x1248 --precision fp32 amp
    python tools/profile_stage_runtime.py --cfg <cfg> --warmup 100 --runs 500 --csv out.csv
"""

import argparse
import csv
import os
import statistics
import sys

import torch
import yaml
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stereo.modeling.models.lightgev.lightgev_modify_4 import LightGEVModify_4

PRECISIONS = ("fp32", "amp", "fp16")


def parse_resolution(text):
    h, w = text.lower().split("x")
    return int(h), int(w)


def model_stats(model):
    """parameter 수와 model size. buffer(BN running stat 등)도 크기에 포함한다."""
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_n = sum(b.numel() for b in model.buffers())
    buf_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "params_total": total_p,
        "params_learnable": train_p,
        "buffers": buf_n,
        "size_mb": (param_bytes + buf_bytes) / (1024 ** 2),
    }


def make_input(h, w, device, dtype):
    g = torch.Generator(device=device).manual_seed(0)
    return {
        "left": torch.randn(1, 3, h, w, generator=g, device=device, dtype=dtype),
        "right": torch.randn(1, 3, h, w, generator=g, device=device, dtype=dtype),
    }


def run_loop(model, data, precision, warmup, runs, stage_timing):
    """(총 latency 리스트[ms], 구간별 시간 dict[list[ms]], peak mem (alloc, reserved) GB).

    peak memory 는 warm-up 이 끝난 뒤에 리셋하고 잰다. cudnn.benchmark 가 켜져
    있으면 warm-up 중 algo 탐색이 거대한 workspace 를 잠깐 잡는데(측정해 보면
    steady-state 의 20 배까지 간다), 그건 추론 시 실제로 필요한 메모리가 아니다.
    """
    autocast = torch.amp.autocast("cuda", enabled=(precision == "amp"))
    model.enable_stage_timing(stage_timing)

    with torch.no_grad():
        for _ in range(warmup):
            with autocast:
                model(data)
        torch.cuda.synchronize()

        # warm-up(= autotune) 이 끝난 뒤부터의 peak 만 센다
        torch.cuda.reset_peak_memory_stats()

        totals = []
        stages = {}
        for _ in range(runs):
            start, end = (torch.cuda.Event(enable_timing=True),
                          torch.cuda.Event(enable_timing=True))
            start.record()
            with autocast:
                model(data)
            end.record()
            torch.cuda.synchronize()
            totals.append(start.elapsed_time(end))
            if stage_timing:
                for k, v in model.stage_times_ms().items():
                    if k != "_start":
                        stages.setdefault(k, []).append(v)

    model.enable_stage_timing(False)
    peak = (torch.cuda.max_memory_allocated() / (1024 ** 3),
            torch.cuda.max_memory_reserved() / (1024 ** 3))
    return totals, stages, peak


def summarize(samples):
    s = sorted(samples)
    return {
        "mean": statistics.fmean(s),
        "std": statistics.pstdev(s) if len(s) > 1 else 0.0,
        "median": statistics.median(s),
        "p95": s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))],
        "min": s[0],
        "max": s[-1],
    }


def profile_one(cfg, h, w, precision, args, device):
    model = LightGEVModify_4(cfg.MODEL).to(device).eval()
    if precision == "fp16":
        model = model.half()
    dtype = torch.float16 if precision == "fp16" else torch.float32
    data = make_input(h, w, device, dtype)

    stats = model_stats(model)

    # 1) 전체 latency: 계측 OFF (파이프라이닝 그대로)
    torch.cuda.empty_cache()
    totals, _, (peak_alloc, peak_reserved) = run_loop(
        model, data, precision, args.warmup, args.runs, stage_timing=False)

    # 2) 구간별: 계측 ON (event 마다 sync 가 걸려 합이 조금 커진다)
    totals_inst, stages, _ = run_loop(model, data, precision, args.warmup_stage, args.runs,
                                      stage_timing=True)

    del model, data
    torch.cuda.empty_cache()

    return {
        "resolution": "{}x{}".format(h, w),
        "precision": precision,
        "stats": stats,
        "total": summarize(totals),
        "total_instrumented": summarize(totals_inst),
        "stages": {k: summarize(v) for k, v in stages.items()},
        "peak_alloc_gb": peak_alloc,
        "peak_reserved_gb": peak_reserved,
    }


def report(r, stage_order):
    st = r["stats"]
    print("\n" + "=" * 78)
    print(" {}  |  {}".format(r["resolution"], r["precision"]))
    print("=" * 78)
    print("  learnable params : {:,}".format(st["params_learnable"]))
    print("  total params     : {:,}  (buffers {:,})".format(st["params_total"], st["buffers"]))
    print("  model size       : {:.2f} MB".format(st["size_mb"]))
    print("  peak memory      : {:.3f} GB allocated / {:.3f} GB reserved".format(
        r["peak_alloc_gb"], r["peak_reserved_gb"]))

    t = r["total"]
    print("\n  total latency    : {:.3f} ms  (std {:.3f}, median {:.3f}, p95 {:.3f}, "
          "min {:.3f})".format(t["mean"], t["std"], t["median"], t["p95"], t["min"]))
    print("  throughput       : {:.1f} FPS".format(1000.0 / t["mean"]))

    stage_sum = sum(v["mean"] for v in r["stages"].values())
    print("\n  {:<26}{:>10}{:>9}{:>9}{:>9}".format("stage", "mean ms", "std", "p95", "share"))
    print("  " + "-" * 63)
    for name in stage_order:
        if name not in r["stages"]:
            continue
        v = r["stages"][name]
        print("  {:<26}{:>10.3f}{:>9.3f}{:>9.3f}{:>8.1f}%".format(
            name, v["mean"], v["std"], v["p95"], 100.0 * v["mean"] / stage_sum))
    print("  " + "-" * 63)
    print("  {:<26}{:>10.3f}".format("stage sum", stage_sum))
    overhead = r["total_instrumented"]["mean"] - t["mean"]
    print("  {:<26}{:>10.3f}  (event sync 로 인한 계측 오버헤드)".format(
        "instrumented - total", overhead))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", required=True, help="모델 yaml (MODEL 블록만 쓴다)")
    parser.add_argument("--resolution", nargs="+", default=["544x960"],
                        help="HxW 목록 (기본: 544x960 = SceneFlow 평가 해상도)")
    parser.add_argument("--precision", nargs="+", default=["fp32"], choices=PRECISIONS,
                        help="fp32 / amp(autocast fp16) / fp16(model.half()). "
                             "기존 평가 파이프라인은 amp=False 라 fp32 가 기본값이다.")
    parser.add_argument("--warmup", type=int, default=50,
                        help="전체 latency 측정 전 warm-up 횟수 (기본 50)")
    parser.add_argument("--warmup-stage", dest="warmup_stage", type=int, default=20,
                        help="구간별 측정 전 warm-up 횟수 (이미 달궈진 뒤라 적어도 된다, 기본 20)")
    parser.add_argument("--runs", type=int, default=200, help="측정 반복 횟수 (기본 200)")
    parser.add_argument("--cudnn-benchmark", dest="cudnn_benchmark",
                        action="store_true", default=True,
                        help="cuDNN autotune 켜기 (기본 켬, warm-up 이 흡수한다)")
    parser.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--csv", nargs="?", const="AUTO", default=None,
                        help="결과를 csv 로 저장. 값 없이 주면 stage_runtime.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 가 필요하다 (CUDA event 로 측정한다).")

    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    device = torch.device(args.device)
    cfg = EasyDict(yaml.safe_load(open(args.cfg)))

    print("cfg        : {}".format(args.cfg))
    print("device     : {}  ({})".format(args.device, torch.cuda.get_device_name(device)))
    print("torch      : {}  (cuda {})".format(torch.__version__, torch.version.cuda))
    print("warm-up    : {} (total) / {} (stage)".format(args.warmup, args.warmup_stage))
    print("runs       : {}".format(args.runs))
    print("cudnn.benchmark: {}".format(args.cudnn_benchmark))

    stage_order = list(LightGEVModify_4.STAGE_NAMES)
    results = []
    for res in args.resolution:
        h, w = parse_resolution(res)
        for precision in args.precision:
            r = profile_one(cfg, h, w, precision, args, device)
            report(r, stage_order)
            results.append(r)

    if args.csv:
        path = "stage_runtime.csv" if args.csv == "AUTO" else args.csv
        with open(path, "w", newline="") as f:
            w_ = csv.writer(f)
            w_.writerow(["resolution", "precision", "params_learnable", "params_total",
                         "size_mb", "peak_alloc_gb", "peak_reserved_gb",
                         "total_mean_ms", "total_std_ms", "total_p95_ms", "fps"]
                        + ["{}_mean_ms".format(n) for n in stage_order])
            for r in results:
                w_.writerow([r["resolution"], r["precision"],
                             r["stats"]["params_learnable"], r["stats"]["params_total"],
                             round(r["stats"]["size_mb"], 3),
                             round(r["peak_alloc_gb"], 4), round(r["peak_reserved_gb"], 4),
                             round(r["total"]["mean"], 4), round(r["total"]["std"], 4),
                             round(r["total"]["p95"], 4), round(1000.0 / r["total"]["mean"], 2)]
                            + [round(r["stages"].get(n, {}).get("mean", float("nan")), 4)
                               for n in stage_order])
        print("\nsaved: {}".format(path))


if __name__ == "__main__":
    main()


'''
python tools/profile_stage_runtime.py \
    --cfg cfgs/lightgev_modify_4/lightgev_modify_4_eval_sceneflow_full_iter_6_r_075.yaml \
    --resolution 544x960 --precision fp32 --csv
'''
