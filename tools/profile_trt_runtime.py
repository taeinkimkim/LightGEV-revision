"""TensorRT 엔진의 runtime 을 데스크탑 `tools/profile_model_runtime.py` 와 **같은 조건**으로 잰다.

trtexec 로는 이 조건을 그대로 재현할 수 없어서 만들었다:

  * `trtexec --warmUp=N` 은 **N 회가 아니라 N 밀리초**다. "warm-up 50 회" 를 직접
    지정할 방법이 없다 (1 회 지연을 모르면 ms 로 환산도 못 한다).
  * `--iterations=N --duration=0` 으로 측정 횟수는 맞출 수 있다.
  * peak memory 는 `--monitorMemory` 가 대략 보여주지만 데스크탑에서 쓴
    `torch.cuda.max_memory_allocated()` 와 정의가 다르다.

여기서는 데스크탑과 같게 맞춘다:
  * warm-up / 측정을 **횟수**로 지정 (기본 50 / 200)
  * H2D/D2H 복사를 빼고 **실행만** 측정 (데스크탑이 CUDA event 로 forward 만 잰 것과 동일)
  * peak GPU 메모리는 cudaMemGetInfo 의 free 최저점으로 측정
  * mean / median / p95 / min, FPS 출력

참고로 trtexec 의 "GPU Compute Time" 도 순수 실행 시간이라 여기 mean 과 비슷해야
한다. 크게 벌어지면 둘 중 하나가 잘못 측정된 것이니 확인할 것.

Usage:
    python3 tools/profile_trt_runtime.py --engine ~/engines/lightgev_ours_544x960_fp16.engine
    python3 tools/profile_trt_runtime.py --engine a.engine b.engine --warmup 50 --runs 200 --csv
    python3 tools/profile_trt_runtime.py --backend onnx --onnx model.onnx   # 데스크탑 확인용
"""

import argparse
import csv
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infer_backends import make_runner


def parse_resolution(text):
    h, w = text.lower().split("x")
    return int(h), int(w)


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


def bench_one(path, args):
    runner = make_runner(args.backend, engine=path, onnx=path,
                         device=args.device, threads=args.threads)
    h, w = parse_resolution(args.resolution)
    rng = np.random.default_rng(0)
    left = rng.standard_normal((1, 3, h, w), dtype=np.float32)
    right = rng.standard_normal((1, 3, h, w), dtype=np.float32)

    split = hasattr(runner, "bind") and hasattr(runner, "execute")
    mem_fn = getattr(runner, "mem_info_gb", lambda: None)

    base = mem_fn()
    free_floor = base[0] if base else None

    if split:
        runner.bind(left, right)          # 복사는 한 번만
        step = runner.execute
    else:
        # onnx 백엔드에는 분리가 없다. 복사가 포함된 end-to-end 시간이 된다.
        def step():
            runner(left, right)

    for _ in range(args.warmup):
        step()

    after_warm = mem_fn()
    if after_warm:
        free_floor = min(free_floor, after_warm[0]) if free_floor else after_warm[0]

    times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        step()
        times.append((time.perf_counter() - t0) * 1000.0)
        if (i & 15) == 0:                  # 매번 부르면 측정에 영향을 준다
            cur = mem_fn()
            if cur:
                free_floor = min(free_floor, cur[0])

    peak_gb = None
    if base and free_floor is not None:
        peak_gb = base[1] - free_floor     # total - free 최저점 = 사용 최대치

    size_mb = os.path.getsize(path) / (1024 ** 2)
    return {
        "name": os.path.basename(path), "backend": runner.name,
        "size_mb": size_mb, "latency": summarize(times),
        "peak_gb": peak_gb, "copies_excluded": split,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="trt", choices=["trt", "onnx"])
    p.add_argument("--engine", nargs="+", help="TensorRT 엔진 (여러 개 가능)")
    p.add_argument("--onnx", nargs="+", help="ONNX 파일 (--backend onnx)")
    p.add_argument("--resolution", default="544x960")
    p.add_argument("--warmup", type=int, default=50, help="warm-up **횟수** (기본 50)")
    p.add_argument("--runs", type=int, default=200, help="측정 **횟수** (기본 200)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--csv", nargs="?", const="AUTO", default=None,
                   help="csv 저장. 값 없이 주면 trt_runtime.csv")
    args = p.parse_args()

    paths = args.engine if args.backend == "trt" else args.onnx
    if not paths:
        raise SystemExit("--engine (또는 --backend onnx 와 --onnx) 가 필요하다")

    h, w = parse_resolution(args.resolution)
    print("입력     : {}x{}".format(h, w), flush=True)
    print("warm-up  : {} 회   측정: {} 회".format(args.warmup, args.runs), flush=True)

    head = "{:<38}{:>9}{:>9}{:>9}{:>9}{:>9}{:>10}".format(
        "engine", "size MB", "mean ms", "med ms", "p95 ms", "FPS", "peak GB")
    print("\n" + "=" * len(head), flush=True)
    print(head, flush=True)
    print("-" * len(head), flush=True)

    rows = []
    for path in paths:
        if not os.path.isfile(path):
            print("{:<38}{:>56}".format(os.path.basename(path), "파일 없음"), flush=True)
            continue
        print("{:<38}".format(os.path.basename(path)[:37]), end="", flush=True)
        try:
            r = bench_one(path, args)
        except Exception as e:
            print("{:>56}".format("FAILED: " + str(e)[:48]), flush=True)
            continue
        lat = r["latency"]
        print("{:>9.1f}{:>9.3f}{:>9.3f}{:>9.3f}{:>9.1f}{:>10}".format(
            r["size_mb"], lat["mean"], lat["median"], lat["p95"], 1000.0 / lat["mean"],
            "{:.3f}".format(r["peak_gb"]) if r["peak_gb"] is not None else "-"), flush=True)
        rows.append(r)

    print("-" * len(head), flush=True)
    if rows:
        print("  백엔드: {}".format(rows[0]["backend"]), flush=True)
        if rows[0]["copies_excluded"]:
            print("  측정 구간: 실행만 (H2D/D2H 복사 제외). 데스크탑 CUDA event 측정과 같은 조건.",
                  flush=True)
        else:
            print("  측정 구간: end-to-end (복사 포함). 이 백엔드는 분리를 지원하지 않는다.",
                  flush=True)
        if rows[0]["peak_gb"] is None:
            print("  peak GB: cuda-python 이 없어 측정 못 함 (pip install cuda-python).", flush=True)
        else:
            print("  peak GB: cudaMemGetInfo 의 free 최저점 기준. Orin 은 CPU 와 메모리를 "
                  "공유하므로 다른 프로세스 영향을 받는다.", flush=True)

    if args.csv and rows:
        path = "trt_runtime.csv" if args.csv == "AUTO" else args.csv
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["engine", "backend", "resolution", "warmup", "runs", "size_mb",
                           "mean_ms", "std_ms", "median_ms", "p95_ms", "min_ms", "fps",
                           "peak_gb", "copies_excluded"])
            for r in rows:
                lat = r["latency"]
                wcsv.writerow([r["name"], r["backend"], args.resolution, args.warmup, args.runs,
                               round(r["size_mb"], 2), round(lat["mean"], 4),
                               round(lat["std"], 4), round(lat["median"], 4),
                               round(lat["p95"], 4), round(lat["min"], 4),
                               round(1000.0 / lat["mean"], 2),
                               round(r["peak_gb"], 4) if r["peak_gb"] is not None else "",
                               r["copies_excluded"]])
        print("\nsaved: {}".format(path), flush=True)


if __name__ == "__main__":
    main()
