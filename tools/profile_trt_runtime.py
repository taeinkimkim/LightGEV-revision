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
  * GPU 메모리는 TRT 가 보고하는 값(가중치 + activation workspace + I/O)으로 낸다.
    cudaMemGetInfo 의 total-free 는 Orin 에서 **시스템 전체** 사용량이라 모델이
    달라도 4GB 근처로 똑같이 나온다(공유 메모리). 참고용으로만 같이 찍는다.
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

from infer_backends import make_runner, cuda_mem_probe


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
    # 기준선은 **CUDA 컨텍스트를 띄운 뒤, 엔진을 로드하기 전**에 잡는다.
    # 그래야 델타가 "엔진 가중치 + TRT persistent + activation + I/O" 가 되어
    # torch.cuda.max_memory_allocated() 와 같은 구성이 된다. 컨텍스트 이전에
    # 재면 컨텍스트(수백 MB)가 델타에 섞여 비교가 불가능해진다.
    probe = cuda_mem_probe()
    mem_fn = (probe.free_gb if probe else (lambda: None))

    def free_mb():
        v = mem_fn()
        return v[0] * 1024 if v else None

    # 단계별로 free 를 찍어 두면 메모리가 어디서 늘었는지 바로 보인다.
    stages = {}
    stages["ctx"] = free_mb()                      # CUDA 컨텍스트만 띄운 상태 = 기준선
    base_mb = stages["ctx"]
    floor_mb = base_mb

    runner = make_runner(args.backend, engine=path, onnx=path,
                         device=args.device, threads=args.threads)
    stages["engine"] = free_mb()                   # 엔진 역직렬화(가중치 적재) 후

    h, w = parse_resolution(args.resolution)
    rng = np.random.default_rng(0)
    left = rng.standard_normal((1, 3, h, w), dtype=np.float32)
    right = rng.standard_normal((1, 3, h, w), dtype=np.float32)

    split = hasattr(runner, "bind") and hasattr(runner, "execute")
    for k in ("engine",):
        if stages[k] is not None and floor_mb is not None:
            floor_mb = min(floor_mb, stages[k])

    if split:
        runner.bind(left, right)          # 복사는 한 번만
        stages["buffers"] = free_mb()     # 입출력 버퍼 할당 후
        step = runner.execute
        step()                            # 첫 실행: activation + CUDA 커널 모듈 적재
        stages["first_run"] = free_mb()
    else:
        # onnx 백엔드에는 분리가 없다. 복사가 포함된 end-to-end 시간이 된다.
        def step():
            runner(left, right)

    for _ in range(args.warmup):
        step()
    stages["warmup"] = free_mb()

    for k, v in stages.items():
        if v is not None and floor_mb is not None:
            floor_mb = min(floor_mb, v)

    times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        step()
        times.append((time.perf_counter() - t0) * 1000.0)
        if (i & 15) == 0:                  # 매번 부르면 측정에 영향을 준다
            cur = free_mb()
            if cur is not None and floor_mb is not None:
                floor_mb = min(floor_mb, cur)

    # 엔진 로드 직전 대비 free 의 최저점. torch.cuda.max_memory_allocated() 에
    # 대응하는 값이다.
    #
    # 한계: Orin 은 CPU 와 메모리를 공유해서 cudaMemGetInfo 가 **시스템 전체** free 를
    # 준다. 측정 중 다른 프로세스가 메모리를 잡으면 그만큼 부풀려진다. 단독으로
    # 돌리고, 아래 analytic 값(activation+io)과 크게 다르면 의심할 것.
    peak_alloc_gb = None
    if base_mb is not None and floor_mb is not None:
        peak_alloc_gb = max(0.0, base_mb - floor_mb) / 1024.0

    # 단계별 증가분 (기준선 대비 누적)
    breakdown = None
    if base_mb is not None:
        order = [("engine", "엔진 적재(가중치)"), ("buffers", "I/O 버퍼"),
                 ("first_run", "첫 실행(activation+커널모듈)"), ("warmup", "warm-up 후")]
        prev, breakdown = base_mb, []
        for key, label in order:
            v = stages.get(key)
            if v is None:
                continue
            breakdown.append((label, prev - v, base_mb - v))   # (구간 증가, 누적)
            prev = v

    size_mb = os.path.getsize(path) / (1024 ** 2)
    eng_mem = getattr(runner, "engine_memory_mb", lambda: {})()
    return {
        "name": os.path.basename(path), "backend": runner.name,
        "size_mb": size_mb, "latency": summarize(times),
        "peak_alloc_gb": peak_alloc_gb, "copies_excluded": split,
        "breakdown": breakdown,
        "activation_mb": eng_mem.get("activation_mb"),
        "io_mb": eng_mem.get("io_mb"),
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

    head = "{:<34}{:>9}{:>9}{:>9}{:>8}{:>10}{:>8}{:>11}".format(
        "engine", "size MB", "mean ms", "p95 ms", "FPS",
        "activ MB", "io MB", "peak MB")
    print("\n" + "=" * len(head), flush=True)
    print(head, flush=True)
    print("-" * len(head), flush=True)

    rows = []
    for path in paths:
        if not os.path.isfile(path):
            print("{:<34}{:>56}".format(os.path.basename(path), "파일 없음"), flush=True)
            continue
        print("{:<34}".format(os.path.basename(path)[:33]), end="", flush=True)
        try:
            r = bench_one(path, args)
        except Exception as e:
            print("{:>56}".format("FAILED: " + str(e)[:48]), flush=True)
            continue
        lat = r["latency"]
        fmt = lambda v: "{:.1f}".format(v) if v is not None else "-"   # onnx 백엔드엔 없다
        peak = r["peak_alloc_gb"] * 1024 if r["peak_alloc_gb"] is not None else None
        print("{:>9.1f}{:>9.3f}{:>9.3f}{:>8.1f}{:>10}{:>8}{:>11}".format(
            r["size_mb"], lat["mean"], lat["p95"], 1000.0 / lat["mean"],
            fmt(r["activation_mb"]), fmt(r["io_mb"]), fmt(peak)), flush=True)
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
        print("  peak MB : 엔진 로드 직전(CUDA 컨텍스트는 이미 띄운 상태) 대비 GPU 할당 최댓값.", flush=True)
        print("            torch.cuda.max_memory_allocated() 와 같은 구성이다", flush=True)
        print("            (가중치 + TRT persistent + activation + I/O, CUDA 컨텍스트 제외).", flush=True)
        print("            Orin 은 CPU 와 메모리를 공유하므로 단독 실행할 것.", flush=True)
        print("  activ/io: TRT 가 보고하는 값(계산값).", flush=True)
        for r in rows:
            if not r.get("breakdown"):
                continue
            print("\n  [{}] 단계별 GPU 메모리 증가 (MB)".format(r["name"]), flush=True)
            print("    {:<34}{:>10}{:>10}".format("단계", "증가", "누적"), flush=True)
            for label, delta, cum in r["breakdown"]:
                print("    {:<34}{:>10.1f}{:>10.1f}".format(label, delta, cum), flush=True)
            analytic = r["size_mb"] + (r["activation_mb"] or 0) + (r["io_mb"] or 0)
            print("    {:<34}{:>10}{:>10.1f}".format("(계산값 size+activ+io)", "", analytic),
                  flush=True)
            print("    차이는 대부분 cuDNN/cuBLAS 커널 모듈 지연 적재다.", flush=True)
            print("    torch.cuda.max_memory_allocated() 는 이걸 세지 않으므로,", flush=True)
            print("    torch 수치와 맞추려면 '첫 실행' 이전까지의 누적을 보는 게 가깝다.", flush=True)

    if args.csv and rows:
        path = "trt_runtime.csv" if args.csv == "AUTO" else args.csv
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["engine", "backend", "resolution", "warmup", "runs", "size_mb",
                           "mean_ms", "std_ms", "median_ms", "p95_ms", "min_ms", "fps",
                           "activation_mb", "io_mb", "peak_alloc_mb", "copies_excluded"])
            for r in rows:
                lat = r["latency"]
                wcsv.writerow([r["name"], r["backend"], args.resolution, args.warmup, args.runs,
                               round(r["size_mb"], 2), round(lat["mean"], 4),
                               round(lat["std"], 4), round(lat["median"], 4),
                               round(lat["p95"], 4), round(lat["min"], 4),
                               round(1000.0 / lat["mean"], 2),
                               round(r["activation_mb"], 2) if r["activation_mb"] is not None else "",
                               round(r["io_mb"], 2) if r["io_mb"] is not None else "",
                               round(r["peak_alloc_gb"] * 1024, 2) if r["peak_alloc_gb"] is not None else "",
                               r["copies_excluded"]])
        print("\nsaved: {}".format(path), flush=True)


if __name__ == "__main__":
    main()
