"""여러 모델의 전체 inference runtime 을 같은 조건으로 재서 비교한다.

구간별 분해는 하지 않는다 (그건 tools/profile_stage_runtime.py). 여기서는
모델 하나당 total latency / FPS / model size / learnable params /
peak inference memory 만 본다.

기본 비교 대상:
    ours : LightGEVModify_4 (full, GEV_ITERS 6, r=0.75)
    2D   : BANet-2D, MobileStereoNet-2D (MSNet2D)
    3D   : GwcNet, ACVNet, IGEV++

--- 측정 방법 ---

checkpoint 는 쓰지 않는다. runtime 은 "어떤 커널이 어떤 shape 으로 몇 번 도는가"
로 정해지고 가중치 '값' 과는 무관하다 (실측: 학습본 로드 전후 9.873 vs 9.862 ms,
측정 노이즈 수준). 그래서 cfg 의 MODEL 블록으로 구조만 만들어 랜덤 입력으로 잰다.
단 이 논리는 forward 에 값에 따라 갈라지는 분기가 없을 때만 성립한다 - 여기
모델들의 분기는 전부 config 와 self.training 으로 정해진다.

모델은 반드시 eval() 로 둔다. IGEV++ 는 `test_mode = not self.training` 으로
반복 횟수를 VALID_ITERS / TRAIN_ITERS 중에 고르므로, 모드가 틀리면 시간이
완전히 달라진다.

warm-up 과 반복 횟수: 첫 수십 회는 cuDNN algo 선택, 할당자 캐시, GPU 클럭
상승 때문에 느리다. 기본값 warmup 50 / runs 200.

peak memory 는 warm-up 이 끝난 뒤 리셋하고 잰다. cudnn.benchmark 가 켜져 있으면
warm-up 중 algo 탐색이 steady-state 의 20 배까지 workspace 를 잡는데, 그건
추론에 실제로 필요한 메모리가 아니다.

Usage:
    python tools/profile_model_runtime.py
    python tools/profile_model_runtime.py --resolution 544x960 384x1248
    python tools/profile_model_runtime.py --models ours gwcnet --runs 500
    python tools/profile_model_runtime.py --csv
"""

import argparse
import csv
import importlib
import os
import sys
import traceback

import torch
import yaml
from easydict import EasyDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from profile_stage_runtime import model_stats, summarize, make_input, parse_resolution

# cfg 의 MODEL.NAME -> (모듈, 클래스). stereo/modeling/__init__.py 의 Trainer 매핑과
# 같은 대상이지만, Trainer 는 dataloader 까지 만들므로 모델 클래스만 직접 쓴다.
MODEL_CLASSES = {
    "MSNet2D":          ("stereo.modeling.models.msnet.MSNet2D", "MSNet2D"),
    "IGEV_PP":          ("stereo.modeling.models.igev_pp.igev_stereo", "IGEVStereo"),
    "GwcNet":           ("stereo.modeling.models.gwcnet.gwcnet", "GwcNet"),
    "ACVNet":           ("stereo.modeling.models.acvnet.acv", "ACVNet"),
    "BANet":            ("stereo.modeling.models.banet_2d.BANet", "BANet"),
    "LightStereo":      ("stereo.modeling.models.lightstereo.lightstereo", "LightStereo"),
    "LightGEVModify_4": ("stereo.modeling.models.lightgev.lightgev_modify_4", "LightGEVModify_4"),
}

# (label, cfg 경로, 그룹)
ENTRIES = [
    ("lightgev (ours)", "cfgs/lightgev_modify_4/lightgev_modify_4_eval_sceneflow_full_iter_6_r_075.yaml", "2D"),
    ("lightstereo_m", "cfgs/lightstereo/lightstereo_m_eval_sceneflow.yaml", "2D"),
    ("banet_2d", "cfgs/banet_2d/banet_eval_sceneflow.yaml",   "2D"),
    ("msnet_2d", "cfgs/msnet/msnet_eval_sceneflow.yaml",      "2D"),
    ("gwcnet",   "cfgs/gwcnet/gwcnet_eval_sceneflow.yaml",    "3D"),
    ("acvnet",   "cfgs/acvnet/acvnet_eval_sceneflow.yaml",    "3D"),
    ("igev_pp",  "cfgs/igev_pp/igev_pp_eval_sceneflow.yaml",  "3D"),
]

PRECISIONS = ("fp32", "amp", "fp16")


def build_model(cfg):
    name = cfg.MODEL.NAME
    if name not in MODEL_CLASSES:
        raise KeyError("MODEL.NAME '{}' 이 MODEL_CLASSES 에 없다. "
                       "stereo/modeling/__init__.py 를 보고 추가할 것.".format(name))
    module_path, cls_name = MODEL_CLASSES[name]
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(cfg.MODEL)


def config_notes(cfg):
    """모델마다 runtime 을 좌우하는 핵심 설정을 한 줄로. 표에 같이 찍어 조건을 드러낸다."""
    m = cfg.MODEL
    bits = ["MAX_DISP={}".format(m.get("MAX_DISP", "?"))]
    for k in sorted(m.keys()):
        if "ITERS" in k.upper() and "TRAIN" not in k.upper():
            bits.append("{}={}".format(k, m[k]))
    return " ".join(bits)


def run_loop(model, data, precision, warmup, runs):
    """(latency 리스트[ms], (peak_alloc, peak_reserved) GB)."""
    autocast = torch.amp.autocast("cuda", enabled=(precision == "amp"))
    with torch.no_grad():
        for _ in range(warmup):
            with autocast:
                model(data)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()   # warm-up(autotune) 이후부터의 peak

        times = []
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with autocast:
                model(data)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    return times, (torch.cuda.max_memory_allocated() / (1024 ** 3),
                   torch.cuda.max_memory_reserved() / (1024 ** 3))


def profile_one(label, cfg_path, group, h, w, precision, args, device):
    cfg = EasyDict(yaml.safe_load(open(cfg_path)))
    model = build_model(cfg).to(device).eval()      # eval() 필수 (IGEV++ 의 VALID_ITERS)
    if precision == "fp16":
        model = model.half()
    dtype = torch.float16 if precision == "fp16" else torch.float32
    data = make_input(h, w, device, dtype)

    stats = model_stats(model)
    torch.cuda.empty_cache()
    times, (peak_alloc, peak_reserved) = run_loop(model, data, precision, args.warmup, args.runs)

    del model, data
    torch.cuda.empty_cache()

    return {
        "label": label, "group": group, "model_name": cfg.MODEL.NAME,
        "notes": config_notes(cfg),
        "resolution": "{}x{}".format(h, w), "precision": precision,
        "stats": stats, "latency": summarize(times),
        "peak_alloc_gb": peak_alloc, "peak_reserved_gb": peak_reserved,
    }


def print_table(results, h, w, precision):
    print("\n" + "=" * 108)
    print(" input {}x{}  |  {}".format(h, w, precision))
    print("=" * 108)
    head = "{:<6}{:<17}{:<18}{:>11}{:>9}{:>9}{:>9}{:>10}{:>10}".format(
        "group", "label", "model", "params(M)", "size MB", "mean ms", "p95 ms", "FPS", "peak GB")
    print(head)
    print("-" * len(head))
    for r in results:
        if "error" in r:
            print("{:<6}{:<17}{:<18}{:>58}".format(r["group"], r["label"], r.get("model_name", "?"),
                                                   "FAILED: " + r["error"][:50]))
            continue
        lat = r["latency"]
        print("{:<6}{:<17}{:<18}{:>11.2f}{:>9.1f}{:>9.3f}{:>9.3f}{:>10.1f}{:>10.3f}".format(
            r["group"], r["label"], r["model_name"],
            r["stats"]["params_learnable"] / 1e6, r["stats"]["size_mb"],
            lat["mean"], lat["p95"], 1000.0 / lat["mean"], r["peak_alloc_gb"]))
    print("-" * len(head))
    for r in results:
        if "error" not in r:
            print("  {:<17}{}".format(r["label"], r["notes"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=None,
                        help="측정할 label 부분집합 (기본: 전부). 가능: "
                             + ", ".join(e[0] for e in ENTRIES))
    parser.add_argument("--cfg", nargs="+", default=None, metavar="LABEL=PATH",
                        help="특정 label 의 cfg 경로를 덮어쓴다 (예: ours=cfgs/.../x.yaml)")
    parser.add_argument("--resolution", nargs="+", default=["544x960"],
                        help="HxW 목록 (기본: 544x960 = SceneFlow 평가 해상도)")
    parser.add_argument("--precision", nargs="+", default=["fp32"], choices=PRECISIONS,
                        help="평가 파이프라인이 amp=False 라 fp32 가 기본값")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--cudnn-benchmark", dest="cudnn_benchmark",
                        action="store_true", default=True)
    parser.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--csv", nargs="?", const="AUTO", default=None,
                        help="csv 로 저장. 값 없이 주면 model_runtime.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA 가 필요하다 (CUDA event 로 측정한다).")

    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    device = torch.device(args.device)

    overrides = {}
    for item in (args.cfg or []):
        label, _, path = item.partition("=")
        overrides[label] = path

    entries = [e for e in ENTRIES if args.models is None or e[0] in args.models]
    entries = [(l, overrides.get(l, c), g) for l, c, g in entries]
    if not entries:
        raise SystemExit("측정할 모델이 없다. --models 값을 확인할 것.")

    print("device     : {}  ({})".format(args.device, torch.cuda.get_device_name(device)))
    print("torch      : {}  (cuda {})".format(torch.__version__, torch.version.cuda))
    print("warm-up    : {}   runs: {}   cudnn.benchmark: {}".format(
        args.warmup, args.runs, args.cudnn_benchmark))
    print("checkpoint : 쓰지 않음 (runtime 은 가중치 값과 무관)")

    all_results = []
    for res in args.resolution:
        h, w = parse_resolution(res)
        for precision in args.precision:
            results = []
            for label, cfg_path, group in entries:
                if not os.path.isfile(cfg_path):
                    results.append({"label": label, "group": group,
                                    "error": "cfg 없음: " + cfg_path})
                    continue
                try:
                    results.append(profile_one(label, cfg_path, group, h, w,
                                               precision, args, device))
                except Exception as e:  # OOM 이나 미지원 조합에서 전체가 죽지 않게
                    torch.cuda.empty_cache()
                    results.append({"label": label, "group": group,
                                    "error": "{}: {}".format(type(e).__name__, e)})
                    traceback.print_exc(limit=1)
            print_table(results, h, w, precision)
            all_results.extend(results)

    if args.csv:
        path = "model_runtime.csv" if args.csv == "AUTO" else args.csv
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", newline="") as f:
            w_ = csv.writer(f)
            w_.writerow(["group", "label", "model_name", "resolution", "precision",
                         "params_learnable", "size_mb", "mean_ms", "std_ms", "p95_ms",
                         "fps", "peak_alloc_gb", "peak_reserved_gb", "notes"])
            for r in all_results:
                if "error" in r:
                    w_.writerow([r["group"], r["label"], "", "", "", "", "", "", "", "",
                                 "", "", "", r["error"]])
                    continue
                lat = r["latency"]
                w_.writerow([r["group"], r["label"], r["model_name"], r["resolution"],
                             r["precision"], r["stats"]["params_learnable"],
                             round(r["stats"]["size_mb"], 3), round(lat["mean"], 4),
                             round(lat["std"], 4), round(lat["p95"], 4),
                             round(1000.0 / lat["mean"], 2), round(r["peak_alloc_gb"], 4),
                             round(r["peak_reserved_gb"], 4), r["notes"]])
        print("\nsaved: {}".format(path))


if __name__ == "__main__":
    main()


'''
python tools/profile_model_runtime.py --resolution 544x960 --precision fp32 --csv
'''
