"""추론 효율 프로파일링: 파라미터 / 메모리 / FLOPs / latency 를 한 번에 측정한다.

논문의 efficiency 표에 필요한 항목을 한 곳에서, 일관된 조건으로 잰다.
eval_single.py 도 InferTime/PeakMem/FLOPs 를 찍지만 그쪽은 평가 루프 안에서
warm-up 없이 샘플마다 재기 때문에 첫 샘플에 CUDA 초기화·cuDNN autotune 비용이
섞인다. 여기서는 warm-up 후 고정 횟수를 재고 분산까지 보고한다.

측정 항목
  - 총 파라미터 / 학습 가능 파라미터 / buffer(BN running stat 등)
  - 모델 크기 (파라미터+buffer 바이트, 체크포인트 파일 크기)
  - FLOPs (ptflops. MACs 와 2*MACs 를 모두 표기)
  - latency: warm-up 후 N회, mean/std/median/p95/min, FPS
  - 메모리: 가중치 상주량, 추론 peak allocated/reserved, 활성값(peak - 가중치)
  - 입력 해상도 / 수치 정밀도 / warm-up 횟수 / 측정 횟수 (실험 조건 기록)

사용법
  python tools/profile_model.py --cfg cfgs/a.yaml cfgs/b.yaml \
      --input_size 384 1248 --precision fp32 --warmup 50 --runs 200
  python tools/profile_model.py --cfg <yaml> --ckpt <pth> --out report.csv
"""

import os
import sys
import csv
import json
import time
import argparse
import importlib

import numpy as np
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict

sys.path.insert(0, './')


# --------------------------------------------------------------------------
# 모델 레지스트리: 각 모델 패키지의 trainer.py 에 있는 __all__ 을 합친다
# --------------------------------------------------------------------------
def build_model_registry():
    reg = {}
    root = 'stereo/modeling/models'
    for pkg in sorted(os.listdir(root)):
        tp = os.path.join(root, pkg, 'trainer.py')
        if not os.path.isfile(tp):
            continue
        try:
            mod = importlib.import_module(f'stereo.modeling.models.{pkg}.trainer')
        except Exception as e:  # noqa: BLE001
            print(f'[warn] {pkg} import 실패: {e}')
            continue
        for k, v in getattr(mod, '__all__', {}).items():
            reg[k] = v
    return reg


def load_cfg(path):
    return EasyDict(yaml.safe_load(open(path)))


# --------------------------------------------------------------------------
# 개별 측정
# --------------------------------------------------------------------------
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    bnum = sum(b.numel() for b in model.buffers())
    bbytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return dict(params_total=total, params_learnable=train, params_frozen=total - train,
                buffers=bnum, model_size_mb=(pbytes + bbytes) / 1024 ** 2)


def count_flops(model, shape):
    """ptflops 로 MACs 를 센다. shape = (C, H, W)."""
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        return dict(macs_g=float('nan'), flops_g=float('nan'))

    device = next(model.parameters()).device

    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            if isinstance(x, (tuple, list)):
                x = x[0]
            return self.m({'left': x, 'right': x})

    w = Wrapper(model).to(device).eval()
    try:
        with torch.no_grad():
            macs, _ = get_model_complexity_info(
                w, input_res=shape, as_strings=False,
                print_per_layer_stat=False, verbose=False)
        macs = float(macs)
    except Exception as e:  # noqa: BLE001
        print(f'[warn] FLOPs 계산 실패: {e}')
        return dict(macs_g=float('nan'), flops_g=float('nan'))
    # ptflops 는 MAC(곱셈-누산) 단위. 통상 FLOPs = 2 * MACs 로 환산한다.
    return dict(macs_g=macs / 1e9, flops_g=2.0 * macs / 1e9)


def measure_latency(model, data, warmup, runs, device, amp):
    """warm-up 후 CUDA event 로 runs 회 측정."""
    cuda = device.type == 'cuda'
    ctx = torch.amp.autocast('cuda', enabled=amp) if cuda else torch.amp.autocast('cpu', enabled=False)

    with torch.no_grad():
        for _ in range(warmup):
            with ctx:
                model(data)
        if cuda:
            torch.cuda.synchronize()

        ts = []
        if cuda:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(runs):
                s.record()
                with ctx:
                    model(data)
                e.record()
                torch.cuda.synchronize()
                ts.append(s.elapsed_time(e))          # ms
        else:
            for _ in range(runs):
                t0 = time.perf_counter()
                model(data)
                ts.append((time.perf_counter() - t0) * 1e3)

    t = np.asarray(ts)
    return dict(latency_mean_ms=float(t.mean()), latency_std_ms=float(t.std()),
                latency_median_ms=float(np.median(t)), latency_p95_ms=float(np.percentile(t, 95)),
                latency_min_ms=float(t.min()), fps=float(1000.0 / t.mean()))


def measure_memory(model, data, device, amp):
    """가중치 상주량, 추론 peak allocated/reserved, 활성값 사용량."""
    if device.type != 'cuda':
        return dict(weights_mem_mb=float('nan'), peak_alloc_mb=float('nan'),
                    peak_reserved_mb=float('nan'), activation_mem_mb=float('nan'))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    weights = torch.cuda.memory_allocated()                 # 모델만 올라간 상태
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp):
        model(data)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    MB = 1024 ** 2
    return dict(weights_mem_mb=weights / MB, peak_alloc_mb=peak / MB,
                peak_reserved_mb=reserved / MB,
                activation_mem_mb=(peak - weights) / MB)


# --------------------------------------------------------------------------
def profile_one(cfg_path, args, registry):
    cfg = load_cfg(cfg_path)
    name = cfg.MODEL.NAME
    if name not in registry:
        raise KeyError(f'{name} 이 레지스트리에 없습니다. 등록된 것: {sorted(registry)}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = registry[name](cfg.MODEL).to(device).eval()

    if args.ckpt:
        from stereo.utils.common_utils import load_params_from_file
        load_params_from_file(model, args.ckpt, str(device), False, None, strict=False)

    # 수치 정밀도
    amp = args.precision == 'amp'
    if args.precision == 'fp16':
        model = model.half()
    dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'amp': torch.float32}[args.precision]

    H, W = args.input_size
    data = {'left': torch.randn(args.batch, 3, H, W, device=device, dtype=dtype),
            'right': torch.randn(args.batch, 3, H, W, device=device, dtype=dtype)}

    row = dict(config=os.path.basename(cfg_path), model=name)
    row.update(count_params(model))
    row['ckpt_size_mb'] = (os.path.getsize(args.ckpt) / 1024 ** 2) if args.ckpt and os.path.isfile(args.ckpt) else float('nan')
    # 실험 조건도 함께 기록한다 (재현성)
    row.update(input_h=H, input_w=W, batch=args.batch, precision=args.precision,
               warmup=args.warmup, runs=args.runs,
               cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
               device=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
               torch=torch.__version__)
    row.update(count_flops(model, (3, H, W)))
    row.update(measure_memory(model, data, device, amp))
    row.update(measure_latency(model, data, args.warmup, args.runs, device, amp))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cfg', nargs='+', required=True, help='config yaml (여러 개 가능)')
    p.add_argument('--ckpt', type=str, default=None,
                   help='선택. 가중치는 속도/메모리에 영향을 주지 않지만 ckpt 파일 크기를 함께 기록한다')
    p.add_argument('--input_size', type=int, nargs=2, default=[384, 1248], metavar=('H', 'W'))
    p.add_argument('--batch', type=int, default=1, help='latency 는 보통 batch 1 로 보고한다')
    p.add_argument('--precision', choices=['fp32', 'fp16', 'amp'], default='fp32')
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--runs', type=int, default=200)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--cudnn_benchmark', action='store_true', default=True,
                   help='배포 시 관행대로 autotune 을 켠다. warm-up 이 이를 흡수한다')
    p.add_argument('--no_cudnn_benchmark', dest='cudnn_benchmark', action='store_false')
    p.add_argument('--out', type=str, default=None, help='CSV 저장 경로')
    args = p.parse_args()

    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    registry = build_model_registry()

    rows = [profile_one(c, args, registry) for c in args.cfg]

    print('\n' + '=' * 100)
    print(f'입력 {args.input_size[0]}x{args.input_size[1]}  batch {args.batch}  '
          f'{args.precision}  warm-up {args.warmup}  runs {args.runs}  '
          f'cudnn.benchmark={args.cudnn_benchmark}')
    print(f'device: {rows[0]["device"]}   torch {rows[0]["torch"]}')
    print('=' * 100)
    hdr = ['config', 'params_total', 'params_learnable', 'model_size_mb', 'macs_g', 'flops_g',
           'latency_mean_ms', 'latency_std_ms', 'fps', 'peak_alloc_mb', 'activation_mem_mb']
    w = [38, 13, 16, 13, 8, 9, 15, 14, 7, 13, 17]
    print('  '.join(h.rjust(x) for h, x in zip(hdr, w)))
    for r in rows:
        cells = []
        for h, x in zip(hdr, w):
            v = r[h]
            cells.append((v if isinstance(v, str) else
                          (f'{v:,}' if isinstance(v, int) else f'{v:.3f}')).rjust(x))
        print('  '.join(cells))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
        with open(args.out, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f'\nCSV 저장: {args.out}')
        print(json.dumps(rows[0], indent=2, ensure_ascii=False, default=str)[:400] + ' ...')


if __name__ == '__main__':
    main()
