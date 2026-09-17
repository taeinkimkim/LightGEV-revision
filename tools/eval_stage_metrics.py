"""Stage-wise stereo evaluation: D0 (aggregation) vs DT (refiner) vs final (context upsample).

eval_single.py 는 최종 예측(disp_pred)만 리포트한다. 그런데 LightGEV 계열에서
"gate 가 매칭 품질을 개선했는가"와 "refiner 가 서브픽셀을 깎았는가"는 서로 다른 질문이고,
최종 지표 하나로는 구분되지 않는다. 이 도구는 파이프라인을 단계별로 끊어서 잰다.

  D0    aggregation 직후의 disparity (cost volume argmax/soft-argmin)
        -> gate 의 담당 영역. D1(outlier) 로 본다.
  DT    refiner 를 T번 돌린 뒤
        -> refiner 의 담당 영역. EPE(서브픽셀) 로 본다.
  final context upsample 까지 통과한 최종 예측

D0/DT 는 둘 다 bilinear x4 로 올려서 비교한다 (context upsampler 가 교란변수가 되지 않도록).
final 만 모델 고유의 upsample 을 쓴다.

cost_agg 출력에 hook 을 걸어 D0 를 계산하므로 LightStereo / LightGEV / LightGEVModify 를
동일한 기준으로 비교할 수 있다 (LightStereo 는 disp_init 을 result 에 내보내지 않는다).

Usage:
    python tools/eval_stage_metrics.py \
        --model_name <name> --cfg <yaml> --ckpt <pth> --output_root <dir>
"""

import os
import sys
import csv
import argparse
import datetime
import logging
import random

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, './')
from stereo.utils import common_utils
from stereo.runners.single_runner import _build_trainer_from_cfg
from stereo.modeling.disp_pred.disp_regression import disparity_regression
from stereo.evaluation.metric_per_image import d1_metric, epe_metric, threshold_metric


def parse_args():
    p = argparse.ArgumentParser('Stage-wise stereo evaluation (D0 / DT / final)')
    p.add_argument('--model_name', type=str, required=True)
    p.add_argument('--cfg', type=str, required=True)
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--output_root', type=str, required=True)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--max_disp', type=int, default=192)
    p.add_argument('--gt_mask', type=str, default='positive',
                   choices=['positive', 'in_range'],
                   help="유효 픽셀 정의. positive: gt>0 (eval_single.py 와 동일, 기본값) / "
                        "in_range: 0<gt<max_disp (모델이 예측 가능한 범위로 제한)")
    p.add_argument('--max_samples', type=int, default=-1,
                   help='앞에서 N장만 평가 (기본 -1 = 전체)')
    p.add_argument('--seed', type=int, default=666)
    p.add_argument('--no_csv', action='store_true')
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StageAccumulator:
    """스테이지별 per-image 지표와 픽셀 가중 global 지표를 함께 모은다."""

    def __init__(self):
        self.per_image = {}     # stage -> {metric: [값들]}
        self.err_sum = {}       # stage -> |err| 합 (global EPE 용)
        self.bad3_sum = {}      # stage -> >3px 픽셀 수
        self.npix = 0

    def add(self, stage, pred, gt, mask):
        """pred/gt/mask: [B, H, W]"""
        d = self.per_image.setdefault(stage, {})
        d.setdefault('epe', []).extend(epe_metric(pred, gt, mask).tolist())
        d.setdefault('d1', []).extend(d1_metric(pred, gt, mask).tolist())
        d.setdefault('bad1', []).extend(threshold_metric(pred, gt, mask, 1).tolist())
        d.setdefault('bad3', []).extend(threshold_metric(pred, gt, mask, 3).tolist())

        e = (pred - gt).abs()[mask]
        self.err_sum[stage] = self.err_sum.get(stage, 0.0) + float(e.sum())
        self.bad3_sum[stage] = self.bad3_sum.get(stage, 0.0) + float((e > 3).sum())

    def set_npix(self, n):
        self.npix += n

    def stages(self):
        return list(self.per_image.keys())

    def summary(self, stage):
        d = self.per_image[stage]
        out = {k: float(np.mean(v)) for k, v in d.items()}
        out['global_epe'] = self.err_sum[stage] / max(self.npix, 1)
        out['global_bad3'] = self.bad3_sum[stage] / max(self.npix, 1) * 100
        return out


def build_gate_probe(model):
    """모델이 가진 geo gate 종류에 맞는 진단 hook 을 단다.

    LightGEVModify : gate_head -> 실제 곱수 (1 + r*tanh) 분포
    LightGEV(원본) : geo_to_cost -> |alpha*geo_cost| / |gwc| 비율
    그 외(LightStereo) : 없음
    """
    cap, stats = {}, {}

    if getattr(model, 'geo_encoder', None) is not None:
        model.geo_encoder.register_forward_pre_hook(
            lambda mod, i: cap.__setitem__('gwc', i[0].detach()))

    if getattr(model, 'gate_head', None) is not None:
        r = float(getattr(model, 'gate_range', 0.5))

        def hook(mod, i, o):
            m = 1.0 + r * torch.tanh(o.detach().float())
            stats.setdefault('gate_dev_mean', []).append(float((m - 1.0).abs().mean()))
            stats.setdefault('gate_mult_min', []).append(float(m.min()))
            stats.setdefault('gate_mult_max', []).append(float(m.max()))

        model.gate_head.register_forward_hook(hook)
        return stats, 'multiplicative (LightGEVModify)'

    if getattr(model, 'geo_to_cost', None) is not None:
        def hook(mod, i, o):
            a = model.geo_gate_alpha
            a = float(a) if a is not None else 1.0
            gc = (o.detach().float() * a).abs().mean()
            gwc = cap.get('gwc')
            stats.setdefault('geo_over_gwc', []).append(
                float(gc / (gwc.float().abs().mean() + 1e-9)) if gwc is not None else float('nan'))

        model.geo_to_cost.register_forward_hook(hook)
        return stats, 'additive (LightGEV original)'

    return stats, 'none'


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = os.path.join(args.output_root, args.model_name)
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(
        out_dir, 'stage_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=0)

    logger.info('=' * 80)
    logger.info('Stage-wise evaluation (D0 / DT / final)')
    logger.info('=' * 80)
    logger.info('Model     : %s' % args.model_name)
    logger.info('Config    : %s' % args.cfg)
    logger.info('Checkpoint: %s' % args.ckpt)
    logger.info('Output dir: %s' % out_dir)
    logger.info('GT mask   : %s' % ('gt>0 (eval_single.py 와 동일)' if args.gt_mask == 'positive'
                                    else '0<gt<%d' % args.max_disp))

    quiet = logging.getLogger('stage_eval_build')
    quiet.addHandler(logging.NullHandler())
    trainer = _build_trainer_from_cfg(args.cfg, args.ckpt, out_dir, logger=quiet)

    model = trainer.model.to(args.device).eval()
    core = model.module if hasattr(model, 'module') else model
    loader = trainer.test_loader

    cap = {}
    core.cost_agg.register_forward_hook(
        lambda mod, i, o: cap.__setitem__('agg', (o[0] if isinstance(o, (list, tuple)) else o).detach()))
    gate_stats, gate_kind = build_gate_probe(core)
    logger.info('Geo gate  : %s' % gate_kind)

    acc = StageAccumulator()
    rows = []
    n = 0

    for data in loader:
        for k, v in data.items():
            if torch.is_tensor(v):
                data[k] = v.to(args.device)

        with torch.no_grad():
            pred = model(data)

        gt = data['disp']
        if gt.dim() == 4:
            gt = gt.squeeze(1)
        mask = gt > 0
        if args.gt_mask == 'in_range':
            mask = mask & (gt < args.max_disp)
        if mask.sum() < 1:
            continue
        H, W = gt.shape[-2:]

        def up4(d):
            """1/4 해상도 disparity -> full 해상도 (bilinear x4)"""
            if d.dim() == 3:
                d = d.unsqueeze(1)
            return (F.interpolate(d, (H, W), mode='bilinear', align_corners=False) * 4.0).squeeze(1)

        stage_preds = {}

        # --- D0: aggregation 직후 (모든 모델에서 동일하게 hook 으로 계산) ---
        c = cap['agg'].float()
        prob = F.softmax(c.reshape(c.size(0), -1, c.size(2), c.size(3)), dim=1)
        d0 = disparity_regression(prob, args.max_disp // 4)
        stage_preds['D0_bilinear'] = up4(d0)

        # --- 반복 단계별 (refiner 가 있을 때만) ---
        seq = pred.get('disp_seq_1_4')
        if seq is not None and len(seq) > 1:
            for i, ds in enumerate(seq):
                stage_preds['iter%d_bilinear' % (i + 1)] = up4(ds.float())

        # --- DT: refiner 통과 후 ---
        if 'disp_ref_low' in pred:
            stage_preds['DT_bilinear'] = up4(pred['disp_ref_low'].float())

        # --- final: 모델 고유의 upsample ---
        fp = pred['disp_pred'].float()
        stage_preds['final'] = fp.squeeze(1) if fp.dim() == 4 else fp

        for name, p in stage_preds.items():
            acc.add(name, p, gt, mask)
        acc.set_npix(int(mask.sum()))

        row = {'index': n}
        for name, p in stage_preds.items():
            row['%s_epe' % name] = float(epe_metric(p, gt, mask).mean())
            row['%s_d1' % name] = float(d1_metric(p, gt, mask).mean())
        rows.append(row)

        n += 1
        if n % 200 == 0:
            logger.info('  %d samples' % n)
        if 0 < args.max_samples <= n:
            break

    if n == 0:
        logger.info('평가할 샘플이 없습니다.')
        return

    logger.info('')
    logger.info('-' * 88)
    logger.info('%-18s %10s %10s %10s %10s %12s %12s'
                % ('stage', 'EPE', 'D1(%)', 'bad1(%)', 'bad3(%)', 'globalEPE', 'globalBad3'))
    logger.info('-' * 88)

    order = (['D0_bilinear']
             + sorted([s for s in acc.stages() if s.startswith('iter')],
                      key=lambda s: int(s[4:s.index('_')]))
             + [s for s in ['DT_bilinear', 'final'] if s in acc.stages()])

    for s in order:
        m = acc.summary(s)
        logger.info('%-18s %10.4f %10.4f %10.4f %10.4f %12.4f %12.4f'
                    % (s, m['epe'], m['d1'], m['bad1'], m['bad3'],
                       m['global_epe'], m['global_bad3']))
    logger.info('-' * 88)
    logger.info('samples=%d, valid pixels=%d  (per-image 평균 / 픽셀가중 global 둘 다 표기)'
                % (n, acc.npix))

    if gate_stats:
        logger.info('')
        logger.info('[geo gate] %s' % gate_kind)
        for k, v in gate_stats.items():
            logger.info('  %-16s mean=%.4f  min=%.4f  max=%.4f'
                        % (k, float(np.mean(v)), float(np.min(v)), float(np.max(v))))

    if not args.no_csv:
        csv_path = os.path.join(out_dir, 'stage_metrics.csv')
        keys = ['index'] + [k for k in rows[0].keys() if k != 'index']
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
            summary = {'index': 'SUMMARY'}
            for s in order:
                m = acc.summary(s)
                summary['%s_epe' % s] = '%.6f' % m['epe']
                summary['%s_d1' % s] = '%.6f' % m['d1']
            w.writerow(summary)
        logger.info('')
        logger.info('CSV saved to %s' % csv_path)


if __name__ == '__main__':
    main()
