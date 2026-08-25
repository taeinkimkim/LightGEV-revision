import os, math, numpy as np, torch

def _np_stats(x):
    x = np.asarray(x)
    return dict(
        shape=tuple(x.shape),
        nan=bool(np.isnan(x).any()),
        inf=bool(np.isinf(x).any()),
        min=float(np.nanmin(x)) if x.size else float("nan"),
        max=float(np.nanmax(x)) if x.size else float("nan"),
        mean=float(np.nanmean(x)) if x.size else float("nan"),
    )

def _th_stats(t):
    t = t.detach()
    # NaN/Inf 확인
    is_float = t.is_floating_point()
    has_nan = bool((~torch.isfinite(t)).any().item()) if is_float else False
    has_inf = bool(torch.isinf(t).any().item()) if is_float else False

    # 안전한 통계 (NaN/Inf를 0으로 대체 후 min/max 계산)
    safe = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    # 평균은 NaN 무시한 평균
    if is_float and torch.isfinite(t).any():
        mean_val = float(torch.nanmean(t).item())
    else:
        mean_val = float(safe.float().mean().item())

    return dict(
        shape=tuple(t.shape),
        nan=has_nan,
        inf=has_inf,                     # ← 추가
        min=float(safe.min().item()),
        max=float(safe.max().item()),
        mean=mean_val,
    )

def _shape_hw(a):
    """
    Return (H, W) robustly.
    Works for numpy/tensor with shapes like:
      - (H, W)
      - (H, W, C) or (C, H, W)
      - (B, C, H, W) or (B, H, W, C)
    If ndim < 2, return (None, None).
    """
    import numpy as np, torch

    if isinstance(a, torch.Tensor):
        if a.ndim >= 2:
            H, W = a.shape[-2], a.shape[-1]
            return int(H), int(W)
        return (None, None)

    arr = np.asarray(a)
    if arr.ndim >= 2:
        H, W = arr.shape[-2], arr.shape[-1]
        return int(H), int(W)
    return (None, None)

def probe_batch(batch, max_disp=192, logger=None, where="train", strict=True):
    """
    batch(dict) 안의 대표 키들을 검사:
      - left/right(이미지), disp/disp_right(시차), depth(깊이), occ_mask 등
    이상 징후가 있으면 list[str]로 원인 메시지 반환.
    strict=True면 심각한 결함(Non-finite/형상 불일치/범위 폭주)이면 배치 스킵 권장.
    """
    issues = []
    name = None
    try:
        name = batch.get("name", ["unknown"])[0]
    except Exception:
        pass

    # 1) 좌/우 이미지 통계
    for k in ("left", "right"):
        if k in batch:
            x = batch[k]
            if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
            st = _np_stats(x)
            if st["nan"] or st["inf"]:
                issues.append(f"{where}:{k} has NaN/Inf {st}")
            H, W = _shape_hw(x)
            if H <= 0 or W <= 0:
                issues.append(f"{where}:{k} bad shape {st['shape']}")

    # 2) GT/Pred 맵들
    for k in ("disp", "disp_right", "depth", "occ_mask", "occ_mask_right"):
        if k not in batch: continue
        x = batch[k]
        if isinstance(x, torch.Tensor):
            st = _th_stats(x)
        else:
            st = _np_stats(x)
        if st["nan"] or st["inf"]:
            issues.append(f"{where}:{k} has NaN/Inf {st}")

        # 값 범위 기본 규칙
        if "disp" in k:
            if st["min"] < -1e-6: issues.append(f"{where}:{k} negative disp {st['min']}")
            if st["max"] > max_disp * 1.5: issues.append(f"{where}:{k} too large disp {st['max']} (> {max_disp})")
        if "depth" in k:
            if st["min"] <= 0: issues.append(f"{where}:{k} depth<=0 present (min={st['min']})")

    # 3) 형상 정합 (left와 다른 맵들이 같은 (H,W)인지)
    if "left" in batch:
        if isinstance(batch["left"], torch.Tensor):
            H, W = batch["left"].shape[-2:]
        else:
            H, W = _shape_hw(batch["left"])
        for k in ("disp", "disp_right", "depth", "occ_mask", "occ_mask_right"):
            if k in batch:
                h2, w2 = _shape_hw(batch[k])
                if (h2, w2) != (H, W):
                    issues.append(f"{where}:{k} shape {h2,w2} != left {(H,W)}")

    # 4) 결과 로깅
    if issues and logger:
        logger.warning(f"[DataProbe] name={name} | " + " | ".join(issues))
    return issues