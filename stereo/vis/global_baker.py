"""
Global visual baking helpers:
- extract_global_params: GlobalStatsManager.update_all 반환에서 vmin/vmax/tau 추출
- bake_global_visuals:   cache_npz를 읽어 글로벌 스케일로 disp/depth/error/region/left 저장
"""

from pathlib import Path
import numpy as np
import cv2
import os
import torch.nn.functional as F

from stereo.vis.disp_color import disp_to_color_bgr
from stereo.io.path_utils import ensure_dirs
from stereo.vis.overlay import ErrorOverlay
from stereo.evaluation.region_visualizer import RegionVisualizer

# ---------- small utilities ----------
def _imwrite_bgr(path: Path, img_bgr):
    cv2.imwrite(str(path), img_bgr)

# ---------- public helpers ----------
def extract_global_params(stats) -> tuple[float, float, float]:
    """
    stats: GlobalStatsManager.update_all(...) 반환 딕셔너리
    return: (vmin, vmax, tau)
    """
    if stats is None:
        return 0.0, 192.0, 3.0
    vr = stats["vrange"]
    vmin = float(vr["vmin"])
    vmax = float(vr["vmax"])
    tau = float(stats["tau"])
    return vmin, vmax, tau

def _collect_masks_from_npz(z) -> dict:
    out = {}
    if not hasattr(z, "files"):
        return out
    for k in z.files:
        if k.startswith("mask::"):
            out[k[len("mask::"):]] = (z[k] > 0)
    return out

def bake_global_visuals(
    output_root: str | Path,
    model_names: list[str],
    vmin: float,
    vmax: float,
    tau_px: float,
    *,
    region_viz: RegionVisualizer | None = None,
    logger=None,
    err_bg_percentile: float = 99.5,   # error heat 배경 클리핑 퍼센타일
):
    """
    cache_npz/*.npz를 읽어 글로벌 vmin/vmax/tau로 시각화 png를 저장한다.

    생성물:
      <root>/<MODEL>/
        left_bgr/*.png
        disp_vis_global/*.png
        depth_vis_global/*.png
        error_vis/*.png
        region_overlay/*.png
    """
    if region_viz is None:
        region_viz = RegionVisualizer()
    ol = ErrorOverlay()

    if logger:
        logger.info(f"[BAKE] using global vmin={vmin}, vmax={vmax}, tau_px={tau_px}")

    for name in model_names:
        out_dir         = output_root
        io              = ensure_dirs(out_dir)
        cache_dir       = io["cache_npz"]
        out_left        = io["left_bgr"]
        out_disp        = io["disp_vis_global"]
        out_err         = io["error_vis_global"]
        out_region      = io["region_overlay"]
        out_region_err  = io["region_error_vis_global"]

        for npz in sorted(Path(cache_dir).glob("*.npz")):
            z = np.load(str(npz), allow_pickle=True)
            stem       = npz.stem
            left_bgr   = z.get("left_bgr", None)
            disp_pred  = z.get("disp_pred", None)
            disp_gt    = z.get("disp_gt", None)
            masks      = _collect_masks_from_npz(z)

            # 1) left
            if left_bgr is not None:
                _imwrite_bgr(os.path.join(out_left, f"{stem}.png"), left_bgr)

            # 2) disparity (global vmin/vmax)
            if disp_pred is not None:
                disp_bgr = disp_to_color_bgr(np.clip(disp_pred, vmin, vmax), max_disp=vmax)
                _imwrite_bgr(os.path.join(out_disp, f"{stem}.png"), disp_bgr)

            # 4) region overlay (GT 기반, 글로벌 스케일 불필요)
            if left_bgr is not None and masks is not None:
                region_bgr = region_viz.overlay_regions(left_bgr, masks)
                _imwrite_bgr(os.path.join(out_region, f"{stem}.png"), region_bgr)

            # 5) error overlay (글로벌 τ 컨투어가 핵심)
            if left_bgr is not None and disp_pred is not None and disp_gt is not None:
                err = np.abs(disp_pred.astype(np.float32) - disp_gt.astype(np.float32))
                if np.isfinite(err).any():
                    vmax_bg = float(np.nanpercentile(err, err_bg_percentile))
                else:
                    vmax_bg = 1.0
                err_bgr = ol.make_overlay(
                    left_bgr, err, vmin=0.0, vmax=vmax_bg, tau_px=float(tau_px),
                    contour_bgr=(0, 255, 0), dilate_px=1, contour_thickness=2
                )
                _imwrite_bgr(os.path.join(out_err, f"{stem}.png"), err_bgr)
                
                # 6) 에러맵 region별 마스킹
                order = ["thick:thickQ", "thick:thinQ", "texless", "edge"]
                for region in order:
                    region_err_bgr = region_viz.render_error_by_region_on_white(err_bgr, masks, order=[region])
                    _imwrite_bgr(os.path.join(out_region_err, f"{stem}_{region}.png"), region_err_bgr)
                
        if logger:
            logger.info(f"[BAKE] {name}: left/disp/error/region 완료")