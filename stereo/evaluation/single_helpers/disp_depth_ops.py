from typing import Tuple, Optional
import cv2
import numpy as np

def _resize_disparity_with_scale_x(disp: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    H, W = out_hw
    Hd, Wd = disp.shape[-2], disp.shape[-1]
    if (Hd, Wd) == (H, W):
        return disp.astype(np.float32)
    disp_rs = cv2.resize(disp.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    scale_x = float(W) / float(Wd)
    disp_rs *= scale_x
    return disp_rs.astype(np.float32)

def _denorm_disparity(disp: np.ndarray, max_disp: float, pred_is_normalized: bool) -> np.ndarray:
    disp = disp.astype(np.float32)
    if pred_is_normalized:
        disp *= float(max_disp)
    return disp

def _sanitize_and_clip(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.clip(arr, vmin, vmax).astype(np.float32)

def _disp_to_depth(disp: np.ndarray, fpx: float, baseline_m: float, max_depth: float) -> Optional[np.ndarray]:
    if fpx <= 0.0 or baseline_m <= 0.0:
        return None
    eps = 1e-6
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = (float(fpx) * float(baseline_m)) / np.maximum(disp, eps)
    return _sanitize_and_clip(depth, 0.0, float(max_depth))