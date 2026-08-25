import os
import cv2
import numpy as np
from stereo.vis.disp_color import disp_to_color_bgr

def _save_scalar_vis(dirpath: str, stem: str, arr: np.ndarray, vmax: float) -> None:
    if not dirpath:
        return
    
    bgr = disp_to_color_bgr(arr, vmax)
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(dirpath, f"{stem}.png"), bgr)

    return bgr

def _write_cache_npz(cache_io, cache_dir: str, cache_dict: dict, stem: str) -> None:
    path = os.path.join(cache_dir, f"{stem}.npz")
    cache_io.save_npz(path, cache_dict)