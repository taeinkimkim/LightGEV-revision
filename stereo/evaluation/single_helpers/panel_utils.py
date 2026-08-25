from typing import Optional, List
import os
import cv2
import numpy as np
from stereo.vis.disp_color import disp_to_color_bgr

def _colorize_scalar_to_bgr(arr: np.ndarray, vmax: float) -> np.ndarray:
    bgr = disp_to_color_bgr(arr, vmax)
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    return bgr

def _hstack_same_height(images: List[np.ndarray], pad: int = 8, pad_color=(20,20,20)) -> Optional[np.ndarray]:
    imgs = [im for im in images if im is not None]
    if not imgs:
        return None
    H = max(im.shape[0] for im in imgs)
    resized = []
    for im in imgs:
        if im.shape[0] != H:
            scale = H / im.shape[0]
            newW = int(round(im.shape[1] * scale))
            im = cv2.resize(im, (newW, H), interpolation=cv2.INTER_AREA)
        resized.append(im)
    if pad > 0 and len(resized) > 1:
        pads = [np.full((H, pad, 3), pad_color, dtype=np.uint8) for _ in range(len(resized)-1)]
        out = []
        for i, im in enumerate(resized):
            out.append(im)
            if i < len(resized)-1:
                out.append(pads[i])
        return np.concatenate(out, axis=1)
    return np.concatenate(resized, axis=1)

def _compose_panel_triptych(left_bgr, disp_bgr, depth_bgr=None) -> Optional[np.ndarray]:
    return _hstack_same_height([left_bgr, disp_bgr, depth_bgr], pad=8)

def _build_panel(panel_builder, panel_dir: str, stem: str, left_bgr, disp_bgr, depth_bgr=None) -> None:
    if not panel_dir:
        return

    panel_img = panel_builder.build_panel(left_bgr, disp_bgr, depth_bgr)
    cv2.imwrite(os.path.join(panel_dir, f"{stem}.png"), panel_img)
        
    # fallback
    panel_img = _compose_panel_triptych(left_bgr, disp_bgr, depth_bgr)
    if panel_img is not None:
        cv2.imwrite(os.path.join(panel_dir, f"{stem}.png"), panel_img)