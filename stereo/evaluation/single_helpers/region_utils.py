from typing import Optional, Dict, Any
import os
import cv2
import numpy as np

def render_and_save(region_dir: str,
                    stem: str,
                    left_bgr,
                    masks: Dict[str, Any],
                    region_viz=None,
                    logger=None) -> Optional[str]:
    
    out_path = os.path.join(region_dir, f"{stem}.png")

    if region_viz is None:
        cv2.imwrite(out_path, left_bgr)
        return out_path

    try:
        blend_bgr = region_viz.overlay_regions(left_bgr, masks)
        if not isinstance(blend_bgr, np.ndarray):
            raise TypeError("overlay_regions returned non-array")
        if blend_bgr.dtype != np.uint8:
            blend_bgr = np.clip(blend_bgr, 0, 255).astype(np.uint8)
        cv2.imwrite(out_path, blend_bgr)
        return out_path
    
    except Exception as e:
        if logger: logger.warning(f"[{stem}] overlay_regions failed: {e}")