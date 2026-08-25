import numpy as np
import cv2

def disp_to_color_bgr(disp, max_disp=None, invalid_val=0.0, percentile=99.5, gamma: float = None):
    """disparity/depth → JET(BGR) 컬러맵. 반환은 항상 BGR!"""
    disp = np.asarray(disp, dtype=np.float32)

    # ---- disp를 반드시 2D로 맞추기 ----
    # disp shape 가능성: [H,W], [1,H,W], [H,W,1], [B,1,H,W], [H,W,3], ...
    if disp.ndim == 4:
        disp = disp[0, 0]  # [B,1,H,W] → [H,W]
    elif disp.ndim == 3:
        if disp.shape[0] == 1:        # [1,H,W]
            disp = disp[0]
        elif disp.shape[-1] == 1:     # [H,W,1]
            disp = disp[..., 0]
        else:
            # 예: [H,W,3] → 첫 채널만 사용
            disp = disp[..., 0]

    # 이제 disp는 항상 [H, W]
    H, W = disp.shape[:2]
    mask = np.isfinite(disp)

    if not np.any(mask):
        return np.zeros((H, W, 3), dtype=np.uint8)

    # 스케일 결정
    if max_disp is None or not np.isfinite(max_disp) or max_disp <= 0:
        max_disp = float(np.percentile(disp[mask], percentile))
        if max_disp <= 0:
            max_disp = float(disp[mask].max())
        if max_disp <= 0:
            max_disp = 1.0

    disp = np.clip(disp, 0, max_disp)
    disp[~mask] = invalid_val

    disp_u8 = (disp / max_disp * 255.0).astype(np.uint8)

    # 반드시 disp_u8 shape == (H, W)이어야 applyColorMap 성공
    return cv2.applyColorMap(disp_u8, cv2.COLORMAP_JET)
