import cv2
import numpy as np
import torch

def _tensor_to_bgr(tensor: torch.Tensor):
    img = tensor.detach().cpu().numpy()
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def _tensor_to_numpy_chw(x: torch.Tensor) -> np.ndarray:
    """
    텐서를 2D (H,W) 스칼라 맵으로 변환.
    어떤 네트워크든 disp_pred는 1채널이어야 하므로
    3채널 이상이면 첫 채널만 강제 선택.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    else:
        x = np.asarray(x, dtype=np.float32)

    # squeeze batch dim
    while x.ndim > 2 and x.shape[0] == 1:
        x = x[0]

    # shape now could be:
    # (1,H,W), (C,H,W), (H,W), (H,W,1), (H,W,C)...

    # If CHW
    if x.ndim == 3 and x.shape[0] != 1:      # C,H,W
        x = x[0]                             # take first channel

    # If HWC
    if x.ndim == 3 and x.shape[-1] != 1:     # H,W,C
        x = x[..., 0]

    # If still H,W,1
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    # final guarantee: 2D
    if x.ndim != 2:
        raise ValueError(f"disp must be 2D after conversion, got {x.shape}")

    return x.astype(np.float32)