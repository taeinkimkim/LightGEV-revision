import cv2
import numpy as np

class ErrorOverlay:
    """
    Generate overlay visualization for disparity error maps.
    - Uses GLOBAL (vmin, vmax) for fair normalization across models/images
    - Draws >tau contours over the colormapped error
    """
    def __init__(self, colormap=cv2.COLORMAP_INFERNO):
        self.colormap = colormap

    @staticmethod
    def _normalize_fixed(err: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        # robust to NaNs
        arr = np.array(err, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=vmax, neginf=vmin)
        if vmax <= vmin:
            vmax = vmin + 1e-6
        norm = (arr - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0.0, 1.0)
        return (norm * 255.0).astype(np.uint8)

    def _reduce_to_hw(self, arr, H, W):
        a = np.asarray(arr)

        # 너무 고차원이면 먼저 singleton 축 제거
        if a.ndim > 3:
            a = np.squeeze(a)

        # 이미 (H,W)면 그대로
        if a.ndim == 2 and a.shape == (H, W):
            return a

        # 3차원 케이스들을 처리
        if a.ndim == 3:
            B, h, w = a.shape

            # (1,H,W) -> squeeze
            if B == 1 and (h, w) == (H, W):
                return a[0]

            # (H,W,1) -> squeeze
            if (B, h) == (H, W) and w == 1:
                return a[..., 0]

            # (N,H,W) -> N축 축약
            if (h, w) == (H, W):
                return np.median(a, axis=0)

            # (H,W,N) -> N축 축약
            if (B, h) == (H, W):
                return np.median(a, axis=2)

            # 전치로 해결 가능한 흔한 뒤바뀜: (W,H,N) or (N,W,H)
            if a.transpose(0, 2, 1).shape[1:] == (H, W):  # (N,W,H) -> (N,H,W)
                a2 = a.transpose(0, 2, 1)
                return np.median(a2, axis=0)
            if a.transpose(2, 0, 1).shape[1:] == (H, W):  # (W,H,N) -> (N,H,W)
                a2 = a.transpose(2, 0, 1)
                return np.median(a2, axis=0)

        # 2D인데 (H,W)와 다르면 전치/리사이즈 마지막 시도
        if a.ndim == 2:
            if a.shape[::-1] == (H, W):
                return a.T
            return cv2.resize(a.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)

        # 1D 등 기타 이상 케이스: 리사이즈로 강제 정형화
        a = np.squeeze(a)
        if a.ndim == 2:
            if a.shape[::-1] == (H, W):
                return a.T
            return cv2.resize(a.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        # 최후 수단
        return cv2.resize(a.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)

    def make_overlay(
        self,
        left_bgr: np.ndarray,
        err: np.ndarray,
        *,
        vmin: float,
        vmax: float,
        tau_px: float = 3.0,
        mask=None,
        contour_bgr=(0, 255, 0),
        contour_thickness: int = 2,
        dilate_px: int = 0,
    ) -> np.ndarray:
        """
        Returns:
            BGR uint8 image, contiguous.
        """
        H, W = left_bgr.shape[:2]
        err = self._reduce_to_hw(err, H, W)
        cmap_in = self._normalize_fixed(err, vmin, vmax)
        vis = cv2.applyColorMap(cmap_in, self.colormap)

        if mask is None:
            mask = np.ones(err.shape[:2], dtype=bool)
        else:
            mask = mask.astype(bool)

        # if tau_px is not None and tau_px > 0:
        #     tau_mask = (np.nan_to_num(err, nan=0.0) > float(tau_px)) & mask
        #     if dilate_px > 0:
        #         k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        #         tau_mask = cv2.dilate(tau_mask.astype(np.uint8), k).astype(bool)
        #     if np.any(tau_mask):
        #         cnts, _ = cv2.findContours(tau_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #         cv2.drawContours(vis, cnts, -1, contour_bgr, contour_thickness, cv2.LINE_AA)

        return np.ascontiguousarray(vis)