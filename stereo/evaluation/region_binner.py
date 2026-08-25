import numpy as np
import cv2
from stereo.evaluation.region_masks import RegionMasker


class BinningConfig:
    def __init__(self, near_m=15, mid_m=30, far_m=80, thin_q=0.9, thick_q=0.3):
        self.near_m = near_m
        self.mid_m = mid_m
        self.far_m = far_m
        self.thin_q = thin_q
        self.thick_q = thick_q

class RegionBinner:
    def __init__(self, cfg: BinningConfig):
        self.cfg = cfg
        self.masker = RegionMasker()
        
    def build_masks(self, left_bgr, disp_gt):
        H, W = left_bgr.shape[:2]

        # --- [추가] 어떤 GT든 (H,W)로 만드는 안전 축약기 ---
        def _reduce_to_hw(arr, H, W):
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

            # 1D 등 기타 이상 케이스: 리사이즈로 강제 정형화 (응급처치)
            a = np.squeeze(a)
            if a.ndim == 2:
                if a.shape[::-1] == (H, W):
                    return a.T
                return cv2.resize(a.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            # 최후 수단
            return cv2.resize(a.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)

        thick_masks = {}

        # ---- disp_gt 정규화: (H,W)로 축약 후 그래디언트/두께 마스크
        if disp_gt is not None:
            disp = _reduce_to_hw(disp_gt, H, W)

            valid = np.isfinite(disp) & (disp > 0)
            disp_safe = np.where(valid, disp, 0.0).astype(np.float32)

            # 주의: np.gradient는 (gy, gx) 순서
            gy, gx = np.gradient(disp_safe)
            grad = np.sqrt(gx**2 + gy**2)

            thin_thr  = np.quantile(grad, self.cfg.thin_q)
            thick_thr = np.quantile(grad, self.cfg.thick_q)
            thick_masks = {
                "thick:thinQ": grad > thin_thr,
                "thick:thickQ": grad < thick_thr
            }

        base_masks = {
            "texless": self.masker.textureless(left_bgr),   # (H,W)라고 가정
            # "boundary": self.masker.boundary_band_from_disp(disp),  # 필요 시
            "edge": self.masker.thin_from_edges(left_bgr),  # (H,W)라고 가정
        }

        return {**base_masks, **thick_masks}
