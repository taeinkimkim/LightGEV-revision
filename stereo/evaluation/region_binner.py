import numpy as np
import cv2
from stereo.evaluation.region_masks import RegionMasker


# SceneFlow TRAIN 300장에서 구한 disparity gradient 분포 기준 고정 임계값.
# (유효 GT 4-이웃을 모두 가진 픽셀만 사용, 총 1.55억 픽셀 풀링)
#   q=0.30 -> 0.00899      q=0.90 -> 0.18411
#   참고 분위: q=0.50 0.02491, q=0.80 0.09738, q=0.95 0.47258, q=0.99 27.28094
# 이전에는 이미지마다 np.quantile 로 임계값을 새로 구했기 때문에, 정의상 모든
# 이미지에서 정확히 상위 10% / 하위 30% 가 각 bin 에 들어갔다. 즉 bin 이 이미지·
# 데이터셋 간에 같은 의미를 갖지 못했다. 고정값으로 바꿔 절대 기준을 쓴다.
SCENEFLOW_THIN_THR = 0.18411    # 이보다 크면 기울기가 가파름 (thin/경계)
SCENEFLOW_THICK_THR = 0.00899   # 이보다 작으면 평평함 (thick)


class BinningConfig:
    """thin_q / thick_q : 이미지별 quantile (유효 기울기 픽셀 기준).

    고정 임계값으로 되돌리려면 아래 주석 처리된 __init__ 과
    build_masks 안의 [백업] 블록을 함께 살리면 된다.
    """
    def __init__(self, near_m=15, mid_m=30, far_m=80, thin_q=0.9, thick_q=0.3):
        self.near_m = near_m
        self.mid_m = mid_m
        self.far_m = far_m
        self.thin_q = thin_q
        self.thick_q = thick_q

    # ---- [백업] 고정 임계값 방식 ----
    # def __init__(self, near_m=15, mid_m=30, far_m=80,
    #              thin_thr=SCENEFLOW_THIN_THR, thick_thr=SCENEFLOW_THICK_THR):
    #     self.near_m = near_m
    #     self.mid_m = mid_m
    #     self.far_m = far_m
    #     self.thin_thr = thin_thr
    #     self.thick_thr = thick_thr

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

            # 무효 GT 를 0 으로 채운 뒤 미분하면 실제로 존재하지 않는 기울기가 나온다.
            # 실측 (유효 GT 픽셀 중 오염 비율):
            #   SceneFlow 0.6%   ETH3D 15.9%   KITTI 85.3%
            # 특히 sparse GT(KITTI LiDAR, 유효율 22.7%)에서는 사실상 전부 가짜였다.
            # 기울기가 실제로 계산 가능한 픽셀만 인정하고, 나머지는 어느 bin 에도
            # 넣지 않는다("모름"이지 "평평함"이 아니다).
            #
            # np.gradient 는 축마다 스텐실이 다르다 (edge_order=1 기본):
            #   - 내부      : 중심차분  (i-1, i+1 필요)
            #   - 첫/마지막 : 단측차분  (인접 한 칸만 필요)
            # 그래서 가장자리를 통째로 버리지 않고 축별로 따로 판정한다.
            grad_valid = np.zeros_like(valid)
            if valid.shape[0] >= 2 and valid.shape[1] >= 2:
                vy = np.zeros_like(valid)                       # y축 기울기 계산 가능?
                vy[1:-1, :] = valid[:-2, :] & valid[2:, :]      # 중심차분
                vy[0, :] = valid[0, :] & valid[1, :]            # 단측차분 (첫 행)
                vy[-1, :] = valid[-2, :] & valid[-1, :]         # 단측차분 (마지막 행)

                vx = np.zeros_like(valid)                       # x축 기울기 계산 가능?
                vx[:, 1:-1] = valid[:, :-2] & valid[:, 2:]
                vx[:, 0] = valid[:, 0] & valid[:, 1]
                vx[:, -1] = valid[:, -2] & valid[:, -1]

                grad_valid = valid & vx & vy

            # ---- [백업] 고정 임계값 방식 (SceneFlow train 기준 절대 기준) ----
            # thick_masks = {
            #     "thick:thinQ": grad_valid & (grad > self.cfg.thin_thr),
            #     "thick:thickQ": grad_valid & (grad < self.cfg.thick_thr),
            # }
            # ------------------------------------------------------------------

            # 이미지별 quantile 임계값.
            # 중요: quantile 을 '유효 기울기 픽셀' 에서만 구한다.
            # 예전에는 grad 전체(0으로 채워진 무효 영역 포함)에서 구했는데,
            # KITTI 처럼 유효 기울기가 3.6% 뿐인 데이터셋에서는 분위수가 사실상
            # 0-fill 값으로 결정되어 임계값 자체가 무의미했다.
            if grad_valid.any():
                gvals = grad[grad_valid]
                thin_thr = np.quantile(gvals, self.cfg.thin_q)
                thick_thr = np.quantile(gvals, self.cfg.thick_q)
                thick_masks = {
                    "thick:thinQ": grad_valid & (grad > thin_thr),
                    "thick:thickQ": grad_valid & (grad < thick_thr),
                }
            else:
                z = np.zeros_like(valid)
                thick_masks = {"thick:thinQ": z, "thick:thickQ": z.copy()}

        base_masks = {
            "texless": self.masker.textureless(left_bgr),   # (H,W)라고 가정
            # "boundary": self.masker.boundary_band_from_disp(disp),  # 필요 시
            "edge": self.masker.thin_from_edges(left_bgr),  # (H,W)라고 가정
        }

        return {**base_masks, **thick_masks}
