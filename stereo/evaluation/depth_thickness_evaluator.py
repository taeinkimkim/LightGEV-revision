import torch
import numpy as np
from stereo.evaluation.region_binner import RegionBinner, BinningConfig
from stereo.evaluation.region_visualizer import RegionVisualizer, VisConfig
from stereo.evaluation.metric_per_image import epe_metric, d1_metric, threshold_metric


class DepthThicknessEvaluator:
    def __init__(self, bin_cfg=None, vis_cfg=None, device="cuda"):
        self.binner = RegionBinner(bin_cfg or BinningConfig())
        self.viz = RegionVisualizer(vis_cfg or VisConfig())
        self.device = device

    def compute_metrics(self, disp_pred, disp_gt, masks):
        """
        disp_pred, disp_gt: numpy, shape (H,W) or (1,H,W)
        masks: dict[name -> np.ndarray (H,W) or (1,H,W), bool/0-1]
        """
        if disp_gt is None or disp_gt.size == 0:
            return {}

        res = {}

        pred_t = torch.from_numpy(disp_pred).float().to(self.device)
        gt_t   = torch.from_numpy(disp_gt).float().to(self.device)

        # (H,W) -> (1,H,W) 로 승격 (배치=1 가정)
        if pred_t.ndim == 2:
            pred_t = pred_t.unsqueeze(0)
        if gt_t.ndim == 2:
            gt_t = gt_t.unsqueeze(0)

        def _to_scalar(x):
            # metric이 맵 또는 배치 텐서를 반환해도 안전하게 스칼라화
            if isinstance(x, torch.Tensor):
                return x.mean().item()
            # numpy라면
            if hasattr(x, "mean"):
                return float(x.mean())
            return float(x)

        for name, m in masks.items():
            if m is None:
                continue
            m_np = np.asarray(m)
            if m_np.size == 0 or not np.any(m_np):
                continue

            mt = torch.from_numpy(m_np.astype(bool)).to(self.device)
            if mt.ndim == 2:
                mt = mt.unsqueeze(0)  # (H,W) -> (1,H,W)

            # 유효 픽셀 수
            npix = int(mt.sum().item())
            if npix == 0:
                continue

            e = _to_scalar(epe_metric(pred_t, gt_t, mt))
            d = _to_scalar(d1_metric(pred_t, gt_t, mt))
            t1 = _to_scalar(threshold_metric(pred_t, gt_t, mt, threshold=1.0))
            t3 = _to_scalar(threshold_metric(pred_t, gt_t, mt, threshold=3.0))

            res[name] = dict(
                npix=npix,
                epe=e,
                d1=d,
                t1=t1,
                t3=t3,
            )

        return res

    def _to_hw(self, arr):
        """disp_gt를 (H,W)로 보정하기 위한 유틸."""
        a = np.asarray(arr)
        a = np.squeeze(a)
        if a.ndim == 2:
            return a
        if a.ndim == 3 and a.shape[0] == 1:
            return a[0]
        # 마지막 시도: 가장 뒷 두 축을 (H,W)로 쓴다
        if a.ndim >= 2:
            return a.reshape(a.shape[-2], a.shape[-1])
        raise ValueError(f"Unexpected disp_gt shape: {arr.shape}")

    def evaluate(self, left_bgr, disp_pred, disp_gt):
        """
        left_bgr: (H,W,3) BGR
        disp_pred: (H,W) or (1,H,W) numpy disparity prediction
        disp_gt: (H,W) or (1,H,W) numpy disparity GT
        """
        if disp_gt is None or np.size(disp_gt) == 0:
            return {}, {}

        # disp_gt를 (H,W)로 정규화
        disp_gt_hw = self._to_hw(disp_gt)

        # global 유효 픽셀 마스크 (전체 지표용)
        valid_global = np.isfinite(disp_gt_hw) & (disp_gt_hw > 0)

        # Region-wise 마스크 생성 (edge, texless, thick:thinQ 등)
        masks = self.binner.build_masks(left_bgr, disp_gt_hw)

        # 모든 region 마스크에 valid_global 적용
        for k, m in list(masks.items()):
            m = np.asarray(m).astype(bool)
            masks[k] = (m & valid_global).astype(np.uint8)

        # union 기반 global (선택사항)
        union = np.zeros_like(valid_global, bool)
        for m in masks.values():
            union |= m.astype(bool)
        masks["global"] = union.astype(np.uint8)

        results = self.compute_metrics(disp_pred, disp_gt_hw, masks)
        return results, masks
