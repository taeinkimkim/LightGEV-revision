import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from stereo.vis.overlay import ErrorOverlay

class VisConfig:
    """
    Minimal visualization config dataclass used by PanelCompareEngine.
    """
    def __init__(
        self,
        vmax_mode="global",
        vmin_mode="global",
        clip_mode="hard",
        tau_mode="quantile",
        tau_scope="global",
        tau_q=0.9,
    ):
        self.vmax_mode = vmax_mode
        self.vmin_mode = vmin_mode
        self.clip_mode = clip_mode
        self.tau_mode = tau_mode
        self.tau_scope = tau_scope
        self.tau_q = tau_q

class RegionVisualizer:
    def __init__(self, cfg=None):
        self.cfg = cfg or VisConfig()
        self.overlay = ErrorOverlay()
        self.default_palette = {
            "depth:near":  (0, 0, 255),     # red
            "depth:mid":   (0, 165, 255),   # orange
            "depth:far":   (0, 255, 255),   # yellow
            "thick:thinQ": (255, 0, 0),     # blue
            "thick:thickQ":(255, 0, 255),   # magenta
            "texless":     (0, 255, 0),     # green
            "boundary":    (200, 200, 200), # gray
        }
        self.order = [
                "depth:far", "depth:mid", "depth:near",
                "thick:thickQ", "thick:thinQ",
                "texless", "boundary",
        ]

    def overlay_regions(
        self,
        left_bgr: np.ndarray,
        masks: Dict[str, np.ndarray[bool]],
        *,
        order: Optional[List[str]] = None,
        palette: Optional[Dict[str, Tuple[int, int, int]]] = None,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Region mask들을 서로 다른 색으로 칠한 반투명 오버레이 이미지를 생성한다.

        Args
            left_bgr: (H,W,3) 원본 좌측 컬러
            masks:    {"depth:near": bool(H,W), "thick:thinQ": bool(H,W), ...}
            order:    우선순위(뒤에 오는 것이 앞의 것을 덮어씀)
            palette:  키별 BGR 컬러
            alpha:    오버레이 투명도 (0~1)
            draw_legend: 오른쪽 상단에 범례 표시

        Returns
            (H,W,3) BGR uint8
        """
        assert left_bgr.ndim == 3 and left_bgr.shape[2] == 3
        H, W = left_bgr.shape[:2]
        canvas = left_bgr.copy().astype(np.uint8)

        # 기본 표시 순서 (depth 우선)
        if order is None:
            order = self.order

        if palette:
            self.default_palette.update(palette)

        # 누적 오버레이
        overlay = np.zeros_like(canvas, dtype=np.uint8)
        hit = np.zeros((H, W), dtype=bool)  # 이 픽셀이 이미 칠해졌는지 (우선순위 제어용)

        used_labels = []
        for key in order:
            m = masks.get(key, None)
            if m is None:
                continue
            mk = m.astype(bool)
            target = mk & (~hit)
            if not target.any():
                continue
            color = self.default_palette.get(key, (128, 128, 128))
            overlay[target] = color
            hit[target] = True
            used_labels.append((key, color))

        # 블렌딩
        blend = canvas.copy()
        mask_any = hit
        if mask_any.any():
            blend[mask_any] = (alpha * overlay[mask_any] + (1 - alpha) * canvas[mask_any]).astype(np.uint8)

        return np.ascontiguousarray(blend)
    
    def render_error_by_region_on_white(
        self,
        err_bgr: np.ndarray,                      
        masks: Dict[str, np.ndarray],            
        *,
        order: Optional[List[str]] = None,       
    ) -> np.ndarray:
        """
        에러 맵을 흰색 배경 위에 영역 마스크로 '오려 붙이기' 해서 보여준다.
        - 왼쪽 원본 이미지는 사용하지 않음.
        - 색상화는 베이킹 때 사용한 함수(error_colorize_fn)를 주입하면 1:1 일치함.
          미제공 시 OpenCV TURBO로 폴백.
        - mask 우선순위(order) 뒤쪽 항목이 앞쪽을 덮어쓴다(= 위에 올라와 보임).
        - tau_px가 주어지면 영역에 관계없이 전체 에러맵에 대한 τ-컨투어(노란색 기본)를 추가로 그린다.
        """

        H, W = err_bgr.shape[:2]

        # 빈 흰색 캔버스에 마스크로 '오려 붙이기'
        white = np.full((H, W, 3), 255, np.uint8)
        out = white.copy()

        # 표시 우선순위
        if order is None:
            order = self.order

        used_labels = []
        hit = np.zeros((H, W), dtype=bool)  # 이미 채워진 픽셀(우선순위 제어)

        for key in order:
            m = masks.get(key, None)
            if m is None:
                continue
            mk = m.astype(bool)
            target = mk & (~hit)
            if not target.any():
                continue
            out[target] = err_bgr[target]  # ← 에러 색상 그대로 채움
            hit[target] = True
            used_labels.append((key, self.default_palette.get(key, (128,128,128))))

        return np.ascontiguousarray(out)