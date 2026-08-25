from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple, Dict, Optional, Callable

from stereo.vis.disp_color import disp_to_color_bgr

# 타입: 범례 바 생성 콜백 (W,H) -> np.ndarray(BGR HxWx3)
BarFn = Callable[[int, int], np.ndarray]

class PanelBuilder:
    def __init__(
        self,
        background: str = "black",
        pad: int = 30,             # 타일 간격 # KITTI: 20 # Tartanground: 30
        tile_w: int = 700,         # 타일 가로 # KITTI: 724 # Tartanground: 700
        tile_h: int = 700,         # 타일 세로(이미지+캡션 포함) # KITTI: 256 # Tartanground: 700
        sep_thickness: int = 2,    # 그룹 구분선 두께
        caption_fg: Tuple[int,int,int] = (235,235,235),
        legend_h: int = 180,       # 범례 높이만 키워 가독성↑ (이미지 크기 불변)
        tau_color_bgr: Tuple[int,int,int] = (0,255,255),  # τ 컨투어 색
        region_legend: Optional[Dict[str, Tuple[int,int,int]]] = None,  # Region 색상표(BGR)
        disp_bar_fn: Optional[BarFn] = None,   # Disp 범례 바 생성기 (기본: disp_to_color)
        error_bar_fn: Optional[BarFn] = None,  # Error 범례 바 생성기 (기본: TURBO, 필요시 교체)
    ):
        self.bg = self._bg_color(background)
        self.fg = (255,255,255) if sum(self.bg) < 200 else (0,0,0)
        self.pad = int(pad)
        self.W = int(tile_w)
        self.H = int(tile_h)
        self.sep_th = int(sep_thickness)
        self.caption_fg = caption_fg
        self.legend_h = int(legend_h)
        self.tau_color = tau_color_bgr

        # 기본 Region 팔레트(임시)
        self.region_legend_default = {
            "thick:thinQ": (255, 0, 0),     # blue
            "thick:thickQ":(255, 0, 255),   # magenta
            "texless":     (0, 255, 0),     # green
            "boundary":    (200, 200, 200), # gray
        }
        self.region_legend = region_legend or self.region_legend_default

        # 범례 바 생성기 설정
        self._disp_bar_fn = disp_bar_fn or self._legend_bar_from_disp_to_color
        self._error_bar_fn = error_bar_fn or self._legend_bar_error_default

    # -------------------- utils --------------------
    @staticmethod
    def _bg_color(name: str) -> Tuple[int,int,int]:
        return {"black":(0,0,0), "white":(255,255,255), "gray":(120,120,120)}.get(name,(0,0,0))

    def _auto_font(self, base_h: int) -> Tuple[float,int,int]:
        """
        캡션 밴드 높이 포함해서 폰트 스케일/두께 산정.
        Returns: (font_scale, thickness, bar_h)
        """
        fs = max(0.55, min(1.25, base_h / 420.0))   # 캡션/범례 가독성 강화
        th = max(1, int(round(fs + 0.6)))
        bar_h = max(18, int(round(fs * 16)) + 6)
        bar_h = min(bar_h, max(22, base_h // 3))    # 캡션 밴드 과대 방지
        return fs, th, bar_h

    def _fit_rect(self, img: np.ndarray, W: int, H: int) -> np.ndarray:
        """비율 유지, 레터/필라박스로 W×H 채우기."""
        if img is None:
            return np.full((H, W, 3), self.bg, np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return np.full((H, W, 3), self.bg, np.uint8)
        scale = min(W / float(w), H / float(h))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(img, (nw, nh), interpolation=interp)
        canvas = np.full((H, W, 3), self.bg, np.uint8)
        y0 = (H - nh) // 2
        x0 = (W - nw) // 2
        canvas[y0:y0+nh, x0:x0+nw] = resized
        return canvas

    def _caption_band(self, W: int, H: int, text: str) -> np.ndarray:
        """이미지 아래에 붙는 캡션 밴드 (배경 위에 텍스트만)."""
        band = np.full((H, W, 3), self.bg, np.uint8)
        if not text:
            return band
        fs, th, _ = self._auto_font(self.H)  # 캡션은 타일 높이 기준
        (tw, th_text), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        tx = max(6, (W - tw) // 2)
        ty = max(th + 4, (H + th_text) // 2)
        cv2.putText(band, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, self.caption_fg, th, cv2.LINE_AA)
        return band

    def _tile(self, img: np.ndarray, caption: str, W: int, H: int) -> np.ndarray:
        """타일: [이미지(H - bar_h)] + [캡션(bar_h)]"""
        fs, th, bar_h = self._auto_font(H)
        img_h = max(1, H - bar_h)
        img_canvas = self._fit_rect(img, W, img_h)
        cap_band = self._caption_band(W, bar_h, caption)
        return np.vstack([img_canvas, cap_band])

    # -------------------- legend bars --------------------
    def _legend_bar_from_disp_to_color(self, W: int, H: int) -> np.ndarray:
        """
        disp_to_color 룩업 그대로 사용한 수평 그라디언트 바.
        0..1 범위를 max_disp=1.0으로 전달해 컬러 생성 → BGR 변환.
        """
        grad = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :].repeat(H, axis=0)
        bgr = disp_to_color_bgr(grad, max_disp=1.0)  # (H,W,3), RGB
        return bgr

    def _legend_bar_error_default(self, W: int, H: int) -> np.ndarray:
        """
        기본 에러 바: OpenCV TURBO (베이킹이 TURBO를 썼다면 그대로 일치).
        만약 프로젝트에서 다른 LUT를 썼다면, PanelBuilder 생성 시 error_bar_fn으로 덮어쓰기.
        """
        grad = np.linspace(0, 255, W, dtype=np.uint8)[None, :].repeat(H, axis=0)
        return cv2.applyColorMap(grad, cv2.COLORMAP_INFERNO)

    # -------------------- legend/footer --------------------
    def _legend_block(self, title: str, left_lab: str, right_lab: str, W: int, H: int, bar_fn: BarFn) -> np.ndarray:
        """제목 + 색상 바 + 좌/우 라벨을 포함하는 블록 (주입된 bar_fn 사용)"""
        block = np.full((H, W, 3), self.bg, np.uint8)
        pad = 10

        # 범례는 legend 높이 기준 폰트
        fs, th, _ = self._auto_font(self.legend_h)
        title_h = max(22, int(self.legend_h * 0.26))
        bar_h = max(22, int(self.legend_h * 0.32))
        text_h = H - title_h - bar_h - pad*3
        text_h = max(18, text_h)

        # 제목
        (tw, th_text), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        tx = max(pad, (W - tw) // 2)
        ty = pad + th_text + 2
        cv2.putText(block, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, self.caption_fg, th, cv2.LINE_AA)

        # 색상 바 (주입된 함수로 생성)
        bar = bar_fn(W - pad*2, bar_h)
        yb = ty + pad
        block[yb:yb+bar.shape[0], pad:pad+bar.shape[1]] = bar

        # 좌/우 라벨
        fs2 = max(0.6, fs * 0.95)
        th2 = max(1, th)
        (lw, lh), _ = cv2.getTextSize(left_lab,  cv2.FONT_HERSHEY_SIMPLEX, fs2, th2)
        (rw, rh), _ = cv2.getTextSize(right_lab, cv2.FONT_HERSHEY_SIMPLEX, fs2, th2)
        ly = yb + bar.shape[0] + pad + lh
        cv2.putText(block, left_lab,  (pad, ly), cv2.FONT_HERSHEY_SIMPLEX, fs2, self.caption_fg, th2, cv2.LINE_AA)
        cv2.putText(block, right_lab, (W - pad - rw, ly), cv2.FONT_HERSHEY_SIMPLEX, fs2, self.caption_fg, th2, cv2.LINE_AA)
        return block

    def _region_legend_block(self, region_legend: Dict[str, Tuple[int,int,int]], W: int, H: int) -> np.ndarray:
        """
        Region 팔레트 범례: 스와치 + 라벨을 여러 줄로 출력.
        region_legend: {"RegionName": (B,G,R), ...}
        """
        block = np.full((H, W, 3), self.bg, np.uint8)
        pad = 10
        fs, th, _ = self._auto_font(self.legend_h)

        # 제목
        title = "Region overlay legend"
        (tw, th_text), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        tx = max(pad, (W - tw) // 2)
        ty = pad + th_text + 2
        cv2.putText(block, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, self.caption_fg, th, cv2.LINE_AA)

        # 항목들 (3열 그리드)
        y = ty + 10
        row_h = max(24, int(self.legend_h * 0.22))
        sw_h, sw_w = 18, 46
        col_w = (W - pad*3) // 3
        i = 0
        items = list(region_legend.items())
        for name, bgr in items:
            col = i % 3
            row = i // 3
            x0 = pad + col * col_w
            y0 = y + row * row_h

            # 스와치 박스
            cv2.rectangle(block, (x0, y0), (x0 + sw_w, y0 + sw_h), bgr, thickness=-1)
            cv2.rectangle(block, (x0, y0), (x0 + sw_w, y0 + sw_h), (255,255,255), thickness=1)

            # 라벨
            fs2 = max(0.6, fs * 0.95)
            th2 = max(1, th)
            cv2.putText(block, str(name), (x0 + sw_w + 10, y0 + sw_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, fs2, self.caption_fg, th2, cv2.LINE_AA)
            i += 1
            
        return block

    def _legend_footer(self, page_w: int, region_legend: Optional[Dict[str, Tuple[int,int,int]]] = None) -> np.ndarray:
        """
        푸터 범례 (3열):
        [ Disparity] | [ Error + τ ] | [ Region legend ]
        - Disp 바: disp_to_color
        - Error 바: error_bar_fn (베이킹과 동일 룩업을 외부에서 주입)
        """
        H = self.legend_h
        P = self.pad
        # 3등분 (좌/중/우)
        w_each = (page_w - 4*P) // 3
        left_w  = w_each
        mid_w   = w_each
        right_w = page_w - 4*P - left_w - mid_w

        # Disparity (disp_to_color)
        disp_title = "Disparity color scale"
        disp_left  = "smaller"
        disp_right = "larger"
        left_block = self._legend_block(disp_title, disp_left, disp_right, left_w, H - 2*P, self._disp_bar_fn)

        # Error + τ (베이킹과 동일 룩업)
        err_title = "Error color scale"
        err_left  = "lower"
        err_right = "higher"
        mid_block = self._legend_block(err_title, err_left, err_right, mid_w, H - 2*P, self._error_bar_fn)

        # Region legend
        legend_map = region_legend or self.region_legend
        right_block = self._region_legend_block(legend_map, right_w, H - 2*P)

        # 좌/중/우 배치
        footer = np.full((H, page_w, 3), self.bg, np.uint8)
        x = P
        footer[P:P+left_block.shape[0], x:x+left_block.shape[1]] = left_block
        x += left_block.shape[1] + P
        footer[P:P+mid_block.shape[0],  x:x+mid_block.shape[1]]  = mid_block
        x += mid_block.shape[1] + P
        footer[P:P+right_block.shape[0], x:x+right_block.shape[1]] = right_block

        # 상단에 가는 구분선
        cv2.line(footer, (P, P-2), (page_w - P, P-2), (255,255,255), 1, cv2.LINE_AA)
        return footer

    # -------------------- main collage (two models, rectangular tiles) --------------------
    def build_two_model_collage_rect(
        self,
        left_bgr: np.ndarray,
        model1_tiles: Dict[str, np.ndarray],  # keys: "Disp","Region","Error","Left+Contour"
        model2_tiles: Dict[str, np.ndarray],
        *,
        captions: Dict[str, str] = None,
        show_group_separators: bool = True,
        region_legend: Optional[Dict[str, Tuple[int,int,int]]] = None,
    ) -> np.ndarray:
        """
        하나의 샘플을 두 모델과 함께 직사각형 타일로 비교.
        - 같은 열(같은 맵 타입)의 행1/행2 사이에는 구분선 없음
        - 열 경계와 상/하 블록 경계에만 그룹 구분선
        """
        W, H, P = self.W, self.H, self.pad
        captions = captions or {
            "Left": "Left",
            "M1_Disp": "Model 1 Disp",
            "M2_Disp": "Model 2 Disp",
            "M1_Err": "Model 1 Error",
            "M1_Reg": "Model 1 Region",
            "M2_Err": "Model 2 Error",
            "M2_Reg": "Model 2 Region",
        }

        # --- Top block (LEFT spans two rows) ---
        left_span_h = 2 * H + P
        left_span  = self._tile(left_bgr, captions["Left"], W, left_span_h)

        m1_disp   = self._tile(model1_tiles["Disp"],   captions["M1_Disp"],  W, H)
        m2_disp   = self._tile(model2_tiles["Disp"],   captions["M2_Disp"],  W, H)

        # 캔버스 크기 산정
        top_w = (3 * W + 4 * P)
        top_h = (2 * H + 3 * P)
        top = np.full((top_h, top_w, 3), self.bg, np.uint8)

        # 좌표계
        x1 = P
        x2 = x1 + W + P
        x3 = x2 + W + P
        y1 = P
        y2 = y1 + H + P

        # place Left span
        top[y1:y1+left_span_h, x1:x1+W] = left_span
        # row1 right cols
        top[y1:y1+H, x2:x2+W] = m1_disp
        # row2 right cols
        top[y2:y2+H, x2:x2+W] = m2_disp

        # --- Bottom block (two rows x three cols) ---
        m1_err = self._tile(model1_tiles["Error"], captions["M1_Err"], W, H)
        m1_reg = self._tile(model1_tiles["Region"], captions["M1_Reg"], W, H)

        m2_err = self._tile(model2_tiles["Error"], captions["M2_Err"], W, H)
        m2_reg = self._tile(model2_tiles["Region"], captions["M2_Reg"], W, H)

        bot_w = top_w
        bot_h = top_h
        bot = np.full((bot_h, bot_w, 3), self.bg, np.uint8)

        # row3 (model1)
        y3 = P
        bot[y3:y3+H, x2:x2+W] = m1_err
        bot[y3:y3+H, x3:x3+W] = m1_reg

        # row4 (model2)
        y4 = y3 + H + P
        bot[y4:y4+H, x2:x2+W] = m2_err
        bot[y4:y4+H, x3:x3+W] = m2_reg

        # --- Compose page body ---
        if show_group_separators:
            sep_h = self.sep_th
            hsep = np.full((sep_h, top_w, 3), (255,255,255), np.uint8)
            body = np.vstack([top, hsep, bot])
        else:
            body = np.vstack([top, bot])

        # 세로 그룹 구분선(열 경계): Left | Disp
        if show_group_separators:
            H_total, W_total = body.shape[:2]
            col_boundaries = [x2 - self.pad//2, x3 - self.pad//2]
            for xb in col_boundaries:
                cv2.line(body, (xb, self.pad), (xb, H_total - self.pad), (255,255,255), self.sep_th, cv2.LINE_AA)

        # 외곽선
        cv2.rectangle(body, (self.pad//2, self.pad//2),
                      (body.shape[1]-self.pad//2-1, body.shape[0]-self.pad//2-1),
                      (255,255,255), 1, cv2.LINE_AA)

        # --- Footer legend (3열, region legend 포함) ---
        footer = self._legend_footer(page_w=body.shape[1], region_legend=region_legend)

        # 상/하 합치기
        page = np.vstack([body, footer])
        return page
    
    def build_two_model_region_error_collage(
        self,
        left_bgr: np.ndarray,
        *,
        m1_disp: np.ndarray,
        m2_disp: np.ndarray,
        # others (edge / thick / texless)
        m1_err_edge:       np.ndarray,
        m1_err_thinQ:      np.ndarray,
        m1_err_thickQ:     np.ndarray,
        m1_err_texless:    np.ndarray,
        m2_err_edge:       np.ndarray,
        m2_err_thinQ:      np.ndarray,
        m2_err_thickQ:     np.ndarray,
        m2_err_texless:    np.ndarray,
        captions: Optional[Dict[str, str]] = None,
        show_group_separators: bool = True,
        region_legend: Optional[Dict[str, Tuple[int,int,int]]] = None,
    ) -> np.ndarray:
        """
        Row1: [ Left                | M1 Disp               | M1 err edge ]
        Row2: [ (blank)             | M2 Disp               | M2 err edge ]
        -------------------------------------------------------------------
        Row3: [ M1 err thick:thinQ  | M1 err thick:thickQ   | M1 err texless ]
        Row4: [ M2 err thick:thinQ  | M2 err thick:thickQ   | M2 err texless ]
        -------------------------------------------------------------------
        Footer: [ Disp color bar | Error color bar | Region legend ]
        """
        W, H, P = self.W, self.H, self.pad

        captions = captions or {
            "Left": "Left",
            # Row 1
            "M1_Disp": "M1 Disp",
            "M1_Edge": "M1 Error (edge)",
            # Row 2
            "M2_Disp": "M2 Disp",
            "M2_Edge": "M2 Error (edge)",
            # Row 3
            "M1_ThinQ":  "M1 Error (thick:thinQ)",
            "M1_ThickQ": "M1 Error (thick:thickQ)",
            "M1_Tex":    "M1 Error (texless)",
            # Row 4
            "M2_ThinQ":  "M2 Error (thick:thinQ)",
            "M2_ThickQ": "M2 Error (thick:thickQ)",
            "M2_Tex":    "M2 Error (texless)",
        }

        # ---------- 상단 블록 (2행 x 3열) ----------
        # Row1: [ Left | M1 Disp | M1 err edge ]
        # Row2: [ blank | M2 Disp | M2 err edge ]
        left_tile = self._tile(left_bgr, captions["Left"], W, H)
        blank     = np.full((H, W, 3), self.bg, np.uint8)

        r1_c2 = self._tile(m1_disp,      captions["M1_Disp"], W, H)
        r1_c3 = self._tile(m1_err_edge,  captions["M1_Edge"], W, H)

        r2_c2 = self._tile(m2_disp,      captions["M2_Disp"], W, H)
        r2_c3 = self._tile(m2_err_edge,  captions["M2_Edge"], W, H)

        top_w = 3 * W + 4 * P
        top_h = 2 * H + 3 * P
        top = np.full((top_h, top_w, 3), self.bg, np.uint8)

        x1 = P
        x2 = x1 + W + P
        x3 = x2 + W + P

        y1 = P
        y2 = y1 + H + P

        # 배치 (Top block)
        # Row1
        top[y1:y1+H, x1:x1+W] = left_tile
        top[y1:y1+H, x2:x2+W] = r1_c2
        top[y1:y1+H, x3:x3+W] = r1_c3
        # Row2
        top[y2:y2+H, x1:x1+W] = blank
        top[y2:y2+H, x2:x2+W] = r2_c2
        top[y2:y2+H, x3:x3+W] = r2_c3

        # ---------- 하단 블록 (2행 x 3열) ----------
        # Row3: [ M1 err thick:thinQ | M1 err thick:thickQ | M1 err texless ]
        # Row4: [ M2 err thick:thinQ | M2 err thick:thickQ | M2 err texless ]
        r3_c1 = self._tile(m1_err_thinQ,  captions["M1_ThinQ"],  W, H)
        r3_c2 = self._tile(m1_err_thickQ, captions["M1_ThickQ"], W, H)
        r3_c3 = self._tile(m1_err_texless,captions["M1_Tex"],    W, H)

        r4_c1 = self._tile(m2_err_thinQ,  captions["M2_ThinQ"],  W, H)
        r4_c2 = self._tile(m2_err_thickQ, captions["M2_ThickQ"], W, H)
        r4_c3 = self._tile(m2_err_texless,captions["M2_Tex"],    W, H)

        bot_w = top_w
        bot_h = 2 * H + 3 * P
        bot = np.full((bot_h, bot_w, 3), self.bg, np.uint8)

        y3 = P
        y4 = y3 + H + P

        # Row3
        bot[y3:y3+H, x1:x1+W] = r3_c1
        bot[y3:y3+H, x2:x2+W] = r3_c2
        bot[y3:y3+H, x3:x3+W] = r3_c3
        # Row4
        bot[y4:y4+H, x1:x1+W] = r4_c1
        bot[y4:y4+H, x2:x2+W] = r4_c2
        bot[y4:y4+H, x3:x3+W] = r4_c3

        # ---------- 본문 합치기 + 구분선 ----------
        if show_group_separators:
            sep_h = self.sep_th
            hsep = np.full((sep_h, top_w, 3), (255,255,255), np.uint8)
            body = np.vstack([top, hsep, bot])
        else:
            body = np.vstack([top, bot])

        if show_group_separators:
            H_total = body.shape[0]
            # 세로 구분선: 열 경계 (x2, x3의 왼쪽 패딩 경계 근처)
            col_boundaries = [x2 - self.pad//2, x3 - self.pad//2]
            for xb in col_boundaries:
                cv2.line(body, (xb, self.pad), (xb, H_total - self.pad), (255,255,255), self.sep_th, cv2.LINE_AA)

        # 외곽선
        cv2.rectangle(body, (self.pad//2, self.pad//2),
                      (body.shape[1]-self.pad//2-1, body.shape[0]-self.pad//2-1),
                      (255,255,255), 1, cv2.LINE_AA)

        # ---------- Footer legend ----------
        footer = self._legend_footer(page_w=body.shape[1], region_legend=region_legend)
        page = np.vstack([body, footer])
        return page