import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from stereo.panel.panel_builder import PanelBuilder

def _read(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img

class PanelCompareEngine:
    """
    PNG만 사용해서 비교 패널을 만든다.
    디렉토리 구조(모델별):
      <root>/<model>/
        left_bgr/*.png
        disp_vis_global/*.png
        error_vis/*.png
        region_overlay/*.png
    """
    def __init__(self, panel_builder: PanelBuilder, jpeg_quality: int = 95):
        self.panel_builder = panel_builder
        self.jpeg_quality = int(jpeg_quality)

    # ---------------- I/O ----------------
    def _discover_models(self, input_root: Path) -> Dict[str, Dict[str, Path]]:
        """
        반환: {model_name: {"left":..., "disp":..., "err":..., "region":...}}
        최소 요건: disp_vis_global 폴더가 있어야 비교에 포함.
        """
        models: Dict[str, Dict[str, Path]] = {}
        for d in sorted(input_root.iterdir(), key=lambda p: p.name):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            disp_dir = d / "disp_vis_global"
            if disp_dir.is_dir():
                models[d.name] = {
                    "left":   d / "left_bgr",
                    "disp":   d / "disp_vis_global",
                    "err":    d / "error_vis_global",
                    "region": d / "region_overlay",
                }
        if len(models) < 2:
            raise FileNotFoundError("Need at least two models having 'disp_vis_global' directory.")
        return models

    def _collect_common_stems(self, model_dirs: Dict[str, Dict[str, Path]]) -> List[str]:
        """
        disp_vis_global/*.png 파일명을 기준으로 공통 stem 교집합을 만든다.
        """
        sets: List[set] = []
        for m in model_dirs.values():
            disp_dir = m["disp"]
            stems = {p.stem for p in disp_dir.glob("*.png")}
            sets.append(stems)
        common = sorted(set.intersection(*sets)) if sets else []
        if not common:
            raise RuntimeError("No common stems across models (disp_vis_global/*.png).")
        return common

    # ---------------- tile builder ----------------
    def _assets_for(self, mdl_dirs: Dict[str, Path], stem: str) -> Dict[str, Optional[np.ndarray]]:
        return dict(
            left=_read(mdl_dirs["left"] / f"{stem}.png"),
            disp=_read(mdl_dirs["disp"] / f"{stem}.png"),
            err=_read(mdl_dirs["err"] / f"{stem}.png"),
            region=_read(mdl_dirs["region"] / f"{stem}.png"),
        )
    
    def _assets_for_region_errors(self, region_dir: Path, stem: str, key: str) -> Optional[np.ndarray]:
        """
        region_dir: .../region_error_vis_global
        key: "edge" | "thick:thinQ" | "thick:thickQ" | "texless"
        파일명 규칙: f"{stem}_{key}.png"
        """
        p = region_dir / f"{stem}_{key}.png"
        return _read(p)

    def _make_tiles_for_model(self, left: Optional[np.ndarray], disp: Optional[np.ndarray],
                              err: Optional[np.ndarray],
                              region: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
        
        def _nz(x: Optional[np.ndarray], fallback: Optional[np.ndarray] = None) -> np.ndarray:
            if x is not None:
                return x
            return fallback if fallback is not None else np.zeros((256, 256, 3), np.uint8)

        left_u8  = _nz(left)
        disp_u8  = _nz(disp, left_u8)
        err_u8   = _nz(err,  left_u8)
        region_u8= _nz(region, left_u8)

        return {
            "Disp": disp_u8,
            "Region": region_u8,
            "Error": err_u8,
        }

    # ---------------- run: each sample => one collage page ----------------
    def run(self, input_root: Path, output_dir: Path, *, models: Optional[List[str]] = None,
            stems: Optional[List[str]] = None, fname_prefix: str = "album_rect"):

        model_dirs = self._discover_models(input_root)

        # pick two models (or the first two by default)
        model_names = models if models else sorted(model_dirs.keys())
        if len(model_names) < 2:
            raise ValueError("Provide at least two model names, or ensure input_root has >=2 models.")
        m1, m2 = model_names[0], model_names[1]

        # stems: 교집합 자동 or 사용자 지정
        all_stems = stems if stems else self._collect_common_stems(model_dirs)

        for stem in all_stems:
            A = self._assets_for(model_dirs[m1], stem)
            B = self._assets_for(model_dirs[m2], stem)

            left_bgr = B["left"]
            tiles1 = self._make_tiles_for_model(A["left"], A["disp"], A["err"], A["region"])
            tiles2 = self._make_tiles_for_model(B["left"], B["disp"], B["err"], B["region"])

            page = self.panel_builder.build_two_model_collage_rect(
                left_bgr=left_bgr,
                model1_tiles=tiles1,
                model2_tiles=tiles2,
                captions={
                    "Left": "Left",
                    "M1_Disp": f"{m1} Disp",
                    "M2_Disp": f"{m2} Disp",
                    "M1_Err": f"{m1} Error",
                    "M1_Reg": f"{m1} Region",
                    "M2_Err": f"{m2} Error",
                    "M2_Reg": f"{m2} Region",
                },
                show_group_separators=True,
            )

            out = output_dir / f"{fname_prefix}__{stem}.jpg"
            ok = cv2.imwrite(str(out), page, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                raise IOError(f"Failed to write collage page: {out}")
            print(f"[album_rect] saved: {out}")

        print("✅ Two-model rectangular album pages complete.")

    def run_region_error_album(
        self,
        input_root: Path,
        output_dir: Path,
        *,
        models: Optional[List[str]] = None,
        stems: Optional[List[str]] = None,
        fname_prefix: str = "album_region_error",
    ):
        """
        region_error_vis_global/*.png를 이용해 리전 에러 전용 패널을 생성.
        """
        # 1) 모델 디렉토리 찾기
        model_dirs = self._discover_models(input_root)
        # region_error 디렉토리 추가 확인
        for m, d in model_dirs.items():
            region_err_dir = (Path(d["region"]).parent / "region_error_vis_global")
            if not region_err_dir.is_dir():
                raise FileNotFoundError(f"[{m}] missing 'region_error_vis_global' directory.")
            d["region_err"] = region_err_dir  # 추가 등록

        # 2) 모델 선택
        model_names = models if models else sorted(model_dirs.keys())
        if len(model_names) < 2:
            raise ValueError("Provide at least two model names, or ensure input_root has >=2 models.")
        m1, m2 = model_names[0], model_names[1]

        # 3) 공통 stem 교집합 (disp_vis_global 기준 그대로 재사용)
        all_stems = stems if stems else self._collect_common_stems(model_dirs)

        # 4) 키 정의 (파일명 키)
        keys = {
            "edge":        "edge",       # edge = boundary
            "thinQ":       "thick:thinQ",
            "thickQ":      "thick:thickQ",
            "texless":     "texless",
        }

        # 5) 루프
        for stem in all_stems:
            A = self._assets_for(model_dirs[m1], stem)
            B = self._assets_for(model_dirs[m2], stem)

            left_bgr = B["left"]

            # disparity (각 모델)
            m1_disp = A["disp"]
            m2_disp = B["disp"]

            # others
            m1_edge   = self._assets_for_region_errors(model_dirs[m1]["region_err"], stem, keys["edge"])
            m1_thinQ  = self._assets_for_region_errors(model_dirs[m1]["region_err"], stem, keys["thinQ"])
            m1_thickQ = self._assets_for_region_errors(model_dirs[m1]["region_err"], stem, keys["thickQ"])
            m1_tex    = self._assets_for_region_errors(model_dirs[m1]["region_err"], stem, keys["texless"])

            m2_edge   = self._assets_for_region_errors(model_dirs[m2]["region_err"], stem, keys["edge"])
            m2_thinQ  = self._assets_for_region_errors(model_dirs[m2]["region_err"], stem, keys["thinQ"])
            m2_thickQ = self._assets_for_region_errors(model_dirs[m2]["region_err"], stem, keys["thickQ"])
            m2_tex    = self._assets_for_region_errors(model_dirs[m2]["region_err"], stem, keys["texless"])

            # 안전 보정 (누락시 빈 타일 대체)
            def _nz(x, fallback=None):
                if x is not None:
                    return x
                return fallback if fallback is not None else np.zeros((256,256,3), np.uint8)

            left_u8 = _nz(left_bgr, np.zeros((self.panel_builder.H, self.panel_builder.W, 3), np.uint8))
            m1_disp = _nz(m1_disp, left_u8); m2_disp = _nz(m2_disp, left_u8)

            m1_edge = _nz(m1_edge, left_u8); m1_thinQ = _nz(m1_thinQ, left_u8)
            m1_thickQ = _nz(m1_thickQ, left_u8); m1_tex = _nz(m1_tex, left_u8)

            m2_edge = _nz(m2_edge, left_u8); m2_thinQ = _nz(m2_thinQ, left_u8)
            m2_thickQ = _nz(m2_thickQ, left_u8); m2_tex = _nz(m2_tex, left_u8)

            # 페이지 생성
            page = self.panel_builder.build_two_model_region_error_collage(
                left_bgr = left_u8,
                m1_disp  = m1_disp,
                m2_disp  = m2_disp,
                m1_err_edge   = m1_edge,
                m1_err_thinQ  = m1_thinQ,
                m1_err_thickQ = m1_thickQ,
                m1_err_texless= m1_tex,
                m2_err_edge   = m2_edge,
                m2_err_thinQ  = m2_thinQ,
                m2_err_thickQ = m2_thickQ,
                m2_err_texless= m2_tex,
                captions={
                    "Left": "Left",
                    "M1_Disp": f"{m1} Disp",
                    "M2_Disp": f"{m2} Disp",
                    "M1_Edge":   f"{m1} Error (edge)",
                    "M1_ThinQ":  f"{m1} Error (thick:thinQ)",
                    "M1_ThickQ": f"{m1} Error (thick:thickQ)",
                    "M1_Tex":    f"{m1} Error (texless)",
                    "M2_Edge":   f"{m2} Error (edge)",
                    "M2_ThinQ":  f"{m2} Error (thick:thinQ)",
                    "M2_ThickQ": f"{m2} Error (thick:thickQ)",
                    "M2_Tex":    f"{m2} Error (texless)",
                },
                show_group_separators=True,
            )

            out = output_dir / f"{fname_prefix}__{stem}.jpg"
            ok = cv2.imwrite(str(out), page, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                raise IOError(f"Failed to write region-error collage page: {out}")
            print(f"[album_region_error] saved: {out}")

        print("✅ Two-model region-error album pages complete.")