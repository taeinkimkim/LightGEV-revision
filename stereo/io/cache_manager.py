from __future__ import annotations
import os
import numpy as np
from typing import Dict, Any, Optional


class CacheIO:
    """
    Panel/비교 단계에서 공통 사용되는 npz 캐시 I/O.
    - save_npz: dict를 통째로 저장 (left_bgr, disp_pred, disp_gt, masks, depth_bgr 등)
    - load_npz: dict 형태로 로드 (마스크 키는 문자열 그대로 유지: "depth:near" 등)
    """

    @staticmethod
    def save_npz(path: str, payload: Dict[str, Any]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 마스크는 bool→uint8로 저장 (용량↓), 로드 시 bool로 변환
        pay = {}
        for k, v in payload.items():
            if k == "masks" and isinstance(v, dict):
                for mk, mv in v.items():
                    pay[f"mask::{mk}"] = (mv.astype(np.uint8) if hasattr(mv, "astype") else mv)
            else:
                pay[k] = v
        np.savez_compressed(path, **pay)

    @staticmethod
    def load_npz(path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(path):
            return None
        z = np.load(path, allow_pickle=True)
        out: Dict[str, Any] = {"masks": {}}
        for k in z.files:
            if k.startswith("mask::"):
                mk = k.split("mask::", 1)[1]
                mv = z[k]
                out["masks"][mk] = (mv.astype(bool) if hasattr(mv, "astype") else mv)
            else:
                out[k] = z[k]
        # disp_gt가 비어있는 경우 처리
        if "disp_gt" in out and (out["disp_gt"] is not None) and out["disp_gt"].size == 0:
            out["disp_gt"] = None
        return out