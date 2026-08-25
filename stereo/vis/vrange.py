from __future__ import annotations
import os
import json
from typing import Optional


class VRangeManager:
    """
    Global visualization range manager.
    panel_compare.py에서 사용하던 _compare/global_vrange.json, _compare/global_tau.json을
    통합적으로 관리한다.
    """
    def __init__(self, vmin: Optional[float] = None, vmax: Optional[float] = None,
                 clip: Optional[str] = None, tau: Optional[float] = None):
        self.vmin = vmin
        self.vmax = vmax
        self.clip = clip
        self.tau = tau

    # ---- Loaders ----
    @classmethod
    def from_compare_dir(cls, output_root: str):
        base = os.path.join(output_root, "_compare")
        vr_path = os.path.join(base, "global_vrange.json")
        tau_path = os.path.join(base, "global_tau.json")

        vmin = vmax = clip = tau = None

        if os.path.exists(vr_path):
            with open(vr_path, "r") as f:
                obj = json.load(f)
                vmin = obj.get("vmin", None)
                vmax = obj.get("vmax", None)
                clip = obj.get("clip", None)

        if os.path.exists(tau_path):
            with open(tau_path, "r") as f:
                obj = json.load(f)
                tau = obj.get("tau", None)

        return cls(vmin=vmin, vmax=vmax, clip=clip, tau=tau)

    # ---- Savers ----
    def save_to_compare_dir(self, output_root: str):
        base = os.path.join(output_root, "_compare")
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, "global_vrange.json"), "w") as f:
            json.dump({"vmin": self.vmin, "vmax": self.vmax, "clip": self.clip}, f, indent=2)
        with open(os.path.join(base, "global_tau.json"), "w") as f:
            json.dump({"tau": self.tau}, f, indent=2)