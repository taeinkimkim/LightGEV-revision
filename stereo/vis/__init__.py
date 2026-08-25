"""
Visualization utilities: disparity colorization, range handling, overlays.
"""
from .disp_color import disp_to_color_bgr
from .vrange import VRangeManager
from .overlay import ErrorOverlay
from .global_baker import extract_global_params, bake_global_visuals

__all__ = ["disp_to_color_bgr", "VRangeManager", "ErrorOverlay", "extract_global_params", "bake_global_visuals"]