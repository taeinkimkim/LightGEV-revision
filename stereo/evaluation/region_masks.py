from __future__ import annotations
import numpy as np
import cv2
from typing import Optional


def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr.astype(np.uint8)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _local_std(gray: np.ndarray, ksize: int = 7) -> np.ndarray:
    """Fast local std using box filter (approximate)."""
    g = gray.astype(np.float32)
    k = (ksize, ksize)
    mean = cv2.boxFilter(g, ddepth=-1, ksize=k, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean2 = cv2.boxFilter(g * g, ddepth=-1, ksize=k, normalize=True, borderType=cv2.BORDER_REFLECT)
    var = np.maximum(mean2 - mean * mean, 0.0)
    return np.sqrt(var)


def _safe_canny(gray: np.ndarray, low: float, high: float) -> np.ndarray:
    # OpenCV Canny expects 8-bit
    if gray.dtype != np.uint8:
        # auto rescale into 8-bit
        g = gray.astype(np.float32)
        g = (g - g.min()) / (g.max() - g.min() + 1e-6)
        gray8 = (g * 255.0).astype(np.uint8)
    else:
        gray8 = gray
    return cv2.Canny(gray8, threshold1=low, threshold2=high, L2gradient=True)


def _morph_thin(binary01: np.ndarray) -> np.ndarray:
    """1-pixel skeletonization fallback if ximgproc.thinning is not available."""
    bin8 = (binary01.astype(np.uint8) * 255)
    prev = np.zeros_like(bin8)
    skel = np.zeros_like(bin8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(bin8, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(bin8, temp)
        skel = cv2.bitwise_or(skel, temp)
        bin8 = eroded.copy()
        if cv2.countNonZero(bin8) == 0 or np.array_equal(skel, prev):
            break
        prev = skel.copy()
    return (skel > 0).astype(np.uint8)


class RegionMasker:
    def __init__(
        self,
        tex_ksize: int = 7,
        tex_std_thresh: float = 5.0,
        canny_low: float = 50.0,
        canny_high: float = 150.0,
        boundary_band_px: int = 2
    ):
        """
        Args
        - tex_ksize: local std window size (odd).
        - tex_std_thresh: std < thresh → textureless(1).
        - canny_*: edge detector thresholds for boundary/thin extraction.
        - boundary_band_px: band half-width around disparity edges for boundary mask.
        """
        self.tex_ksize = int(tex_ksize if tex_ksize % 2 == 1 else tex_ksize + 1)
        self.tex_std_thresh = float(tex_std_thresh)
        self.canny_low = float(canny_low)
        self.canny_high = float(canny_high)
        self.boundary_band_px = int(boundary_band_px)

        # optional ximgproc
        try:
            self._ximgproc = cv2.ximgproc  # type: ignore[attr-defined]
        except Exception:
            self._ximgproc = None

    # ------------------------ Public API ------------------------

    def textureless(self, left_bgr: np.ndarray) -> np.ndarray:
        """
        Return mask==1 on textureless (low local std) regions.
        """
        gray = _to_gray(left_bgr)
        s = _local_std(gray, self.tex_ksize)
        m = (s < self.tex_std_thresh).astype(np.uint8)
        # small cleanup
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        return m

    def boundary_band_from_disp(self, disp_hw: np.ndarray, band_px: Optional[int] = None) -> np.ndarray:
        """
        Use disparity gradient to locate depth/disparity boundaries and expand to a band.

        If GT disp is provided use that; otherwise can be predicted disp.
        """
        if disp_hw is None or disp_hw.size == 0:
            return np.zeros((0, 0), np.uint8)
        d = disp_hw.astype(np.float32)
        # normalize into [0,255] for Canny stability
        dd = d.copy()
        dd[~np.isfinite(dd)] = 0.0
        if dd.max() > dd.min():
            dn = (dd - dd.min()) / (dd.max() - dd.min() + 1e-6)
        else:
            dn = dd * 0.0
        gray8 = (dn * 255.0).astype(np.uint8)
        edges = _safe_canny(gray8, self.canny_low, self.canny_high)  # 0/255

        band = int(self.boundary_band_px if band_px is None else band_px)
        if band > 0:
            k = 2 * band + 1
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            bandmask = cv2.dilate((edges > 0).astype(np.uint8), se)
        else:
            bandmask = (edges > 0).astype(np.uint8)
        return bandmask

    def thin_from_edges(self, left_bgr: np.ndarray, band: Optional[np.ndarray] = None, width: int = 1) -> np.ndarray:
        """
        Produce a 1-pixel thin edge mask (optionally intersected with a band).
        """
        gray = _to_gray(left_bgr)
        edges = _safe_canny(gray, self.canny_low, self.canny_high) > 0  # bool
        # thinning
        if self._ximgproc is not None and hasattr(self._ximgproc, "thinning"):
            thin = self._ximgproc.thinning((edges.astype(np.uint8) * 255))
            thin01 = (thin > 0).astype(np.uint8)
        else:
            thin01 = _morph_thin(edges.astype(np.uint8))

        if band is not None and band.size != 0:
            # ensure same shape
            if band.shape != thin01.shape:
                band_rs = cv2.resize(band.astype(np.uint8), (thin01.shape[1], thin01.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            else:
                band_rs = band.astype(np.uint8)
            thin01 = (thin01.astype(bool) & (band_rs.astype(bool))).astype(np.uint8)

        if width > 1:
            k = int(max(1, width))
            se = cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))
            thin01 = cv2.dilate(thin01, se)

        return thin01
