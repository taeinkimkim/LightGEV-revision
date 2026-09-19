"""region-wise 지표를 **torch 없이** 계산한다 (Jetson 용).

`DepthThicknessEvaluator` 와 같은 결과를 내되 torch 를 요구하지 않는다. 그게 가능한
이유는 무거운 쪽인 region 마스크 생성(`RegionBinner`/`RegionMasker`)이 이미 numpy+cv2
로만 돼 있어서다. torch 를 쓰는 부분은 지표 산술뿐이라 그것만 numpy 로 옮겼다.

마스크 코드는 **복사하지 않고 원본을 그대로 import 한다.** 다만
`stereo/evaluation/__init__.py` 가 torch 를 쓰는 모듈들(metric_per_image,
single_model_eval, depth_thickness_evaluator)을 import 하기 때문에, **torch 가 없을
때만** 패키지 __init__ 을 건너뛰고 필요한 모듈만 직접 로드한다. torch 가 있는
데스크탑에서는 평범한 import 를 쓰므로 다른 코드에 영향이 없다. 마스크 정의가 한 곳에만 있으니
나중에 region 정의를 바꿔도 Jetson 쪽이 자동으로 따라간다.

지표 정의는 stereo/evaluation/metric_per_image.py 를 1:1 로 옮긴 것이다:
    epe  : 유효 픽셀의 |pred-gt| 평균
    d1   : |e|>3 이고 |e|/|gt|>0.05 인 비율 (%)
    t1/t3: |e|>1, |e|>3 인 비율 (%)
데스크탑에서 torch 구현과 같은 값이 나오는지 대조해 두었다
(tools/eval_sceneflow_engine.py --check-metrics).
"""

import importlib.util
import os
import sys
import types

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EVAL_DIR = os.path.join(_REPO, "stereo", "evaluation")


def _load_region_modules():
    """region 마스크 모듈과 disp 후처리 모듈을 가져온다.

    1) 평범하게 import 해 본다. torch 가 있는 환경(데스크탑)에서는 이걸로 끝이고,
       sys.modules 를 건드리지 않으므로 다른 코드의 import 에 영향이 없다.
    2) 실패하면(= torch 가 없어 stereo/evaluation/__init__.py 가 죽으면) 패키지
       __init__ 을 건너뛰고 필요한 모듈만 파일 경로로 직접 로드한다.
    """
    try:
        from stereo.evaluation import region_binner as binner_mod
        from stereo.evaluation.single_helpers import disp_depth_ops as ops
        return binner_mod, ops
    except ImportError:
        pass
    return _load_bypassing_package_init()


def _load_bypassing_package_init():
    """stereo.evaluation 의 __init__ 을 실행하지 않고 region 모듈만 로드한다."""
    if "stereo" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "stereo", os.path.join(_REPO, "stereo", "__init__.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["stereo"] = mod
        spec.loader.exec_module(mod)

    # 빈 껍데기를 먼저 sys.modules 에 넣어 두면 __init__ 이 실행되지 않는다.
    if "stereo.evaluation" not in sys.modules:
        pkg = types.ModuleType("stereo.evaluation")
        pkg.__path__ = [_EVAL_DIR]
        sys.modules["stereo.evaluation"] = pkg

    def load(name):
        full = "stereo.evaluation." + name
        if full in sys.modules:
            return sys.modules[full]
        spec = importlib.util.spec_from_file_location(
            full, os.path.join(_EVAL_DIR, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    load("region_masks")                       # region_binner 가 이걸 import 한다
    binner_mod = load("region_binner")


    ops_name = "stereo.evaluation.single_helpers.disp_depth_ops"
    if ops_name not in sys.modules:
        if "stereo.evaluation.single_helpers" not in sys.modules:
            sub = types.ModuleType("stereo.evaluation.single_helpers")
            sub.__path__ = [os.path.join(_EVAL_DIR, "single_helpers")]
            sys.modules["stereo.evaluation.single_helpers"] = sub
        spec = importlib.util.spec_from_file_location(
            ops_name, os.path.join(_EVAL_DIR, "single_helpers", "disp_depth_ops.py"))
        ops = importlib.util.module_from_spec(spec)
        sys.modules[ops_name] = ops
        spec.loader.exec_module(ops)
    return binner_mod, sys.modules[ops_name]


_binner_mod, _ops = _load_region_modules()

RegionBinner = _binner_mod.RegionBinner
BinningConfig = _binner_mod.BinningConfig

resize_disparity_with_scale_x = _ops._resize_disparity_with_scale_x
denorm_disparity = _ops._denorm_disparity
sanitize_and_clip = _ops._sanitize_and_clip


def _to_hw(a):
    a = np.asarray(a)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    return a


def region_metrics(pred, gt, mask):
    """metric_per_image.py 의 epe/d1/threshold 를 numpy 로 옮긴 것."""
    e = np.abs(pred - gt)
    m = mask.astype(bool)
    n = int(m.sum())
    if n == 0:
        return None
    e_m = e[m]
    gt_m = np.abs(gt[m])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(gt_m > 0, e_m / np.maximum(gt_m, 1e-12), np.inf)
    return {
        "npix": n,
        "epe": float(e_m.mean()),
        "d1": float(((e_m > 3) & (rel > 0.05)).mean() * 100.0),
        "t1": float((e_m > 1).mean() * 100.0),
        "t3": float((e_m > 3).mean() * 100.0),
    }


class NumpyRegionEvaluator:
    """DepthThicknessEvaluator.evaluate 와 같은 입출력. torch 를 쓰지 않는다."""

    def __init__(self, bin_cfg=None):
        self.binner = RegionBinner(bin_cfg or BinningConfig())

    def evaluate(self, left_bgr, disp_pred, disp_gt):
        if disp_gt is None or np.size(disp_gt) == 0:
            return {}, {}
        gt = _to_hw(disp_gt)
        pred = _to_hw(disp_pred)

        masks = self.binner.build_masks(left_bgr, gt)

        # depth_thickness_evaluator 와 같은 순서: region 마스크에 global 유효 마스크를
        # 곱하고, 그 합집합을 'global' 로 쓴다.
        valid_global = np.isfinite(gt) & (gt > 0)
        out_masks, union = {}, np.zeros_like(valid_global, dtype=bool)
        for k, m in masks.items():
            mm = _to_hw(np.asarray(m)).astype(bool) & valid_global
            out_masks[k] = mm
            union |= mm
        out_masks["global"] = union

        res = {}
        for name, m in out_masks.items():
            r = region_metrics(pred, gt, m)
            if r is not None:
                res[name] = r
        return res, out_masks
