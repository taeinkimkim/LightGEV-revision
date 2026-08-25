import os
from dataclasses import dataclass
from typing import Optional, Tuple
from easydict import EasyDict
import numpy as np

from stereo.utils import common_utils
from stereo.modeling import build_trainer
from cfgs.data_basic import DATA_PATH_DICT
from stereo.io.cache_manager import CacheIO
from stereo.vis.vrange import VRangeManager
from stereo.panel.panel_builder import PanelBuilder
from stereo.evaluation import DepthThicknessEvaluator, RegionMasker, RegionVisualizer
from stereo.evaluation.single_model_eval import evaluate_single_model
from stereo.vis.global_baker import bake_global_visuals

@dataclass
class SingleEvalOptions:
    model_name: str = None
    max_disp: float = 192.0
    tau_px: float = 10
    amp: bool = False
    pred_is_normalized: bool = False
    save_csv: bool = True
    save_npz: bool = True
    save_vis: bool = True
    save_panel: bool = True
    save_every: int = 1
    save_region_overlay: bool = True
    region_overlay_alpha: float = 0.4
    eval_mode: str = "auto"  # "auto", "light", "full"

class SingleRunner:
    def __init__(self, trainer, device, output_root, opts: SingleEvalOptions = SingleEvalOptions()):
        self.trainer = trainer
        self.device = device
        self.output_root = output_root
        self.opts = opts
        self.cache = CacheIO()
        self.vrange = VRangeManager()
        self.panel = PanelBuilder()
        self.region_masker = RegionMasker()
        self.evaluator = DepthThicknessEvaluator(device=device)
        self.region_viz = RegionVisualizer()

    def _fb_from_kitti(self, left_name: str) -> Optional[Tuple[float, float]]:
        if not self.opts.calib_root:
            return None
        
        base = os.path.basename(left_name).split("_")[0]
        calib_file = os.path.join(self.opts.calib_root, base + ".txt")

        if not os.path.exists(calib_file):
            return None
        
        with open(calib_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith("P2:"):
                vals = [float(x) for x in line.strip().split()[1:]]
                fx = vals[0]
                tx = vals[3]
                baseline = abs(tx / fx)
                return fx, baseline
            
        return None
    
    def _setup_model_eval_mode(self):
        """
        모델을 eval_mode 설정에 따라 적절히 구성
        """
        model = self.trainer.model
        model.eval()
        eval_mode = self.opts.eval_mode.lower()
        
        if eval_mode == "light":
            if hasattr(model, 'set_eval_mode'):
                model.set_eval_mode(light=True)
                model.set_fov_window(window_D=model.D_max, tau=4.0)
                setattr(model, "_force_full_window_eval", True)
                if self.trainer.logger:
                    self.trainer.logger.info(f"[Runner] Model set to LIGHT inference mode (memory optimized)")
            else:
                if self.trainer.logger:
                    self.trainer.logger.warning(f"[Runner] Model doesn't support set_eval_mode(), using standard eval()")
        
        elif eval_mode == "full":
            if hasattr(model, 'set_eval_mode'):
                model.set_eval_mode(light=False)
                if self.trainer.logger:
                    self.trainer.logger.info(f"[Runner] Model set to FULL inference mode (with auxiliary outputs)")
        
        else:  # "auto" 또는 기타
            if hasattr(model, 'set_eval_mode'):
                model.set_eval_mode(light=True)
                model.set_fov_window(window_D=model.D_max, tau=4.0)
                setattr(model, "_force_full_window_eval", True)
                if self.trainer.logger:
                    self.trainer.logger.info(f"[Runner] Model set to AUTO mode (defaulting to light)")

    def run(self):
        # 평가 전 모델 모드 설정
        self._setup_model_eval_mode()
        
        # 평가 실행
        all_gts = evaluate_single_model(
                    model=self.trainer.model,
                    eval_loader=self.trainer.test_loader,
                    output_root=self.output_root,
                    opts=self.opts,
                    evaluator=self.evaluator,
                    cache=self.cache,
                    logger=self.trainer.logger,
                )
        
        return all_gts

def _build_trainer_from_cfg(cfg_path: str, ckpt: str, output_root, logger=None):
    yaml_config = common_utils.config_loader(cfg_path)
    cfgs = EasyDict(yaml_config)

    class DummyArgs:
        pass

    args = DummyArgs()
    args.ckpt_dir = output_root
    args.run_mode = 'test'
    args.dist_mode = False
    args.workers = 8
    args.pin_memory = False
    os.makedirs(output_root, exist_ok=True)

    cfgs.MODEL.PRETRAINED_MODEL = ckpt

    for each in cfgs.DATA_CONFIG.DATA_INFOS:
        dataset_name = each.DATASET
        if dataset_name == 'KittiDataset':
            dataset_name = 'KittiDataset15' if 'kitti15' in each.DATA_SPLIT.EVALUATING else 'KittiDataset12'
        each.DATA_PATH = DATA_PATH_DICT[dataset_name]
        assert os.path.exists(each.DATA_PATH), '[Errno 2] No such file or directory: {}, You must modify the data root path in cfgs/databasic.py to the path of your own dataset.'.format(each.DATA_PATH)

    trainer = build_trainer(args, cfgs, local_rank=0, global_rank=0, logger=logger, tb_writer=None)

    return trainer

def compute_global_gt_range(all_gts, percentile=99.5):
    """모든 GT disparity로부터 global vmin/vmax 계산"""
    valid = [gt[gt > 0] for gt in all_gts if gt is not None]
    if len(valid) == 0:
        return 0.0, None  # fallback

    merged = np.concatenate([v.reshape(-1) for v in valid])
    vmin = float(np.percentile(merged, 0.0))      # 일반적으로 0
    vmax = float(np.percentile(merged, percentile))
    return vmin, vmax

def run_single_model(cfg: str, ckpt: str, output_dir: str, opts: SingleEvalOptions,
                     device: str = "cuda:0", logger=None) -> SingleRunner:
    """
    단일 모델 평가 실행
    
    Args:
        cfg: 모델 설정 파일 경로
        ckpt: 체크포인트 파일 경로
        output_root: 결과 저장 디렉토리
        opts: 평가 옵션
        device: 사용할 디바이스
        logger: 로거 객체
    
    Returns:
        SingleRunner 객체
    """
    os.makedirs(output_dir, exist_ok=True)

    if logger:
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating Single Model")
        logger.info(f"  Config: {cfg}")
        logger.info(f"  Checkpoint: {ckpt}")
        logger.info(f"  Eval mode: {opts.eval_mode}")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"{'='*80}\n")

    trainer = _build_trainer_from_cfg(cfg, ckpt, output_dir, logger)
    runner = SingleRunner(trainer, device, output_dir, opts)
    all_gts = runner.run()

    if opts.save_npz and opts.save_vis:
        
        gt_global_vmin, gt_global_vmax = compute_global_gt_range(all_gts)

        bake_global_visuals(
            output_dir, [runner.opts.model_name],
            vmin=gt_global_vmin, vmax=gt_global_vmax, tau_px=runner.opts.tau_px,
            region_viz=runner.region_viz,
            logger=logger
        )

    return runner