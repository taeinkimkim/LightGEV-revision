import os
from dataclasses import dataclass
from typing import Optional, Tuple
from easydict import EasyDict
import torch
import gc

from stereo.utils import common_utils
from stereo.modeling import build_trainer
from cfgs.data_basic import DATA_PATH_DICT
from stereo.io.cache_manager import CacheIO
from stereo.vis.vrange import VRangeManager
from stereo.panel.panel_builder import PanelBuilder
from stereo.evaluation import DepthThicknessEvaluator, RegionMasker, RegionVisualizer
from stereo.evaluation.global_stats import GlobalStatsManager
from stereo.evaluation.single_model_eval import evaluate_single_model
from stereo.vis.global_baker import extract_global_params, bake_global_visuals

# -------------------- 기존 단일모델 Runner 본체 (수정) --------------------
@dataclass
class MultiEvalOptions:
    focal: Optional[float] = 721
    baseline: Optional[float] = 0.54
    calib_root: Optional[str] = None
    max_disp: float = 256.0
    max_depth: float = 80.0
    amp: bool = False
    pred_is_normalized: bool = False
    save_csv: bool = True
    save_npz: bool = True
    save_vis: bool = True
    save_panel: bool = True
    save_depth: bool = True
    save_every: int = 1
    save_region_overlay: bool = True
    region_overlay_alpha: float = 0.4
    eval_mode: str = "auto"  # "auto", "light", "full"

class MultiRunner:
    def __init__(self, trainer, device, output_root, opts: MultiEvalOptions = MultiEvalOptions()):
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
    
    def cleanup(self):
        """
        GPU 메모리 정리:
        - 모델을 CPU로 이동하고 참조 제거
        - 가능한 큰 객체 참조 제거
        - CUDA 캐시/IPC 메모리 수거 + GC
        """
        try:
            model = getattr(self.trainer, "model", None)
            if model is not None:
                try:
                    model.to("cpu")
                except Exception:
                    pass
            # 큰 구성요소 참조 해제
            if hasattr(self.trainer, "eval_loader"):
                self.trainer.eval_loader = None
            if hasattr(self.trainer, "train_loader"):
                self.trainer.train_loader = None

            # 러너 내부 부가 객체도 해제
            self.evaluator = None
            self.region_viz = None   # 마지막 러너의 region_viz가 필요하면 바깥에서 빼두세요
            self.panel = None
            self.cache = None
            self.vrange = None
            self.region_masker = None

            # trainer/model 참조 제거
            if hasattr(self.trainer, "model"):
                delattr(self.trainer, "model")
        except Exception:
            pass

        # 동기화 후 캐시/IPC 수거 + 파이썬 GC
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        gc.collect()
    
    def _setup_model_eval_mode(self):
        """
        모델을 eval_mode 설정에 따라 적절히 구성
        """
        model = self.trainer.model
        model.eval()
        eval_mode = self.opts.eval_mode.lower()
        
        if eval_mode == "light":
            # 경량 추론 모드
            if hasattr(model, 'set_eval_mode'):
                model.set_eval_mode(light=True)
                model.set_fov_window(window_D=model.D_max, tau=4.0)  # 풀 FoV
                setattr(model, "_force_full_window_eval", True)
                if self.trainer.logger:
                    self.trainer.logger.info(f"[Runner] Model set to LIGHT inference mode (memory optimized)")
            else:
                if self.trainer.logger:
                    self.trainer.logger.warning(f"[Runner] Model doesn't support set_eval_mode(), using standard eval()")
        
        elif eval_mode == "full":
            # 전체 파이프라인 모드 (보조 텐서 포함)
            if hasattr(model, 'set_eval_mode'):
                model.set_eval_mode(light=False)
                if self.trainer.logger:
                    self.trainer.logger.info(f"[Runner] Model set to FULL inference mode (with auxiliary outputs)")
        
        else:  # "auto" 또는 기타
            # 기본 동작: set_eval_mode가 있으면 light, 없으면 eval()
            if hasattr(model, 'set_eval_mode'):
                model.set_eval_mode(light=True)
                model.set_fov_window(window_D=model.D_max, tau=4.0)  # 풀 FoV
                setattr(model, "_force_full_window_eval", True)
                if self.trainer.logger:
                    self.trainer.logger.info(f"[Runner] Model set to AUTO mode (defaulting to light)")

    def run(self):
        # 평가 전 모델 모드 설정
        self._setup_model_eval_mode()
        
        # 평가 실행
        evaluate_single_model(
            model=self.trainer.model,
            eval_loader=self.trainer.eval_loader,
            output_root=self.output_root,
            opts=self.opts,
            evaluator=self.evaluator,
            cache=self.cache,
            logger=self.trainer.logger,
        )

def _build_trainer_from_cfg(cfg_path: str, ckpt: str, output_root, out_dir, logger=None):

    yaml_config = common_utils.config_loader(cfg_path)
    cfgs = EasyDict(yaml_config)

    class DummyArgs:
        pass

    args = DummyArgs()
    args.ckpt_dir = output_root
    args.run_mode = 'eval'
    args.dist_mode = False
    args.workers = 8
    args.pin_memory = False
    os.makedirs(out_dir, exist_ok=True)

    cfgs.MODEL.PRETRAINED_MODEL = ckpt

    for each in cfgs.DATA_CONFIG.DATA_INFOS:
        dataset_name = each.DATASET
        if dataset_name == 'KittiDataset':
            dataset_name = 'KittiDataset15' if 'kitti15' in each.DATA_SPLIT.EVALUATING else 'KittiDataset12'
        each.DATA_PATH = DATA_PATH_DICT[dataset_name]
        assert os.path.exists(each.DATA_PATH), '[Errno 2] No such file or directory: {}, You must modify the data root path in cfgs/databasic.py to the path of your own dataset.'.format(each.DATA_PATH)

    trainer = build_trainer(args, cfgs, local_rank=0, global_rank=0, logger=logger, tb_writer=None)

    return trainer

def run_one(cfg: str, ckpt: Optional[str], output_root: str, model_name: str, opts: MultiEvalOptions,
            device: str = "cuda:0", logger=None) -> bool:

    out_dir = os.path.join(output_root, model_name)
    os.makedirs(out_dir, exist_ok=True)

    if logger:
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"  Config: {cfg}")
        logger.info(f"  Checkpoint: {ckpt}")
        logger.info(f"  Eval mode: {opts.eval_mode}")
        logger.info(f"  Output: {out_dir}")
        logger.info(f"{'='*80}\n")

    trainer = _build_trainer_from_cfg(cfg, ckpt, output_root, out_dir, logger)
    runner = MultiRunner(trainer, device, out_dir, opts)
    runner.run()

    return runner

def run_from_yaml(models_yaml: str, output_root: str, opts: MultiEvalOptions, device: str = "cuda:0", logger=None):
    """
    YAML 파일에서 여러 모델을 읽어 순차적으로 평가
    
    각 모델의 eval_mode 설정을 존중:
    - "light": 경량 추론 (메모리 최적화)
    - "full": 전체 파이프라인 (보조 텐서 포함)
    - "auto": 자동 (light 우선)
    """
    
    yaml_config = common_utils.config_loader(models_yaml)
    spec = EasyDict(yaml_config)

    models = spec.get("models") or {}
    if not models:
        raise RuntimeError(f"No 'models' found in {models_yaml}")

    model_names = []
    last_region_viz = None

    for name, m in models.items():
        cfg = m.get("cfg")
        ckpt = m.get("ckpt")
        if not cfg or not ckpt:
            if logger: 
                logger.warning(f"[SKIP] {name}: need cfg and ckpt")
            continue

        # 모델별 옵션 복사
        local = MultiEvalOptions(**vars(opts))
        
        # YAML에서 개별 설정 오버라이드
        if "pred_is_normalized" in m:
            local.pred_is_normalized = bool(m["pred_is_normalized"])
        
        # eval_mode: YAML 설정이 있으면 우선 사용, 없으면 CLI/기본값 사용
        if "eval_mode" in m:
            local.eval_mode = str(m["eval_mode"])
            if logger:
                logger.info(f"[CONFIG] {name}: Using eval_mode='{local.eval_mode}' from YAML")
        elif opts.eval_mode and opts.eval_mode != "auto":
            # CLI에서 명시적으로 지정한 경우
            if logger:
                logger.info(f"[CONFIG] {name}: Using eval_mode='{local.eval_mode}' from CLI")
        else:
            if logger:
                logger.info(f"[CONFIG] {name}: Using default eval_mode='{local.eval_mode}'")
        
        if "description" in m and logger:
            logger.info(f"[INFO] {name}: {m['description']}")

        # 모델 평가 실행
        if logger: 
            logger.info(f"[RUN] {name}")
        
        try:
            runner = run_one(cfg, ckpt, output_root, name, local, device=device, logger=logger)
            # 다음 모델을 위해 region_viz만 보존
            try:
                last_region_viz = getattr(runner, "region_viz", None)
            except Exception:
                last_region_viz = None

            model_names.append(name)

            # ★ 평가 끝나면 즉시 정리
            runner.cleanup()

        except Exception as e:
            if logger:
                logger.error(f"[ERROR] {name} failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
            continue

    # 글로벌 통계 및 시각화
    if logger:
        logger.info(f"\n{'='*80}")
        logger.info("Computing global statistics and generating comparison visualizations...")
        logger.info(f"{'='*80}\n")
    
    stats = GlobalStatsManager.update_all(
        output_root, model_names, 
        vmax_mode="quantile", vmax_q=0.95, 
        vmin_mode="auto", vmin_q=0.01,
        tau_mode="quantile", tau_q=0.9, tau_px=3.0, 
        logger=logger
    )
    
    if model_names:
        global_vmin, global_vmax, global_tau = extract_global_params(stats)
        bake_global_visuals(
            output_root, model_names,
            vmin=global_vmin, vmax=global_vmax, tau_px=global_tau,
            region_viz=last_region_viz,
            logger=logger
        )
    
    if logger:
        logger.info(f"\n{'='*80}")
        logger.info("Multi-model evaluation completed!")
        logger.info(f"Results saved to: {output_root}")
        logger.info(f"{'='*80}\n")