import os, sys, argparse, datetime
import random, numpy as np, torch

sys.path.insert(0, './')
from stereo.utils import common_utils
from stereo.runners.single_runner import run_single_model, SingleEvalOptions


def _make_logger(output_dir):
    log_file = os.path.join(output_dir, 'eval_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=0)
    return logger

def parse_args():
    p = argparse.ArgumentParser("Single-model evaluator (stereo)")
    
    # Model configuration
    p.add_argument("--model_name", type=str, required=True,
                   help="Model's name")
    p.add_argument("--cfg", type=str, required=True,
                   help="Model configuration file path")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Model checkpoint file path")
    p.add_argument("--output_root", type=str, required=True,
                   help="Output directory for evaluation results")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="CUDA device to use")
    
    # Evaluation parameters
    p.add_argument("--max_disp", type=int, default=192,
                   help="Maximum disparity value")
    p.add_argument("--tau_px", type=float, default=10.0,
                   help="Tau px")
    p.add_argument("--gt_out_of_range", type=str, default="clip",
                   choices=["clip", "invalid"],
                   help="GT disparity 가 --max_disp 를 넘을 때의 처리. "
                        "clip: max_disp 로 잘라서 그대로 평가에 포함 (기존 동작). "
                        "invalid: 0 으로 만들어 평가에서 제외. "
                        "Middlebury 를 원본 크기에 가깝게 평가할 때는 invalid 여야 "
                        "수치가 해석 가능하다 (1배에서 GT 최대 1074px, 23.8%% 가 범위 밖).")
    p.add_argument("--eval_mode", type=str, default="auto",
                   choices=["auto", "light", "full"],
                   help="Evaluation mode: auto (default light), light (memory optimized), or full (with auxiliary outputs)")
    p.add_argument("--pred_is_normalized", action="store_true",
                   help="Whether model predictions are normalized")
    
    # Output options
    p.add_argument("--save_every", type=int, default=1,
                   help="Save visualization every N samples")
    p.add_argument("--no_csv", action="store_true",
                   help="Don't save CSV results")
    p.add_argument("--no_npz", action="store_true",
                   help="Don't save NPZ cache")
    p.add_argument("--no_vis", action="store_true",
                   help="Don't save visualizations")
    p.add_argument("--no_panel", action="store_true",
                   help="Don't save panel visualizations")
    p.add_argument("--no_depth", action="store_true",
                   help="Don't save depth maps")
    
    # Other
    p.add_argument("--seed", type=int, default=666,
                   help="Random seed")
    
    return p.parse_args()

def _set_seeds(seed=666):
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def main():
    args = parse_args()
    output_dir = os.path.join(args.output_root, args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    logger = _make_logger(output_dir)
    
    logger.info("="*80)
    logger.info("Single-Model Stereo Evaluation")
    logger.info("="*80)
    logger.info(f"Config: {args.cfg}")
    logger.info(f"Checkpoint: {args.ckpt}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Eval mode: {args.eval_mode}")
    logger.info(f"Seed: {args.seed}")
    logger.info("="*80 + "\n")
    
    _set_seeds(args.seed)

    opts = SingleEvalOptions(
        model_name=args.model_name,
        max_disp=float(args.max_disp), 
        tau_px=float(args.tau_px),
        eval_mode=args.eval_mode,
        gt_out_of_range=args.gt_out_of_range,
        pred_is_normalized=args.pred_is_normalized,
        save_csv=not args.no_csv, 
        save_npz=not args.no_npz, 
        save_vis=not args.no_vis, 
        save_panel=not args.no_panel,
        save_every=args.save_every,
    )

    runner = run_single_model(
        cfg=args.cfg,
        ckpt=args.ckpt,
        output_dir=output_dir,
        opts=opts,
        device=args.device,
        logger=logger
    )
    
    logger.info(f"\n{'='*80}")
    logger.info("Single-model evaluation completed!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*80}\n")

if __name__ == "__main__":
    main()