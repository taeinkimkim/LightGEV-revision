from typing import Any
import os, csv, time
import numpy as np
import torch
import torch.nn as nn
from ptflops import get_model_complexity_info  # FLOPs 계산용 라이브러리

from stereo.io.path_utils import ensure_dirs
from stereo.evaluation.single_helpers import (
    _extract_name_and_input_size,
    _tensor_to_bgr,
    _tensor_to_numpy_chw,
    _resize_disparity_with_scale_x,
    _denorm_disparity,
    _sanitize_and_clip,
    _write_cache_npz,
)
from stereo.datasets.dataset_utils.stereo_trans import unpad_prediction


def force_bn_use_batch_stats(model: torch.nn.Module):
    cnt = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            cnt += 1
    print(f"[BN] forced batch-stats mode for {cnt} BN layers.")


def compute_model_flops(model, input_shape=(1, 3, 256, 512)):
    """
    input_shape = (B, C, H, W) 기준으로 FLOPs 계산 (GMAC 단위).
    OpenStereo 모델들은 보통 data dict를 입력으로 받기 때문에,
    ptflops용으로 Tensor → dict(left, right)로 변환하는 래퍼를 씌워서 계산한다.
    """
    device = next(model.parameters()).device

    class FlopsWrapper(nn.Module):
        def __init__(self, m: nn.Module):
            super().__init__()
            self.m = m

        def forward(self, x):
            # ptflops가 tuple이나 list로 줄 수도 있으니 첫 요소만 사용
            if isinstance(x, (tuple, list)):
                x = x[0]
            # x: [B, C, H, W] Tensor 라고 가정
            data = {
                "left": x,
                "right": x,  # FLOPs만 중요하니 오른쪽도 같은 크기 텐서로 사용
            }
            return self.m(data)

    wrapper = FlopsWrapper(model).to(device)
    wrapper.eval()

    with torch.no_grad():
        try:
            macs, params = get_model_complexity_info(
                wrapper,
                input_res=input_shape[1:],  # (C, H, W)
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False,
            )
        except Exception as e:
            # 실패하면 None 리턴해서 나중에 summary에서 빈 칸으로 처리
            print(f"[FLOPs] estimation failed inside compute_model_flops: {e}")
            return None

    if macs is None:
        return None

    gmac = macs / 1e9  # GMAC
    return gmac


def evaluate_single_model(model, eval_loader, output_root: str, opts: Any,
                          evaluator=None, cache=None, logger=None) -> None:

    max_disp   = float(getattr(opts, "max_disp"))
    use_amp    = bool(getattr(opts, "amp"))
    pred_norm  = bool(getattr(opts, "pred_is_normalized"))
    gt_oor     = str(getattr(opts, "gt_out_of_range", "clip"))

    io = ensure_dirs(output_root)
    csv_path = os.path.join(output_root, "results.csv")

    # -----------------------
    # FLOPs 계산 (첫 입력 크기 기반)
    # -----------------------
    flops_gmac = None
    try:
        sample_batch = next(iter(eval_loader))
        _, H0, W0 = _extract_name_and_input_size(sample_batch)
        flops_gmac = compute_model_flops(model, input_shape=(1, 3, H0, W0))
    except Exception as e:
        if logger:
            logger.warning(f"[FLOPs] estimation failed: {e}")
        flops_gmac = None

    with open(csv_path, "w", newline="") as csv_f:
        csv_w = csv.writer(csv_f)
        csv_w.writerow([
            "name", "metric", "npix", "epe", "d1", "t1", "t3",
            "infer_time", "peak_mem_gb", "flops_gmac"
        ])

        num_samples = 0

        # 순수 inference-only 리스트
        infer_times = []
        infer_peak_mems = []

        # ------------------------
        # 전체 평균을 위한 accumulator (npix-weighted)
        # ------------------------
        total_npix = 0
        total_epe_n = 0.0   # Σ (epe * npix)
        total_d1_n  = 0.0
        total_t1_n  = 0.0
        total_t3_n  = 0.0

        # ------------------------
        # region별 평균을 위한 accumulator
        # ------------------------
        region_stats = {}

        all_gts = []

        with torch.no_grad():
            for i, data in enumerate(eval_loader):
                stem, H, W = _extract_name_and_input_size(data)

                # 데이터 GPU 이동 → 측정 제외
                for k, v in data.items():
                    data[k] = v.to("cuda") if torch.is_tensor(v) else v

                # ------------------------------
                # 순수 inference 측정
                # ------------------------------
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                t0 = time.time()

                with torch.amp.autocast("cuda", enabled=use_amp):
                    output = model(data)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    t1 = time.time()
                    infer_times.append(t1 - t0)

                    peak_mem_bytes = torch.cuda.max_memory_allocated()
                    infer_peak_mems.append(peak_mem_bytes / (1024 ** 3))
                else:
                    t1 = time.time()
                    infer_times.append(t1 - t0)
                    infer_peak_mems.append(0.0)

                # ------------------------------
                # 이후 후처리 (측정 제외)
                # ------------------------------
                if isinstance(output, dict):
                    for k, v in list(output.items()):
                        if torch.is_tensor(v):
                            output[k] = v.float().cpu()
                elif torch.is_tensor(output):
                    output = output.float().cpu()
                else:
                    if logger:
                        logger.warning(f"[eval] unexpected output type {type(output)} for {stem}")
                    continue

                if "disp_pred" not in output:
                    if logger:
                        logger.warning(f"[eval] No disp_pred for {stem}")
                    continue

                try:
                    left_ori_bgr = _tensor_to_bgr(data.get("left_ori", None))
                except Exception:
                    left_ori_bgr = _tensor_to_bgr(data.get("left", None))

                disp_pred = output["disp_pred"]
                pad = data.get("_pad", (0, 0, 0, 0))
                disp_pred = unpad_prediction(disp_pred, pad)
                disp_pred = torch.nan_to_num(disp_pred, nan=0.0)

                disp_np = _tensor_to_numpy_chw(disp_pred)
                disp_np = _resize_disparity_with_scale_x(disp_np, (H, W))
                disp_np = _denorm_disparity(disp_np, max_disp, pred_norm)
                disp_np = _sanitize_and_clip(disp_np, 0.0, max_disp)

                disp_gt_t = data.get("disp", None)
                disp_gt_np = None
                if disp_gt_t is not None:
                    disp_gt_np = _tensor_to_numpy_chw(disp_gt_t)
                    disp_gt_np = _resize_disparity_with_scale_x(disp_gt_np, (H, W))
                    if gt_oor == "invalid":
                        # max_disp 를 넘는 GT 는 clip 하면 "예측도 max_disp 에서
                        # 잘리므로 오차 0" 이 되어 성능이 좋아 보인다. 0 으로
                        # 만들어 유효 마스크(disp > 0)에서 빠지게 한다.
                        disp_gt_np = np.nan_to_num(disp_gt_np, nan=0.0, posinf=0.0, neginf=0.0)
                        disp_gt_np = np.where(disp_gt_np > max_disp, 0.0, disp_gt_np)
                        disp_gt_np = _sanitize_and_clip(disp_gt_np, 0.0, max_disp)
                    else:
                        disp_gt_np = _sanitize_and_clip(disp_gt_np, 0.0, max_disp)

                all_gts.append(disp_gt_np)

                masks = None
                if evaluator and disp_gt_np is not None:
                    results, masks = evaluator.evaluate(left_ori_bgr, disp_np, disp_gt_np)

                    for region_name, m in results.items():
                        npix = int(m.get("npix", 0))
                        epe  = float(m["epe"])
                        d1   = float(m["d1"])
                        t1_  = float(m["t1"])
                        t3_  = float(m["t3"])

                        # per-sample region 결과는 그대로 기록
                        csv_w.writerow([
                            stem, region_name, npix,
                            f"{epe:.6f}", f"{d1:.6f}",
                            f"{t1_:.6f}", f"{t3_:.6f}",
                            "", "", ""
                        ])

                        # -------- 전체 global metric 누적 (npix-weighted) --------
                        total_npix   += npix
                        total_epe_n  += epe * npix
                        total_d1_n   += d1  * npix
                        total_t1_n   += t1_ * npix
                        total_t3_n   += t3_ * npix

                        # -------- region별 통계 누적 --------
                        if region_name not in region_stats:
                            region_stats[region_name] = {
                                "npix":  0,
                                "epe_n": 0.0,
                                "d1_n":  0.0,
                                "t1_n":  0.0,
                                "t3_n":  0.0,
                            }

                        region_stats[region_name]["npix"]  += npix
                        region_stats[region_name]["epe_n"] += epe * npix
                        region_stats[region_name]["d1_n"]  += d1  * npix
                        region_stats[region_name]["t1_n"]  += t1_ * npix
                        region_stats[region_name]["t3_n"]  += t3_ * npix

                num_samples += 1

                if getattr(opts, "save_npz", False):
                    cache_dict = dict(
                        left_bgr=left_ori_bgr,
                        disp_gt=disp_gt_np,
                        disp_pred=disp_np,
                        masks=masks,
                    )
                    _write_cache_npz(cache, io["cache_npz"], cache_dict, stem)
                    if logger:
                        logger.info(f'{stem}.npz was saved')
                
                logger.info(f'{i}')

            # ------------------------------
            # SUMMARY: global metric + inference-only FPS/Memory
            # ------------------------------
            if total_npix > 0:
                mean_epe = total_epe_n / total_npix
                mean_d1  = total_d1_n  / total_npix
                mean_t1  = total_t1_n  / total_npix
                mean_t3  = total_t3_n  / total_npix
            else:
                mean_epe = mean_d1 = mean_t1 = mean_t3 = 0.0

            # 순수 inference-only fps
            if infer_times:
                mean_infer_time = (sum(infer_times) / len(infer_times)) * 1000
            else:
                mean_infer_time = 0.0

            # 평균 peak 메모리
            if infer_peak_mems:
                peak_mem_gb = sum(infer_peak_mems) / len(infer_peak_mems)
            else:
                peak_mem_gb = 0.0

            flops_str = "" if flops_gmac is None else f"{flops_gmac:.4f}"

            # ---- 전체 SUMMARY 행 ----
            csv_w.writerow([
                "SUMMARY",
                "-",                  
                total_npix,
                f"{mean_epe:.6f}",
                f"{mean_d1:.6f}",
                f"{mean_t1:.6f}",
                f"{mean_t3:.6f}",
                f"{mean_infer_time:.2f}",
                f"{peak_mem_gb:.4f}",
                flops_str,
            ])

            # ---- region별 SUMMARY 행 ----
            for region_name, st in region_stats.items():
                r_npix = st["npix"]
                if r_npix > 0:
                    r_epe = st["epe_n"] / r_npix
                    r_d1  = st["d1_n"]  / r_npix
                    r_t1  = st["t1_n"]  / r_npix
                    r_t3  = st["t3_n"]  / r_npix
                else:
                    r_epe = r_d1 = r_t1 = r_t3 = 0.0

                csv_w.writerow([
                    "REGION_SUMMARY",
                    region_name,
                    r_npix,
                    f"{r_epe:.6f}",
                    f"{r_d1:.6f}",
                    f"{r_t1:.6f}",
                    f"{r_t3:.6f}",
                    "", "", "",   
                ])

    if logger:
        logger.info(f"[eval] Completed. Results saved to {csv_path}")
        logger.info(
            f"[inference-only] InferTime: {mean_infer_time:.2f}, PeakMem: {peak_mem_gb:.3f} GB, "
            f"Global EPE: {mean_epe:.4f}, D1: {mean_d1:.4f}, T1: {mean_t1:.4f}, T3: {mean_t3:.4f}, "
            f"FLOPs: {flops_str}"
        )
        for region_name, st in region_stats.items():
            r_npix = st["npix"]
            if r_npix > 0:
                r_epe = st["epe_n"] / r_npix
                r_d1  = st["d1_n"]  / r_npix
                r_t1  = st["t1_n"]  / r_npix
                r_t3  = st["t3_n"]  / r_npix
            else:
                r_epe = r_d1 = r_t1 = r_t3 = 0.0
            logger.info(
                f"[region-summary] {region_name}: "
                f"npix={r_npix}, EPE={r_epe:.4f}, D1={r_d1:.4f}, "
                f"T1={r_t1:.4f}, T3={r_t3:.4f}"
            )

    return all_gts