import os, json
import numpy as np
from pathlib import Path


class GlobalStatsManager:
    @staticmethod
    def compute_global_tau(output_root, model_names, 
                           tau_mode="quantile", tau_q=0.9, tau_px=3.0, logger=None):
        """Compute and update _compare/global_tau.json"""

        compare_dir = os.path.join(output_root, "_compare")
        os.makedirs(compare_dir, exist_ok=True)
        global_tau_path = os.path.join(compare_dir, "global_tau.json")

        if tau_mode == "quantile":
            all_errs_tau = []
            # aggregate disparity errors from all models
            for m in model_names:
                cache_dir = os.path.join(output_root, m, "cache_npz")
                if not os.path.isdir(cache_dir):
                    continue
                for npz in Path(cache_dir).glob("*.npz"):
                    try:
                        z = np.load(npz)
                        pred = z["disp_pred"]
                        gt = z["disp_gt"] if ("disp_gt" in z.files and z["disp_gt"].size > 0) else None
                        if gt is not None:
                            err = np.abs(pred - gt)
                        else:
                            err = np.abs(pred - np.median(pred))
                        all_errs_tau.append(err[np.isfinite(err)])
                    except Exception as e:
                        if logger: logger.warning(f"[global_tau] failed to load {npz}: {e}")

            if not all_errs_tau:
                if logger: logger.warning("[global_tau] No valid errors found; skipping.")
                return None

            allv_tau = np.concatenate(all_errs_tau)
            new_tau = float(np.quantile(allv_tau, float(tau_q)))
        else:
            new_tau = float(tau_px)

        # read existing value if present
        old_tau = None
        if os.path.isfile(global_tau_path):
            try:
                with open(global_tau_path, "r", encoding="utf-8") as f:
                    old_meta = json.load(f)
                old_tau = float(old_meta.get("tau", np.nan))
            except Exception:
                old_tau = None

        # update only if different or missing
        if (old_tau is None) or (abs(old_tau - new_tau) > 1e-6):
            meta = dict(
                tau=new_tau, mode=tau_mode, q=tau_q,
                px=tau_px, models=model_names
            )

            with open(global_tau_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            if logger:
                msg = f"[global_tau] updated: {new_tau:.4g}" if old_tau else f"[global_tau] created: {new_tau:.4g}"
                logger.info(msg)
        else:
            if logger:
                logger.info(f"[global_tau] unchanged ({old_tau:.4g})")

        return new_tau

    @staticmethod
    def compute_global_vrange(output_root, model_names,
                              vmax_mode="quantile", vmax_q=0.95, vmin_mode="auto", vmin_q=0.01, logger=None):
        """Compute and update _compare/global_vrange.json"""
        
        compare_dir = os.path.join(output_root, "_compare")
        os.makedirs(compare_dir, exist_ok=True)
        vrange_path = os.path.join(compare_dir, "global_vrange.json")

        all_preds = []
        for m in model_names:
            cache_dir = os.path.join(output_root, m, "cache_npz")
            if not os.path.isdir(cache_dir):
                continue
            for npz in Path(cache_dir).glob("*.npz"):
                try:
                    z = np.load(npz)
                    pred = z["disp_pred"]
                    all_preds.append(pred[np.isfinite(pred)])
                except Exception as e:
                    if logger: logger.warning(f"[global_vrange] failed to load {npz}: {e}")

        if not all_preds:
            if logger: logger.warning("[global_vrange] No valid disparity found; skipping.")
            return None

        allv = np.concatenate(all_preds)

        if vmax_mode == "quantile":
            vmax = float(np.quantile(allv, float(vmax_q)))
        else:
            vmax = float(np.max(allv))

        if vmin_mode == "quantile":
            vmin = float(np.quantile(allv, float(vmin_q)))
        else:
            vmin = float(np.min(allv))

        meta = dict(
            vmin=vmin, vmax=vmax,
            vmax_mode=vmax_mode, vmax_q=vmax_q,
            vmin_mode=vmin_mode, vmin_q=vmin_q,
            models=model_names
        )
        with open(vrange_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if logger:
            logger.info(f"[global_vrange] updated: vmin={vmin:.4g}, vmax={vmax:.4g}")

        return dict(vmin=vmin, vmax=vmax)

    @staticmethod
    def update_all(output_root, model_names, **kwargs):
        """
        Convenience method to compute both tau and vrange together.
        """
        allowed_vrange_keys = {"vmax_mode", "vmax_q", "vmin_q", "logger"}
        vr_kwargs = {k: v for k, v in kwargs.items() if k in allowed_vrange_keys}

        allowed_tau_keys = {"tau_mode", "tau_q", "tau_px", "logger"}
        tau_kwargs = {k: v for k, v in kwargs.items() if k in allowed_tau_keys}

        vr = GlobalStatsManager.compute_global_vrange(output_root, model_names, **vr_kwargs)
        tau = GlobalStatsManager.compute_global_tau(output_root, model_names, **tau_kwargs)

        return dict(tau=tau, vrange=vr)