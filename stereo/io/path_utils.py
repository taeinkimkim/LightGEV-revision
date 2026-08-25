import os

def ensure_dirs(output_root: str):
    os.makedirs(output_root, exist_ok=True)
    d = {
        "cache_npz": os.path.join(output_root, "cache_npz"),
        "left_bgr": os.path.join(output_root, "left_bgr"),
        "disp_vis_global": os.path.join(output_root, "disp_vis_global"),
        "error_vis_global": os.path.join(output_root, "error_vis_global"),
        "region_overlay": os.path.join(output_root, "region_overlay"),
        "region_error_vis_global": os.path.join(output_root, "region_error_vis_global"),
    }
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d