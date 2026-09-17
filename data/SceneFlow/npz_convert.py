import os
import re
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


# ----------------------------
# PFM Reader
# ----------------------------
def read_pfm(path: str) -> np.ndarray:
    with open(path, "rb") as file:
        header = file.readline().rstrip().decode("utf-8")
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise ValueError(f"Not a PFM file: {path}")

        dim_line = file.readline().decode("utf-8")
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", dim_line)
        if not dim_match:
            raise ValueError(f"Malformed PFM header (dim): {path}")
        width, height = map(int, dim_match.groups())

        scale = float(file.readline().rstrip().decode("utf-8"))
        if scale < 0:
            endian = "<"
            scale = -scale
        else:
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data.astype(np.float32)


# ----------------------------
# Image loader
# ----------------------------
def load_png(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32)


# ----------------------------
# Key normalization for pairing
# ----------------------------
def normalize_disp_key(disp_path: str) -> str:
    """
    cleanpass/finalpass에서 같은 disparity를 확실히 같은 것으로 매칭하기 위한 key.
    기본은 abspath + normpath.
    (상대경로/절대경로 섞임, ./ 포함 등에도 안전)
    """
    return os.path.normpath(os.path.abspath(disp_path))


# ----------------------------
# Read txt -> dict(key -> (L, R, D))
# ----------------------------
def read_txt_as_map(txt_path: str) -> dict:
    m = {}
    with open(txt_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Malformed line {i} in {txt_path}: {line}")

            lp, rp, dp = parts
            key = normalize_disp_key(dp)

            if key in m:
                raise ValueError(
                    f"Duplicate disparity key in {txt_path}\n"
                    f"key={key}\n"
                    f"prev={m[key]}\n"
                    f"new=({lp}, {rp}, {dp})"
                )
            m[key] = (lp, rp, dp)
    return m


# ----------------------------
# Build paired list from clean/final txt
# ----------------------------
def build_paired_list(clean_txt: str, final_txt: str):
    clean_map = read_txt_as_map(clean_txt)
    final_map = read_txt_as_map(final_txt)

    clean_keys = set(clean_map.keys())
    final_keys = set(final_map.keys())

    common = sorted(list(clean_keys & final_keys))
    only_clean = sorted(list(clean_keys - final_keys))
    only_final = sorted(list(final_keys - clean_keys))

    if only_clean:
        print(f"[WARN] Only in clean ({len(only_clean)}). Example key: {only_clean[0]}")
    if only_final:
        print(f"[WARN] Only in final ({len(only_final)}). Example key: {only_final[0]}")

    paired = []
    for k in common:
        lc, rc, dc = clean_map[k]
        lf, rf, df = final_map[k]

        # 실제 저장엔 clean/final 각각의 L/R + disparity 1개만
        # disparity 경로는 clean 쪽을 우선 사용 (normalize key는 동일하므로 파일은 같아야 함)
        paired.append((lc, rc, lf, rf, dc))

    print(f"[OK] Paired samples: {len(paired)}")
    return paired


# ----------------------------
# One paired sample -> npz
# ----------------------------
def process_one_paired_sample(idx_and_item, save_root: str, split_name: str, compress: bool = True):
    idx, item = idx_and_item
    left_c, right_c, left_f, right_f, disp_path = item

    split_dir = os.path.join(save_root, split_name)
    os.makedirs(split_dir, exist_ok=True)
    npz_path = os.path.join(split_dir, f"{idx:06d}.npz")

    if os.path.exists(npz_path):
        return False

    left_clean = load_png(left_c)
    right_clean = load_png(right_c)
    left_final = load_png(left_f)
    right_final = load_png(right_f)
    disp = read_pfm(disp_path)  # disparity는 1번만 저장

    if compress:
        np.savez_compressed(
            npz_path,
            left_clean=left_clean,
            right_clean=right_clean,
            left_final=left_final,
            right_final=right_final,
            disp=disp,
        )
    else:
        np.savez(
            npz_path,
            left_clean=left_clean,
            right_clean=right_clean,
            left_final=left_final,
            right_final=right_final,
            disp=disp,
        )

    return True


# ----------------------------
# Save split with multiprocessing
# ----------------------------
def save_npz_split_paired_mp(clean_txt: str,
                            final_txt: str,
                            save_root: str,
                            split_name: str,
                            num_workers: int = None,
                            compress: bool = True):
    paired = build_paired_list(clean_txt, final_txt)
    n_samples = len(paired)
    print(f"\n[{split_name.upper()}] {n_samples} paired samples")
    print(f"clean_txt: {clean_txt}")
    print(f"final_txt: {final_txt}")

    if num_workers is None:
        num_workers = min(8, max(1, cpu_count() // 2))

    print(f"Using {num_workers} workers...")

    idx_item_list = list(enumerate(paired))
    worker_fn = partial(
        process_one_paired_sample,
        save_root=save_root,
        split_name=split_name,
        compress=compress,
    )

    with Pool(processes=num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(worker_fn, idx_item_list, chunksize=32),
            total=n_samples,
            desc=f"Processing {split_name}",
        ):
            pass

    print(f"✅ Saved paired NPZ files in: {os.path.join(save_root, split_name)}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # cleanpass / finalpass txt 경로
    train_clean = "./flyingthings3d_cleanpass_train.txt"
    train_final = "./flyingthings3d_finalpass_train.txt"

    val_clean = "./flyingthings3d_cleanpass_val.txt"
    val_final = "./flyingthings3d_finalpass_val.txt"

    test_clean = "./flyingthings3d_cleanpass_test.txt"
    test_final = "./flyingthings3d_finalpass_test.txt"

    # npz 저장 루트
    save_root = "./FlyingThings3D_npz"

    # worker 수 (None이면 자동)
    num_workers = 10

    # True: 용량↓, False: 속도↑
    compress = True

    save_npz_split_paired_mp(train_clean, train_final, save_root, "train", num_workers=num_workers, compress=compress)
    save_npz_split_paired_mp(val_clean,   val_final,   save_root, "val",   num_workers=num_workers, compress=compress)
    save_npz_split_paired_mp(test_clean,  test_final,  save_root, "test",  num_workers=num_workers, compress=compress)

    print("\n🎉 All paired NPZ files saved successfully!")