import os
import re
import sys
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


# ----------------------------
# PFM Reader
# ----------------------------
def read_pfm(path: str):
    with open(path, 'rb') as file:
        header = file.readline().rstrip()
        header = header.decode('utf-8')
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception(f'Not a PFM file: {path}')

        dim_line = file.readline().decode('utf-8')
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', dim_line)
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception(f'Malformed PFM header: {path}')

        scale = float(file.readline().rstrip().decode('utf-8'))
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        # color는 SceneFlow disp에서는 False라 (H,W)일 것
        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data.astype(np.float32)


# ----------------------------
# 이미지 로더
# ----------------------------
def load_png(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32)


# ----------------------------
# 한 샘플 처리 함수 (멀티프로세스에서 실행)
# ----------------------------
def process_one_sample(idx_and_line, save_root: str, split_name: str, compress: bool = True):
    idx, line = idx_and_line

    line = line.strip()
    if not line:
        return False

    parts = line.split()
    if len(parts) != 3:
        raise ValueError(f"Line {idx} malformed: {line}")

    left_path, right_path, disp_path = parts

    # 출력 npz 경로
    split_dir = os.path.join(save_root, split_name)
    os.makedirs(split_dir, exist_ok=True)
    npz_path = os.path.join(split_dir, f"{idx:06d}.npz")

    # 이미 만들어져 있으면 스킵
    if os.path.exists(npz_path):
        return False

    # ---- Load 데이터 ----
    left = load_png(left_path)
    right = load_png(right_path)
    disp = read_pfm(disp_path)

    # ---- Save NPZ ----
    if compress:
        np.savez_compressed(
            npz_path,
            left=left,
            right=right,
            disp=disp,
        )
    else:
        np.savez(
            npz_path,
            left=left,
            right=right,
            disp=disp,
        )

    return True


# ----------------------------
# 하나의 split(txt) 처리
# ----------------------------
def save_npz_split_mp(txt_path: str,
                      save_root: str,
                      split_name: str,
                      num_workers: int = None,
                      compress: bool = True):
    """
    txt_path   : sceneflow_finalpass_train.txt 등
    save_root  : npz 저장 루트 (예: './FlyingThings3D_npz')
    split_name : 'train' / 'val' / 'test'
    num_workers: 멀티프로세스 worker 수 (None이면 CPU 기준 자동 결정)
    compress   : np.savez_compressed 사용 여부 (True면 용량↓, 속도는 약간↓)
    """
    with open(txt_path, "r") as f:
        lines = f.readlines()

    n_samples = len(lines)
    print(f"\n[{split_name.upper()}] {n_samples} samples from {txt_path}")

    if num_workers is None:
        num_workers = min(8, max(1, cpu_count() // 2))

    print(f"Using {num_workers} workers...")

    # (idx, line) 튜플 리스트 생성
    idx_line_list = list(enumerate(lines))

    worker_fn = partial(
        process_one_sample,
        save_root=save_root,
        split_name=split_name,
        compress=compress,
    )

    # 멀티프로세싱 Pool
    with Pool(processes=num_workers) as pool:
        # imap_unordered 로 순서 상관 없이 처리 (더 빠름)
        for _ in tqdm(
            pool.imap_unordered(worker_fn, idx_line_list, chunksize=32),
            total=n_samples,
            desc=f"Processing {split_name}",
        ):
            pass

    print(f"✅ Saved NPZ files in: {os.path.join(save_root, split_name)}")


# ----------------------------
# 메인 실행
# ----------------------------
if __name__ == "__main__":
    # txt 파일 경로
    train_txt = "./sceneflow_finalpass_train.txt"
    val_txt   = "./sceneflow_finalpass_val.txt"
    test_txt  = "./sceneflow_finalpass_test.txt"

    # npz 저장 루트
    save_root = "./FlyingThings3D_npz"

    # worker 수
    num_workers = 8     # None이면 자동 결정

    # 압축 여부 (True: 용량↓, False: 속도↑)
    compress = True

    save_npz_split_mp(train_txt, save_root, "train", num_workers=num_workers, compress=compress)
    save_npz_split_mp(val_txt,   save_root, "val",   num_workers=num_workers, compress=compress)
    save_npz_split_mp(test_txt,  save_root, "test",  num_workers=num_workers, compress=compress)

    print("\n🎉 All NPZ files saved successfully (multiprocessing)!")