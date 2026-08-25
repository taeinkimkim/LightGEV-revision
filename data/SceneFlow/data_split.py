import os
import random

# 루트 경로 설정
root_dir = "FlyingThings3D"
frames_root = os.path.join(root_dir, "frames_finalpass")
disp_root = os.path.join(root_dir, "disparity")

# 결과 저장 폴더
save_dir = "./"

# 확장자
img_ext = ".png"
disp_ext = ".pfm"

def collect_triplets(split_name):
    """
    split_name: 'TRAIN' or 'TEST'
    """
    triplets = []
    frame_split_dir = os.path.join(frames_root, split_name)
    disp_split_dir = os.path.join(disp_root, split_name)

    # A, B, C 폴더 순회
    for subset in ["A", "B", "C"]:
        frame_subset_dir = os.path.join(frame_split_dir, subset)
        disp_subset_dir = os.path.join(disp_split_dir, subset)

        if not os.path.isdir(frame_subset_dir):
            continue

        # 시퀀스 (0000, 0001, ...)
        for seq in os.listdir(frame_subset_dir):
            seq_dir = os.path.join(frame_subset_dir, seq)
            disp_seq_dir = os.path.join(disp_subset_dir, seq)

            left_dir = os.path.join(seq_dir, "left")
            right_dir = os.path.join(seq_dir, "right")
            disp_left_dir = os.path.join(disp_seq_dir, "left")

            if not (os.path.isdir(left_dir) and os.path.isdir(right_dir) and os.path.isdir(disp_left_dir)):
                continue

            left_imgs = sorted([f for f in os.listdir(left_dir) if f.endswith(img_ext)])
            for name in left_imgs:
                stem = name.replace(img_ext, "")
                left_path = os.path.join(left_dir, name)
                right_path = os.path.join(right_dir, name)
                disp_path = os.path.join(disp_left_dir, f"{stem}{disp_ext}")

                if os.path.exists(right_path) and os.path.exists(disp_path):
                    triplets.append((left_path, right_path, disp_path))

    print(f"[{split_name}] {len(triplets)} samples found")
    return triplets


def save_txt(file_path, triplets):
    with open(file_path, "w") as f:
        for left, right, disp in triplets:
            f.write(f"{left} {right} {disp}\n")


# ----------------------------
# 데이터 수집
# ----------------------------
train_triplets = collect_triplets("TRAIN")
test_triplets = collect_triplets("TEST")

# ----------------------------
# TRAIN → TRAIN + VAL 분할
# ----------------------------
random.seed(42)
random.shuffle(train_triplets)

train_ratio = 0.85
n = len(train_triplets)
train_end = int(n * train_ratio)

train_split = train_triplets[:train_end]
val_split = train_triplets[train_end:]

# ----------------------------
# 저장
# ----------------------------
save_txt(os.path.join(save_dir, "sceneflow_finalpass_train.txt"), train_split)
save_txt(os.path.join(save_dir, "sceneflow_finalpass_val.txt"), val_split)
save_txt(os.path.join(save_dir, "sceneflow_finalpass_test.txt"), test_triplets)

print("✅ Done!")
print(f"Train: {len(train_split)} | Val: {len(val_split)} | Test: {len(test_triplets)}")
print(f"All files saved in: {save_dir}")