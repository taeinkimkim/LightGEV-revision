import os
from tqdm import tqdm

def write_list(npz_dir, split, save_path):
    """
    npz_dir : './FlyingThings3D_npz'
    split   : 'train', 'val', 'test'
    save_path : txt 파일 경로
    """
    target_dir = os.path.join(npz_dir, split)
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"{target_dir} not found")

    file_list = sorted(os.listdir(target_dir))

    with open(save_path, "w") as f:
        for fname in file_list:
            if fname.endswith(".npz"):
                rel_path = os.path.join(npz_dir, split, fname)
                f.write(rel_path + "\n")

    print(f"✔ Saved {save_path}  ({len(file_list)} samples)")


if __name__ == "__main__":
    npz_root = "./FlyingThings3D_npz"

    write_list(npz_root, "train", "./sceneflow_train_npz.txt")
    write_list(npz_root, "val",   "./sceneflow_val_npz.txt")
    write_list(npz_root, "test",  "./sceneflow_test_npz.txt")

    print("\n🎉 All txt files created successfully!")