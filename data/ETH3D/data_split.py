import os
import glob

def generate_eth3d_test_txt(two_view_dir, gt_dir, output_path):

    # 모든 scene 폴더 이름 가져오기
    scene_dirs = sorted([
        name for name in os.listdir(two_view_dir)
        if os.path.isdir(os.path.join(two_view_dir, name))
    ])

    samples = []

    for scene in scene_dirs:
        scene_path = os.path.join(two_view_dir, scene)
        gt_scene_path = os.path.join(gt_dir, scene)

        left_img = os.path.join(scene_path, "im0.png")
        right_img = os.path.join(scene_path, "im1.png")
        disp_gt = os.path.join(gt_scene_path, "disp0GT.pfm")

        # 파일 존재 여부 확인
        if not os.path.exists(left_img):
            print(f"[WARN] left image not found: {left_img}")
            continue
        if not os.path.exists(right_img):
            print(f"[WARN] right image not found: {right_img}")
            continue
        if not os.path.exists(disp_gt):
            print(f"[WARN] disparity not found: {disp_gt}")
            continue

        samples.append((left_img, right_img, disp_gt))

    print(f"Total ETH3D stereo scenes: {len(samples)}")

    # txt 저장
    with open(output_path, "w") as f:
        for left, right, disp in samples:
            f.write(f"{left} {right} {disp}\n")

    print("Saved:", output_path)


# 실행 예시
if __name__ == "__main__":
    two_view_dir = "./two_view_training"
    gt_dir = "./two_view_training_gt"
    output_path = "./eth3d_test.txt"

    generate_eth3d_test_txt(two_view_dir, gt_dir, output_path)