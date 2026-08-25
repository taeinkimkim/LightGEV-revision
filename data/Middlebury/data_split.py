import os
import glob

def generate_middlebury_test_txt(root_dir, output_path):

    # perfect/*-perfect 전체 스캔
    scene_dirs = sorted(glob.glob(os.path.join(root_dir, "perfect", "*-perfect")))

    samples = []

    for scene in scene_dirs:
        left_img = os.path.join(scene, "im0.png")
        right_img = os.path.join(scene, "im1.png")   # 기본 im1 사용
        disp = os.path.join(scene, "disp0.pfm")

        if not (os.path.exists(left_img) and os.path.exists(right_img) and os.path.exists(disp)):
            print(f"[WARN] Missing file(s) in: {scene}")
            continue

        samples.append((left_img, right_img, disp))

    print(f"Total test scenes: {len(samples)}")

    # 저장
    with open(output_path, "w") as f:
        for l, r, d in samples:
            f.write(f"{l} {r} {d}\n")

    print("Saved:", output_path)


# 실행 예시
if __name__ == "__main__":
    root = "./2014"                        # Middlebury 2014 루트
    output = "./middlebury2014_test.txt"   # 출력 파일 경로
    generate_middlebury_test_txt(root, output)