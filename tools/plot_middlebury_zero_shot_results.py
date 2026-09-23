import os
import matplotlib.pyplot as plt
import cv2

# ---------------------------------------------------------
# 이미지 경로 (row x col 구조)
# 예: 4행 x 4열
# ---------------------------------------------------------
parent_dir = 'outputs/Middlebury'

image_grid = [
    ["middlebury_row1_left.png", "middlebury_row1_lightstereo.png", "middlebury_row1_banet.png", "middlebury_row1_ours.png"],
    ["middlebury_row2_left.png", "middlebury_row2_lightstereo.png", "middlebury_row2_banet.png", "middlebury_row2_ours.png"],
    ["middlebury_row3_left.png", "middlebury_row3_lightstereo.png", "middlebury_row3_banet.png", "middlebury_row3_ours.png"],
    ["middlebury_row4_left.png", "middlebury_row4_lightstereo.png", "middlebury_row4_banet.png", "middlebury_row4_ours.png"],
]

rows = len(image_grid)
cols = len(image_grid[0])

# ---------------------------------------------------------
# 패딩 잘라내기 (행마다 다르다)
#
# Middlebury 평가 파이프라인: 0.25 배 축소 -> RightTopPad([497, 700])
# -> InputPadder(DIVIS_BY=32, MODE=sintel). 장면마다 원본 크기가 달라서 패딩량도
# 행마다 다르다. 실제로 704x512 / 736x512 / 768x512 세 종류가 섞여 있다.
#
# 그래서 각 행의 left 이미지에서 **맨 가장자리와 완전히 동일한 행/열**을 세어
# 패딩량을 추정하고, 그 행의 네 이미지에 같은 값을 적용한다.
#
# 상한을 두는 이유: 균일한 영역(하늘·벽)이 있으면 오탐이 난다. 실제로 row2 의
# 오른쪽이 45 로 잡히는데 계산상 3~4 가 맞다.
#   InputPadder 가 왼쪽/아래에 넣는 양은 각각 15/16 이 최대다 (DIVIS_BY=32).
#   위/오른쪽은 RightTopPad 가 더해져 더 클 수 있다.
# ---------------------------------------------------------
PAD_CAP = {"top": 40, "bottom": 20, "left": 20, "right": 20}

# 자동 추정이 어긋나면 여기서 행별로 직접 지정한다 (위, 아래, 왼, 오른)
# row2(인덱스 1) 는 오른쪽이 균일해서 19 로 오탐된다. 같은 704x512 인 row1 이 3 이고
# 계산상으로도 InputPadder 분 2~3 이 맞으므로 직접 지정한다.
PAD_OVERRIDE = {1: (7, 8, 2, 3)}


def _dup_run(im, side, cap):
    """맨 가장자리 행/열과 **완전히 동일한** 줄이 몇 개 이어지는지 (edge 패딩 탐지)."""
    import numpy as _np
    n = 0
    if side == "top":
        ref = im[0]
        for r in range(min(cap, im.shape[0])):
            if _np.array_equal(im[r], ref): n += 1
            else: break
    elif side == "bottom":
        ref = im[-1]
        for r in range(im.shape[0] - 1, max(-1, im.shape[0] - cap - 1), -1):
            if _np.array_equal(im[r], ref): n += 1
            else: break
    elif side == "left":
        ref = im[:, 0]
        for c in range(min(cap, im.shape[1])):
            if _np.array_equal(im[:, c], ref): n += 1
            else: break
    else:
        ref = im[:, -1]
        for c in range(im.shape[1] - 1, max(-1, im.shape[1] - cap - 1), -1):
            if _np.array_equal(im[:, c], ref): n += 1
            else: break
    return max(0, n - 1)   # 기준 줄 자신은 패딩이 아니다


def detect_pad(row_idx):
    if row_idx in PAD_OVERRIDE:
        return PAD_OVERRIDE[row_idx]
    ref_path = os.path.join(parent_dir, image_grid[row_idx][0])   # 그 행의 left 이미지
    im = cv2.imread(ref_path)
    if im is None:
        return (0, 0, 0, 0)
    return tuple(_dup_run(im, s, PAD_CAP[s]) for s in ("top", "bottom", "left", "right"))


ROW_PAD = {r: detect_pad(r) for r in range(rows)}
for r in range(rows):
    t, b, l, rt = ROW_PAD[r]
    print(f"row{r+1} 패딩 추정: 위 {t}, 아래 {b}, 왼 {l}, 오른 {rt}")

# ---------------------------------------------------------
# figure 크기
#
# 행마다 원본 크기가 달라서 모든 칸을 같은 비율로 맞춘 뒤 같은 크기로 리사이즈한다.
# imshow 가 aspect='equal' 이라 figsize 비율이 격자 비율과 달라지면 칸 안에 여백이
# 생기므로, TARGET_RATIO 에서 figsize 를 역산한다.
#
# 기존 코드는 center_crop(1.5) 뒤에 704x512(=1.375) 로 리사이즈해서 가로로 9%
# 눌려 있었다. 아래처럼 crop 비율과 리사이즈 비율을 같게 맞춘다.
# ---------------------------------------------------------
TARGET_RATIO = 1.5                       # 잘라낸 뒤 맞출 가로/세로 비율
TARGET_H = 512
TARGET_W = int(round(TARGET_H * TARGET_RATIO))    # 768

FIG_WIDTH = 8.0      # 논문 지면 폭
wspace = 0.03        # 가로 간격: axes 폭 대비 비율

# wspace 는 axes '폭', hspace 는 axes '높이' 기준이라 같은 값을 주면 세로 간격이
# 좁아 보인다. 비율만큼 키워 눈에 같게 만든다.
hspace = wspace * TARGET_RATIO

ax_w = FIG_WIDTH / (cols + (cols - 1) * wspace)
ax_h = ax_w / TARGET_RATIO
FIG_HEIGHT = ax_h * (rows + (rows - 1) * hspace)

print(f"타깃 {TARGET_W}x{TARGET_H} (aspect {TARGET_RATIO}) | 격자 {rows}x{cols}")
print(f"figsize = ({FIG_WIDTH:.2f}, {FIG_HEIGHT:.2f})  wspace={wspace}  hspace={hspace:.4f}")

fig, axes = plt.subplots(rows, cols, figsize=(FIG_WIDTH, FIG_HEIGHT))

# ---------------------------------------------------------
# 이미지 삽입
# ---------------------------------------------------------
def center_crop(img, target_ratio=2.0):  # 예: 512x256 → ratio=2
    '''
    중앙 기준 crop해서 비율 맞춘 후 resize
    '''
    h, w = img.shape[:2]
    current_ratio = w / h

    if current_ratio > target_ratio:
        # 너무 넓음 → 좌우 crop
        new_w = int(h * target_ratio)
        start = (w - new_w) // 2
        img = img[:, start:start+new_w]
    else:
        # 너무 높음 → 위아래 crop
        new_h = int(w / target_ratio)
        start = (h - new_h) // 2
        img = img[start:start+new_h, :]

    return img

for r in range(rows):
    for c in range(cols):
        img_path = parent_dir + '/' + image_grid[r][c]

        if not os.path.exists(img_path):
            print(f"[WARN] {img_path} not found")
            axes[r, c].axis("off")
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1) 그 행의 패딩 제거  2) 공통 비율로 center crop  3) 공통 크기로 리사이즈
        pt, pb, pl, pr = ROW_PAD[r]
        h0, w0 = img.shape[:2]
        img = img[pt:h0 - pb if pb else h0, pl:w0 - pr if pr else w0]

        img = center_crop(img, target_ratio=TARGET_RATIO)
        img = cv2.resize(img, (TARGET_W, TARGET_H))

        axes[r, c].imshow(img)
        axes[r, c].axis("off")

# ---------------------------------------------------------
# 간격 적용 (값은 위에서 계산했다)
# ---------------------------------------------------------
plt.subplots_adjust(
    left=0,
    right=1,
    top=1,
    bottom=0,
    wspace=wspace,
    hspace=hspace
)

# ---------------------------------------------------------
# 저장
# ---------------------------------------------------------
out_path = "outputs/middlebury_zeroshot_grid_for_paper.png"
plt.savefig(out_path, dpi=500, bbox_inches="tight", pad_inches=0)
print(f"Saved to {out_path}")

plt.show()