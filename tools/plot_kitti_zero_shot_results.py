import os
import matplotlib.pyplot as plt
import cv2

# ---------------------------------------------------------
# 이미지 경로 (row x col 구조)
# 예: 4행 x 4열
# ---------------------------------------------------------
parent_dir = 'outputs/KITTI'

image_grid = [
    ["kitti_row1_left.png", "kitti_row1_lightstereo.png", "kitti_row1_banet.png", "kitti_row1_ours.png"],
    ["kitti_row2_left.png", "kitti_row2_lightstereo.png", "kitti_row2_banet.png", "kitti_row2_ours.png"],
    ["kitti_row3_left.png", "kitti_row3_lightstereo.png", "kitti_row3_banet.png", "kitti_row3_ours.png"],
    ["kitti_row4_left.png", "kitti_row4_lightstereo.png", "kitti_row4_banet.png", "kitti_row4_ours.png"],
]

rows = len(image_grid)
cols = len(image_grid[0])

# ---------------------------------------------------------
# 패딩 잘라내기
#
# KITTI 원본은 375x1242 근처인데 모델 입력을 맞추려고 384x1248 로 edge 패딩된다.
# 그래서 **위쪽 몇 행이 맨 윗줄의 복사**라 늘어난 것처럼 보인다. 실측(left 이미지
# 기준 맨 윗줄과 동일한 행 수): row1 9행, row2 9행, row3 11행, row4 9행.
# 오른쪽도 같은 이유로 6~15열이 복제돼 있다.
#
# 행마다 다르게 자르면 칸 높이가 달라져 격자가 어긋나므로 **일괄로** 자른다.
# 원본 파일은 건드리지 않고 그릴 때만 잘라낸다.
# ---------------------------------------------------------
CROP_TOP = 12        # 위에서 잘라낼 픽셀 (실측 최대 11 을 덮는 값)
CROP_RIGHT = 16      # 오른쪽 패딩 (실측 최대 15 를 덮는 값)

# ---------------------------------------------------------
# figure 크기 계산
#
# imshow 는 기본이 aspect='equal' 이라 이미지 비율을 지킨다. 그래서 figsize 의
# 비율이 격자 비율과 다르면 **칸 안에 여백**이 생긴다 (이미지가 작아질 뿐이다).
# KITTI 는 1248x384 (3.25:1) 로 매우 넓어서, figsize 를 잘못 잡으면 칸 위아래에
# 여백이 크게 남는다.
#
# 따라서 figsize 를 이미지 실제 비율에서 역산한다:
#   axes 하나의 높이 = 폭 / 이미지비율
#   figure 폭  = cols*폭   + (cols-1)*wspace*폭
#   figure 높이 = rows*높이 + (rows-1)*hspace*높이
# ---------------------------------------------------------
FIG_WIDTH = 10.0     # 논문 지면 폭에 맞춰 이것만 정하면 된다 (2단이면 ~7)
wspace = 0.03        # 가로 간격: axes 폭에 대한 비율

# 대표 이미지에서 비율을 얻는다 (칸마다 크기가 조금씩 달라도 첫 장 기준)
_probe = None
for _r in image_grid:
    for _f in _r:
        _p = os.path.join(parent_dir, _f)
        if os.path.exists(_p):
            _im = cv2.imread(_p)
            if _im is not None:
                # 잘라낸 뒤 크기로 비율을 잡아야 칸에 여백이 안 생긴다
                _probe = (_im.shape[1] - CROP_RIGHT, _im.shape[0] - CROP_TOP)   # (W, H)
                break
    if _probe:
        break
img_w, img_h = _probe if _probe else (1248, 384)
img_aspect = img_w / img_h

# 가로·세로 간격이 **눈에 같아 보이게** hspace 를 비율만큼 키운다.
# wspace 는 axes 폭의 비율, hspace 는 axes 높이의 비율이라 같은 값을 주면
# 가로로 긴 이미지에서는 세로 간격이 훨씬 좁아 보인다.
hspace = wspace * img_aspect

ax_w = FIG_WIDTH / (cols + (cols - 1) * wspace)
ax_h = ax_w / img_aspect
FIG_HEIGHT = ax_h * (rows + (rows - 1) * hspace)

print(f"crop: 위 {CROP_TOP}px, 오른쪽 {CROP_RIGHT}px")
print(f"이미지 {img_w}x{img_h} (잘라낸 뒤, aspect {img_aspect:.3f}) | 격자 {rows}x{cols}")
print(f"figsize = ({FIG_WIDTH:.2f}, {FIG_HEIGHT:.2f})  wspace={wspace}  hspace={hspace:.4f}")

# ---------------------------------------------------------
# figure 생성
# ---------------------------------------------------------
fig, axes = plt.subplots(rows, cols, figsize=(FIG_WIDTH, FIG_HEIGHT))

# ---------------------------------------------------------
# 이미지 삽입
# ---------------------------------------------------------
for r in range(rows):
    for c in range(cols):
        img_path = parent_dir + '/' + image_grid[r][c]

        if not os.path.exists(img_path):
            print(f"[WARN] {img_path} not found")
            axes[r, c].axis("off")
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h0, w0 = img.shape[:2]
        img = img[CROP_TOP:, :w0 - CROP_RIGHT if CROP_RIGHT else w0]

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
out_path = "outputs/kitti_zeroshot_grid_for_paper.png"
plt.savefig(out_path, dpi=500, bbox_inches="tight", pad_inches=0)
print(f"Saved to {out_path}")

plt.show()