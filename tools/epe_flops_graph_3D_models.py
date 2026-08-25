import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. 각 모델 × 각 데이터셋의 results.csv 경로
# ---------------------------------------------------------
# 3D Models
csv_map = {
    "LightGEV": {
        "Middlebury": "outputs/middlebury_eval/lightgev_pretrained_sceneflow_final2/results.csv",
        "ETH3D":      "outputs/eth3d_eval/lightgev_pretrained_sceneflow_final2/results.csv",
        "KITTI":      "outputs/kitti_eval/lightgev_pretrained_sceneflow_final2/results.csv",
    },
    "IGEV++": {
        "Middlebury": "outputs/middlebury_eval/igev_pp_pretrained_sceneflow/results.csv",
        "ETH3D":      "outputs/eth3d_eval/igev_pp_pretrained_sceneflow/results.csv",
        "KITTI":      "outputs/kitti_eval/igev_pp_pretrained_sceneflow/results.csv",
    },
    "ACVNet": {
        "Middlebury": "outputs/middlebury_eval/acvnet_pretrained_sceneflow/results.csv",
        "ETH3D":      "outputs/eth3d_eval/acvnet_pretrained_sceneflow/results.csv",
        "KITTI":      "outputs/kitti_eval/acvnet_pretrained_sceneflow/results.csv",
    },
    "GwcNet": {
        "Middlebury": "outputs/middlebury_eval/gwcnet_pretrained_sceneflow/results.csv",
        "ETH3D":      "outputs/eth3d_eval/gwcnet_pretrained_sceneflow/results.csv",
        "KITTI":      "outputs/kitti_eval/gwcnet_pretrained_sceneflow/results.csv",
    },
}

# ---------------------------------------------------------
# 2. 마커/색 설정
#    - 모델마다: 네모, 동그라미, 세모, 마름모, 별
#    - 데이터셋마다: Middlebury=빨강, ETH3D=초록, KITTI=파랑
# ---------------------------------------------------------
marker_list = ["*", "D", "o", "^"]  # square, circle, triangle, diamond, star

dataset_colors = {
    "Middlebury": "red",
    "ETH3D": "green",
    "KITTI": "blue",
}

# ---------------------------------------------------------
# 3. 플롯 준비
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

model_handles = {}
dataset_handles = {}

for m_idx, (model_name, ds_paths) in enumerate(csv_map.items()):
    marker = marker_list[m_idx % len(marker_list)]

    # --- 별일 때만 점 크기 키우기 ---
    if marker == "*":
        marker_size = 100   # 일반보다 크게
    else:
        marker_size = 70    # 기본 정도
    # --------------------------------

    for dataset_name, csv_path in ds_paths.items():
        if not os.path.exists(csv_path):
            print(f"[WARN] File not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # --------- EPE 추출: name=REGION_SUMMARY & metric=global ----------
        epe_mask = (df["name"] == "REGION_SUMMARY") & (df["metric"] == "global")
        if not epe_mask.any():
            print(f"[WARN] No REGION_SUMMARY/global row in {csv_path}")
            continue
        epe_val = df.loc[epe_mask, "epe"].iloc[0]

        # --------- FLOPs 추출: name=SUMMARY ----------
        flops_mask = (df["name"] == "SUMMARY")
        if not flops_mask.any():
            print(f"[WARN] No SUMMARY row in {csv_path}")
            continue
        flops_val = df.loc[flops_mask, "flops_gmac"].iloc[0]

        color = dataset_colors.get(dataset_name, "black")

        # 점 찍기
        plt.scatter(
            flops_val,
            epe_val,
            marker=marker,
            color=color,
            edgecolors="black",
            linewidths=0.5,
            s=marker_size,          # ★ 마커 크기 적용
        )

        # 모델 범례용 핸들
        if model_name not in model_handles:
            # ★ 별 모양만 더 크게
            if marker == "*":
                legend_marker_size = 8.5   # 일반보다 큼
            else:
                legend_marker_size = 6    # 기본 크기

            model_handles[model_name] = plt.Line2D(
                [], [], marker=marker, linestyle="None",
                markeredgecolor="black", markerfacecolor="white",
                label=model_name,
                markersize=legend_marker_size,  # ★ 범례에서만 마커 크기 차별 적용
            )

        # 데이터셋 범례용 핸들
        if dataset_name not in dataset_handles:
            dataset_handles[dataset_name] = plt.Line2D(
                [], [], marker="o", linestyle="None",
                markeredgecolor="black", markerfacecolor=color,
                label=dataset_name,
            )

# ---------------------------------------------------------
# 4. 축 / 타이틀 / 범례
# ---------------------------------------------------------
plt.xlabel("FLOPs (G)")
plt.ylabel("Global EPE (px)")
plt.title("FLOPs vs Global EPE (Middlebury / ETH3D / KITTI)")

plt.grid(True, linestyle="--", linewidth=0.5)

# 모델 범례 (오른쪽 위)
legend_models = plt.legend(
    handles=list(model_handles.values()),
    title="Models",
    loc="upper right",
    bbox_to_anchor=(1.0, 1.0),   # 오른쪽 끝
)
plt.gca().add_artist(legend_models)

# 데이터셋 범례 (모델 범례 왼쪽)
legend_datasets = plt.legend(
    handles=list(dataset_handles.values()),
    title="Datasets",
    loc="upper right",
    bbox_to_anchor=(0.835, 1.0),  # 살짝 왼쪽으로 이동
)
plt.gca().add_artist(legend_datasets)

plt.tight_layout()

out_path = "outputs/flops_vs_epe_3D_models.png"
plt.savefig(out_path, dpi=300)
print(f"Saved figure to {out_path}")