import os
import pandas as pd
import matplotlib.pyplot as plt

csv_map = {
    "LightGEV": "outputs/sceneflow_eval/lightgev_pretrained_sceneflow_full_final2/results.csv",
    "LightStereo": "outputs/sceneflow_eval/lightstereo_sota_pretrained_sceneflow/results.csv",
    "MobileStereoNet": "outputs/sceneflow_eval/msnet_pretrained_sceneflow/results.csv",
    "BANet": "outputs/sceneflow_eval/banet_pretrained_sceneflow/results.csv",
}

# ---------------------------------------------------------
# 모델별 색상만 구분
# ---------------------------------------------------------
model_colors = {
    "LightGEV": "red",
    "LightStereo": "blue",
    "MobileStereoNet": "green",
    "BANet": "yellow",
}

plt.figure(figsize=(7, 5))
model_handles = {}

for model_name, csv_path in csv_map.items():
    if not os.path.exists(csv_path):
        print(f"[WARN] File not found: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    summary_mask = (df["name"] == "SUMMARY")
    if not summary_mask.any():
        print(f"[WARN] No SUMMARY row in {csv_path}")
        continue

    flops_val = df.loc[summary_mask, "flops_gmac"].iloc[0]
    runtime_val = df.loc[summary_mask, "infer_time"].iloc[0]

    color = model_colors.get(model_name, "black")

    # 모든 모델 marker = 'o'
    plt.scatter(
        flops_val,
        runtime_val,
        marker="o",
        color=color,
        edgecolors="black",
        linewidths=0.6,
        s=90,
    )

    if model_name not in model_handles:
        model_handles[model_name] = plt.Line2D(
            [], [], marker="o", linestyle="None",
            markeredgecolor="black",
            markerfacecolor=color,
            label=model_name,
        )

plt.xlabel("FLOPs (G)")
plt.ylabel("Runtime (ms)")
plt.grid(True, linestyle="--", linewidth=0.5)

plt.legend(handles=list(model_handles.values()), title="Models")

plt.tight_layout()
out_path = "outputs/flops_vs_runtime_2D_models.png"
plt.savefig(out_path, dpi=300)
print(f"Saved figure to {out_path}")