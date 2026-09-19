"""Middlebury 해상도 sweep 결과를 하나의 표로 모은다.

`V_experiments/evaluation/eval_pretrained_sceneflow_by_middlebury_lightgev_modify_4_scale.sh`
가 만든 run 디렉토리들을 읽어 모델 x 배율 표를 찍는다. 각 run 의 results.csv 에서
REGION_SUMMARY 행(region 별)과 SUMMARY 행(inference time / peak mem / FLOPs)을 쓴다.

주의: 배율이 다르면 유효 GT 픽셀 수가 달라진다 (원본 1배에서는 MAX_DISP=192 를 넘는
GT 가 23.8% 라 --gt_out_of_range invalid 로 빠진다). 그래서 배율 간 EPE 는 "같은
장면을 다른 해상도로 푼 난이도" 뿐 아니라 "평가 대상 픽셀 집합" 도 함께 바뀐다.
표에 npix 를 같이 찍는 것은 그 때문이다.

Usage:
    python tools/collect_middlebury_scale.py
    python tools/collect_middlebury_scale.py --metric d1 --region global
    python tools/collect_middlebury_scale.py --csv                    # <root>/_summary_epe.csv
    python tools/collect_middlebury_scale.py --metric d1 --csv        # <root>/_summary_d1.csv
    python tools/collect_middlebury_scale.py --root <dir> --csv /path/to/out.csv
"""

import argparse
import csv
import glob
import os
import re

DEFAULT_ROOT = ("/media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/"
                "middlebury_eval/lightgev_modify_4_scale")

# "..._full_iter_6_r_075_dil_136_scale_050_gtoor_invalid_bs12_..." 에서 variant / scale / gtoor
RUN_RE = re.compile(r"lightgev_modify_4_pretrained_sceneflow_(?P<variant>.+?)"
                    r"_scale_(?P<scale>\d{3})_gtoor_(?P<gtoor>\w+?)_bs12")

SCALE_LABEL = {"100": "1.0x", "050": "0.5x", "025": "0.25x"}
SCALE_ORDER = ["100", "050", "025"]


def collect(root):
    """Return {(variant, scale, gtoor): {region: (npix, {metric: value})}, ...} + summary rows."""
    runs = {}
    for run_dir in sorted(glob.glob(os.path.join(root, "*"))):
        csv_path = os.path.join(run_dir, "results.csv")
        if not os.path.isfile(csv_path):
            continue
        m = RUN_RE.search(os.path.basename(run_dir))
        if not m:
            print("skip (이름 형식이 다름): {}".format(os.path.basename(run_dir)))
            continue

        regions, summary = {}, None
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["name"] == "REGION_SUMMARY":
                    regions[row["metric"]] = (
                        int(row["npix"]),
                        {k: float(row[k]) for k in ("epe", "d1", "t1", "t3")},
                    )
                elif row["name"] == "SUMMARY":
                    summary = {k: row[k] for k in ("infer_time", "peak_mem_gb", "flops_gmac")}
        if regions:
            runs[(m.group("variant"), m.group("scale"), m.group("gtoor"))] = (regions, summary)
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=DEFAULT_ROOT, help="run 디렉토리들이 있는 폴더")
    parser.add_argument("--metric", default="epe", choices=["epe", "d1", "t1", "t3"])
    parser.add_argument("--region", nargs="+", default=["all"],
                        help="찍을 region, 또는 'all' (기본: all)")
    parser.add_argument("--csv", nargs="?", const="AUTO", default=None,
                        help="표를 csv 로도 저장한다. 경로를 주면 그 경로에, "
                             "값 없이 --csv 만 주면 <root>/_summary_<metric>.csv 에 저장. "
                             "csv 에는 --metric 으로 고른 지표 하나만 들어가므로 "
                             "지표별로 파일이 하나씩 생긴다 (gt_out_of_range 는 컬럼).")
    args = parser.parse_args()

    runs = collect(args.root)
    if not runs:
        raise SystemExit("결과가 없다: {}".format(args.root))

    variants = sorted({k[0] for k in runs})
    scales = [s for s in SCALE_ORDER if any(k[1] == s for k in runs)]
    gtoors = sorted({k[2] for k in runs})

    all_regions = ["texless", "edge", "thick:thinQ", "thick:thickQ", "global"]
    present = [r for r in all_regions if any(r in v[0] for v in runs.values())]
    regions = present if args.region == ["all"] else args.region

    out_rows = []
    for gtoor in gtoors:
        print("\n=== gt_out_of_range = {} | metric = {} ===".format(gtoor, args.metric.upper()))
        header = "{:<30}{:>12}".format("variant", "region")
        header += "".join("{:>14}".format(SCALE_LABEL.get(s, s)) for s in scales)
        print(header)
        print("-" * len(header))
        for variant in variants:
            for region in regions:
                cells = []
                for s in scales:
                    entry = runs.get((variant, s, gtoor))
                    if entry is None or region not in entry[0]:
                        cells.append("{:>14}".format("-"))
                        continue
                    cells.append("{:>14.4f}".format(entry[0][region][1][args.metric]))
                    out_rows.append([gtoor, variant, region, s, entry[0][region][1][args.metric],
                                     entry[0][region][0]])
                print("{:<30}{:>12}{}".format(variant, region, "".join(cells)))
            print()

        # 평가 대상 픽셀 수 (배율 간 비교 시 반드시 같이 볼 것)
        print("{:<30}{:>12}".format("(global 유효 픽셀 수)", "npix")
              + "".join("{:>14}".format(
                  runs[(variants[0], s, gtoor)][0]["global"][0]
                  if (variants[0], s, gtoor) in runs else "-") for s in scales))
        row = runs.get((variants[0], scales[0], gtoor))
        if row and row[1]:
            print("{:<30}{:>12}".format("(infer ms / peak GB)", "")
                  + "".join("{:>14}".format(
                      "{}/{}".format(runs[(variants[0], s, gtoor)][1]["infer_time"],
                                     runs[(variants[0], s, gtoor)][1]["peak_mem_gb"])
                      if (variants[0], s, gtoor) in runs else "-") for s in scales))

    if args.csv:
        csv_path = (os.path.join(args.root, "_summary_{}.csv".format(args.metric))
                    if args.csv == "AUTO" else args.csv)
        parent = os.path.dirname(os.path.abspath(csv_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gt_out_of_range", "variant", "region", "scale", args.metric, "npix"])
            w.writerows(out_rows)
        print("\nsaved: {}".format(csv_path))


if __name__ == "__main__":
    main()


'''
bash V_experiments/evaluation/eval_pretrained_sceneflow_by_middlebury_lightgev_modify_4_scale.sh
python tools/collect_middlebury_scale.py
'''
