"""Plot the per-region REGION_SUMMARY metric of a GEV-iteration sweep in one graph.

Every eval run writes a results.csv whose last rows aggregate the whole test set
per region, e.g.

    SUMMARY,-,4088337854,1.046913,3.180736,8.290325,4.045699,10.25,0.3679,38.6294
    REGION_SUMMARY,texless,1304743672,0.926731,2.449987,6.827351,3.333115,,,
    REGION_SUMMARY,edge,121624979,1.865831,6.993634,15.719470,7.853541,,,
    REGION_SUMMARY,thick:thinQ,224970094,2.528962,10.298826,26.497978,12.243075,,,
    REGION_SUMMARY,thick:thickQ,664909161,0.617208,1.500658,3.394990,1.983237,,,
    REGION_SUMMARY,global,1772089948,1.052275,3.183803,8.382878,4.042198,,,

This script collects those rows from every run directory matching --pattern,
reads the GEV iteration count out of the directory name (``..._full_iter_<N>_...``)
and draws one line per region against the iteration count.

The region-to-region spread (~0.6 to ~2.5 px EPE) is two orders of magnitude
larger than the change across iterations (~0.03 px), so on a shared absolute
axis every line looks flat no matter how fine the ticks are. --layout picks how
to get around that: 'delta' re-bases each region to its own first iteration
(one graph, change is the signal), 'facet' keeps absolute values but gives each
region its own zoomed axis, 'shared' is the plain absolute version.

Usage:
    python tools/plot_region_epe_by_iter.py                      # --layout delta
    python tools/plot_region_epe_by_iter.py --layout facet
    python tools/plot_region_epe_by_iter.py --metric d1 --out region_d1.png
    python tools/plot_region_epe_by_iter.py --root <outputs/sceneflow_eval_test_all> --print-only
"""

import argparse
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DEFAULT_ROOT = ("/media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/"
                "sceneflow_eval_test_all/lightgev_modify_4")
DEFAULT_PATTERN = ("lightgev_modify_4_pretrained_sceneflow_full_iter_*"
                   "_r_075_bs12_lr0004_normT_biasF_valid_GT_revised")

# "..._full_iter_6_r_075_..." -> 6
ITER_RE = re.compile(r"_iter_(\d+)")

# the regions in reading order; 'global' is the aggregate, drawn apart from them
REGIONS = ["texless", "edge", "thick:thinQ", "thick:thickQ"]
GLOBAL = "global"

# categorical slots 1-4, assigned in fixed order (never cycled)
COLORS = {
    "texless": "#2a78d6",
    "edge": "#eb6834",
    "thick:thinQ": "#1baf7a",
    "thick:thickQ": "#eda100",
}
GLOBAL_COLOR = "#57534e"
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#57534e"


def collect(root, pattern, metric):
    """Return {region: {iters: value}} parsed from every matching results.csv."""
    records = {}
    for run_dir in sorted(glob.glob(os.path.join(root, pattern))):
        csv_path = os.path.join(run_dir, "results.csv")
        if not os.path.isfile(csv_path):
            print("skip (no results.csv): {}".format(run_dir))
            continue

        m = ITER_RE.search(os.path.basename(run_dir))
        if not m:
            print("skip (no _iter_<N> in name): {}".format(run_dir))
            continue
        iters = int(m.group(1))

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if metric not in reader.fieldnames:
                raise SystemExit("metric '{}' not in {}; available: {}".format(
                    metric, csv_path, ", ".join(reader.fieldnames)))
            for row in reader:
                if row["name"] != "REGION_SUMMARY":
                    continue
                records.setdefault(row["metric"], {})[iters] = float(row[metric])
    return records


def unit(metric):
    return "px" if metric == "epe" else "%"


def style_axes(ax, all_iters, xlabel=True):
    ax.set_xticks(all_iters)
    if xlabel:
        ax.set_xlabel("GEV iterations", fontsize=10, color=TEXT_SECONDARY)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d6d3d1")
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    # denser ticks than the default, so small wiggles still land between gridlines
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))


def series(records, region):
    points = sorted(records[region].items())
    return [i for i, _ in points], [v for _, v in points]


def draw_line(ax, region, xs, ys, label=True):
    is_global = region == GLOBAL
    ax.plot(xs, ys,
            marker="s" if is_global else "o",
            markersize=6.5, linewidth=2.0,
            linestyle="--" if is_global else "-",
            color=COLORS.get(region, GLOBAL_COLOR),
            markeredgecolor="white", markeredgewidth=1.0,
            label=region, zorder=2 if is_global else 3)
    if label:
        # direct label at the right end, so identity is never color-alone
        ax.annotate(" {}".format(region), xy=(xs[-1], ys[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=TEXT_SECONDARY)


def plot_shared(records, regions, metric, title):
    """One axis, absolute values. Region spread dominates; per-iteration wiggle is tiny."""
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for region in regions:
        draw_line(ax, region, *series(records, region))

    all_iters = sorted({i for r in regions for i in records[r]})
    style_axes(ax, all_iters)
    ax.set_ylabel("{} ({})".format(metric.upper(), unit(metric)),
                  fontsize=10, color=TEXT_SECONDARY)
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left")
    ax.margins(x=0.16, y=0.12)
    # below the axes, so the legend never lands on top of a line
    ax.legend(fontsize=8.5, frameon=False, ncol=len(regions),
              loc="upper center", bbox_to_anchor=(0.5, -0.13),
              columnspacing=1.4, handlelength=1.8)
    return fig


def plot_delta(records, regions, metric, title):
    """One axis, each region re-based to its own first iteration, so the wiggle is the signal."""
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    base_iter = min(i for r in regions for i in records[r])

    for region in regions:
        xs, ys = series(records, region)
        base = records[region][base_iter]
        draw_line(ax, region, xs, [y - base for y in ys])

    all_iters = sorted({i for r in regions for i in records[r]})
    ax.axhline(0.0, linewidth=1.0, color="#a8a29e", zorder=1)
    style_axes(ax, all_iters)
    ax.set_ylabel("{} change vs. iter {} ({})".format(metric.upper(), base_iter, unit(metric)),
                  fontsize=10, color=TEXT_SECONDARY)
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left")
    ax.margins(x=0.16, y=0.18)
    ax.legend(fontsize=8.5, frameon=False, ncol=len(regions),
              loc="upper center", bbox_to_anchor=(0.5, -0.13),
              columnspacing=1.4, handlelength=1.8)
    return fig


def plot_facet(records, regions, metric, title):
    """Small multiples: every region keeps absolute values on its own zoomed axis."""
    ncol = min(3, len(regions))
    nrow = (len(regions) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.7 * nrow), squeeze=False)
    all_iters = sorted({i for r in regions for i in records[r]})

    for ax, region in zip(axes.ravel(), regions):
        xs, ys = series(records, region)
        draw_line(ax, region, xs, ys, label=False)
        style_axes(ax, all_iters, xlabel=False)
        ax.set_title(region, fontsize=9.5, color=TEXT_PRIMARY, loc="left")
        ax.margins(x=0.12, y=0.30)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    for ax in axes.ravel()[len(regions):]:
        ax.set_visible(False)

    fig.supxlabel("GEV iterations", fontsize=10, color=TEXT_SECONDARY)
    fig.supylabel("{} ({})".format(metric.upper(), unit(metric)),
                  fontsize=10, color=TEXT_SECONDARY)
    fig.suptitle(title, fontsize=11, color=TEXT_PRIMARY, x=0.02, ha="left")
    return fig


LAYOUTS = {"shared": plot_shared, "delta": plot_delta, "facet": plot_facet}


def plot(records, regions, metric, out_path, title, layout):
    fig = LAYOUTS[layout](records, regions, metric, title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("saved: {}".format(out_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="directory holding the eval run directories")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN,
                        help="glob for the run directories under --root")
    parser.add_argument("--metric", default="epe", choices=["epe", "d1", "t1", "t3"],
                        help="results.csv column to plot (default: epe)")
    parser.add_argument("--region", nargs="+", default=["all"],
                        help="regions to plot, or 'all' (default: all)")
    parser.add_argument("--layout", default="delta", choices=sorted(LAYOUTS),
                        help="'delta': one axis, re-based to the first iteration, so the "
                             "per-iteration change is readable; 'shared': one axis, absolute "
                             "values (region spread hides the change); 'facet': small multiples, "
                             "absolute values on a per-region zoomed axis (default: delta)")
    parser.add_argument("--out", default=None,
                        help="output image path (default: <root>/region_<metric>_by_iter.png)")
    parser.add_argument("--print-only", action="store_true", help="print the table, skip the plot")
    args = parser.parse_args()

    records = collect(args.root, args.pattern, args.metric)
    if not records:
        raise SystemExit("no REGION_SUMMARY rows found under {}/{}".format(args.root, args.pattern))

    known = [r for r in REGIONS + [GLOBAL] if r in records]
    known += [r for r in sorted(records) if r not in known]  # any region not in REGIONS
    if args.region == ["all"]:
        regions = known
    else:
        unknown = [r for r in args.region if r not in records]
        if unknown:
            raise SystemExit("region(s) {} not found; available: {}".format(
                ", ".join(unknown), ", ".join(known)))
        regions = args.region

    # table to stdout
    all_iters = sorted({i for r in regions for i in records[r]})
    print("  ".join(h.rjust(14) for h in ["iters"] + regions))
    for iters in all_iters:
        cells = [str(iters)] + ["{:.6f}".format(records[r][iters]) if iters in records[r] else "-"
                                for r in regions]
        print("  ".join(c.rjust(14) for c in cells))

    if args.print_only:
        return

    out = args.out or os.path.join(args.root, "region_{}_by_iter_{}.png".format(
        args.metric, args.layout))
    plot(records, regions, args.metric, out,
         title="Per-region {} vs. GEV iterations".format(args.metric.upper()),
         layout=args.layout)


if __name__ == "__main__":
    main()


'''
python tools/plot_region_epe_by_iter.py \
    --root /media/user/PortableSSD_taein/TaeinKim/LightGEV/outputs/sceneflow_eval_test_all \
    --out outputs/region_epe_by_iter.png
'''
