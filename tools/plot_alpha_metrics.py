"""Plot the learnable geometry gate scalar (geo_gate_alpha) parsed from a training log.

The trainer writes one line per stage, e.g.

    2026-08-27 21:29:53,001   INFO  GeoGateAlpha init: geo_gate_alpha=1.000000
    2026-08-27 21:35:32,709   INFO  GeoGateAlpha epoch 0: geo_gate_alpha=0.983412
    2026-08-28 04:11:02,455   INFO  GeoGateAlpha final: geo_gate_alpha=0.874903

This script scrapes those lines and plots alpha against epoch, marking the
initial value (before training) and the final value (after training).

Usage:
    python tools/plot_alpha_metrics.py --log <path/to/train_*.log>
    python tools/plot_alpha_metrics.py --log <log> --out alpha.png
    python tools/plot_alpha_metrics.py --log <log> --print-only
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# "GeoGateAlpha epoch 3: ..." / "GeoGateAlpha init: ..." / "GeoGateAlpha final: ..."
STAGE_RE = re.compile(r"GeoGateAlpha\s+(init|final|epoch\s+(\d+))\s*:\s*(.*)")
# "geo_gate_alpha=0.983412" (parameter names may be dotted, e.g. "module.geo_gate_alpha")
VALUE_RE = re.compile(r"([\w.]+)\s*=\s*(-?[\d.]+(?:[eE][-+]?\d+)?)")


def parse_log(path):
    """Return {param_name: {'init': v, 'final': v, 'epochs': [(epoch, v), ...]}}."""
    records = {}
    with open(path) as f:
        for line in f:
            m = STAGE_RE.search(line)
            if not m:
                continue
            epoch = int(m.group(2)) if m.group(2) is not None else None
            for name, value in VALUE_RE.findall(m.group(3)):
                entry = records.setdefault(name, {"init": None, "final": None, "epochs": []})
                if epoch is not None:
                    entry["epochs"].append((epoch, float(value)))
                elif m.group(1) == "init":
                    entry["init"] = float(value)
                else:
                    entry["final"] = float(value)

    for entry in records.values():
        # a resumed run can log the same epoch twice; keep the last one
        entry["epochs"] = sorted(dict(entry["epochs"]).items())
    return records


def plot(records, params, out_path, title):
    fig, axes = plt.subplots(1, len(params), figsize=(6 * len(params), 4.5), squeeze=False)

    for ax, name in zip(axes[0], params):
        entry = records[name]
        points = entry["epochs"]
        epochs = [e for e, _ in points]
        values = [v for _, v in points]

        if points:
            ax.plot(epochs, values, marker="o", markersize=4, linewidth=1.6, color="#2563eb",
                    label="per-epoch")

        if entry["init"] is not None:
            ax.axhline(entry["init"], linestyle="--", linewidth=1.2, color="#6b7280",
                       label="init: {:.4f}".format(entry["init"]))

        if epochs:
            # ticks every 10 epochs, plus the last epoch
            ticks = [t for t in range(0, max(epochs) + 1, 10) if t >= min(epochs)]
            if ticks[-1:] != [max(epochs)]:
                ticks.append(max(epochs))
            ax.set_xticks(ticks)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("geo_gate_alpha")
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3, linestyle="--")
        ax.margins(y=0.15)
        ax.legend(fontsize=8, loc="best")

    # fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("saved: {}".format(out_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True, help="path to the training .log file")
    parser.add_argument("--param", nargs="+", default=["all"],
                        help="geo_gate_alpha parameter names to plot, or 'all' (default: all)")
    parser.add_argument("--out", default=None,
                        help="output image path (default: <log dir>/geo_gate_alpha.png)")
    parser.add_argument("--print-only", action="store_true", help="print the table, skip the plot")
    args = parser.parse_args()

    records = parse_log(args.log)
    if not records:
        raise SystemExit("no 'GeoGateAlpha ...' lines found in {} "
                         "(only LightGEV runs with the geo-cost gate log them)".format(args.log))

    params = sorted(records) if args.param == ["all"] else args.param
    unknown = [p for p in params if p not in records]
    if unknown:
        raise SystemExit("param(s) {} not in log; available: {}".format(
            ", ".join(unknown), ", ".join(sorted(records))))

    # table to stdout
    header = ["epoch"] + params
    print("  ".join(h.rjust(14) for h in header))
    lookup = {p: dict(records[p]["epochs"]) for p in params}

    def row(label, getter):
        cells = [label] + ["{:.6f}".format(getter(p)) if getter(p) is not None else "-"
                           for p in params]
        print("  ".join(c.rjust(14) for c in cells))

    row("init", lambda p: records[p]["init"])
    epochs = sorted({e for p in params for e in lookup[p]})
    for epoch in epochs:
        row(str(epoch), lambda p, e=epoch: lookup[p].get(e))
    row("final", lambda p: records[p]["final"])

    for p in params:
        init_v, final_v = records[p]["init"], records[p]["final"]
        if final_v is None and lookup[p]:
            final_v = lookup[p][max(lookup[p])]
        if init_v is not None and final_v is not None:
            print("{}: {:.6f} -> {:.6f} (delta {:+.6f})".format(p, init_v, final_v, final_v - init_v))

    if args.print_only:
        return

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.log)),
                                   "geo_gate_alpha.png")
    plot(records, params, out, title=os.path.basename(args.log))


if __name__ == "__main__":
    main()

'''
python tools/plot_alpha_metrics.py \
    --log train_outputs/SceneFlowDataset/LightGEV/lightgev_train_sceneflow/pretrain/train_20260827-212953_0.log
'''
