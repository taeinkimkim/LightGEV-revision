"""Plot per-epoch training/validation curves parsed from a training log.

Series available (metric names to pass to --metric):

  epe, loss, d1_all, thres_1/2/3
      validation pass, from "Epoch N metrics: {...}"

  train_epe, train_loss, train_d1_all, ...
      the SAME pass run over the training split with validation preprocessing,
      from "Epoch N train_metrics: {...}". Enabled by TRAINER.TRAIN_EVAL_BATCHES
      (0 disables, negative = as many batches as the validation set). Because the
      preprocessing, the model mode and the loss terms are identical to the
      validation pass, train and val are directly comparable and the gap between
      them is a clean generalization gap.

  iter_loss
      the running mean training loss accumulated *during* training, from
      "Training Epoch: 1/90 Iter:1584/1586 Loss:4.71770(3.65638)". Available for
      every epoch (validation only runs every TRAINER.EVAL_INTERVAL epochs), but
      it is measured under augmentation with the weights still changing, so it is
      NOT comparable to the validation loss. Use train_loss for that.

A validation metric and its train_ counterpart passed together are drawn in one
axes with a legend. The table prints "-" where a series has no value for an epoch.

Usage:
    python tools/plot_val_metrics.py --log <path/to/train_*.log>
    python tools/plot_val_metrics.py --log <log> --metric epe train_epe
    python tools/plot_val_metrics.py --log <log> --metric loss train_loss
    python tools/plot_val_metrics.py --log <log> --metric all --out curves.png
    python tools/plot_val_metrics.py --log <log> --print-only
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# "Epoch 5 metrics: {...}"      -> validation pass
# "Epoch 5 train_metrics: {...}" -> train split evaluated with the same preprocessing
EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s+metrics:\s*\{(.*)\}")
TRAIN_EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s+train_metrics:\s*\{(.*)\}")
# "'epe': tensor(2.3741)" or "'epe': 2.3741"
METRIC_RE = re.compile(r"'([^']+)':\s*(?:tensor\()?\s*(-?[\d.]+(?:e-?\d+)?)")
# "Training Epoch: 1/90 Iter:1584/1586 Loss:4.71770(3.65638)"
#  -> epoch, iteration, running mean loss
LOSS_RE = re.compile(
    r"Training Epoch:\s*(\d+)/\d+\s+Iter:\s*(\d+)/\d+\s+"
    r"Loss:\s*[^(]+\(\s*([-\d.]+(?:[eE][-+]?\d+)?|nan|inf)\s*\)")

# y-axis label per metric
YLABEL = {
    "iter_loss": "Training loss (running mean)",
    "loss": "Loss",
    "epe": "EPE (px)",
    "d1_all": "D1-all (%)",
    "thres_1": ">1px (%)",
    "thres_2": ">2px (%)",
    "thres_3": ">3px (%)",
}

# metrics that are two views of the same quantity are drawn in one axes
PAIRS = {
    "epe": "train_epe",
    "loss": "train_loss",
    "d1_all": "train_d1_all",
    "thres_1": "train_thres_1",
    "thres_2": "train_thres_2",
    "thres_3": "train_thres_3",
}


def base_name(metric):
    return metric[len("train_"):] if metric.startswith("train_") else metric


def parse_log(path):
    """Return {metric_name: [(epoch, value), ...]} sorted by epoch.

    Validation metrics keep their own names; the per-epoch mean training loss is
    returned under the name "loss".
    """
    records = {}
    loss_last_iter = {}   # epoch -> (iteration, running mean) for the furthest iteration seen

    with open(path) as f:
        for line in f:
            m = TRAIN_EPOCH_RE.search(line)
            if m:
                epoch = int(m.group(1))
                for name, value in METRIC_RE.findall(m.group(2)):
                    records.setdefault("train_" + name, []).append((epoch, float(value)))
                continue

            m = EPOCH_RE.search(line)
            if m:
                epoch = int(m.group(1))
                for name, value in METRIC_RE.findall(m.group(2)):
                    records.setdefault(name, []).append((epoch, float(value)))
                continue

            m = LOSS_RE.search(line)
            if m:
                epoch, iteration, value = int(m.group(1)), int(m.group(2)), m.group(3)
                try:
                    value = float(value)
                except ValueError:
                    continue
                if value != value:     # nan
                    continue
                # the epoch mean is the running mean at the last logged iteration
                if epoch not in loss_last_iter or iteration >= loss_last_iter[epoch][0]:
                    loss_last_iter[epoch] = (iteration, value)

    if loss_last_iter:
        records["iter_loss"] = [(e, v) for e, (_, v) in loss_last_iter.items()]

    for name in records:
        # a resumed run can log the same epoch twice; keep the last one
        dedup = dict(records[name])
        records[name] = sorted(dedup.items())
    return records


SERIES_COLOR = {"val": "#2563eb", "train": "#ea580c", "": "#2563eb"}


def group_panels(metrics):
    """Return [(base_name, [(metric, series_label), ...]), ...].

    A validation metric and its train-split counterpart go into the same axes so
    the generalization gap is readable at a glance.
    """
    panels, used = [], set()
    for m in metrics:
        if m in used:
            continue
        base = base_name(m)
        partner = PAIRS.get(base)
        if partner and base in metrics and partner in metrics:
            series = [(base, "val"), (partner, "train")]
        else:
            series = [(m, "train" if m.startswith("train_") else "")]
        used.update(name for name, _ in series)
        panels.append((base, series))
    return panels


def plot(records, metrics, out_path, title):
    panels = group_panels(metrics)
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5), squeeze=False)

    for ax, (base, series) in zip(axes[0], panels):
        all_epochs = []
        for name, label in series:
            points = records[name]
            epochs = [e for e, _ in points]
            values = [v for _, v in points]
            all_epochs.extend(epochs)
            ax.plot(epochs, values, marker="o", markersize=4, linewidth=1.6,
                    color=SERIES_COLOR.get(label, "#2563eb"), label=label or None)

        # ticks every 10 epochs, plus the last epoch
        ticks = [t for t in range(0, max(all_epochs) + 1, 10) if t >= min(all_epochs)]
        if ticks[-1:] != [max(all_epochs)]:
            ticks.append(max(all_epochs))
        ax.set_xticks(ticks)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(YLABEL.get(base, base))
        ax.grid(alpha=0.3, linestyle="--")
        ax.margins(y=0.15)
        if len(series) > 1:
            ax.legend(frameon=False)

    # fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print("saved: {}".format(out_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True, help="path to the training .log file")
    parser.add_argument("--metric", nargs="+", default=None,
                        help="metric names to plot, or 'all'. Validation metrics keep their "
                             "bare names (epe, loss, d1_all, ...); the train-split pass is "
                             "prefixed with train_ (train_epe, train_loss, ...); the running "
                             "training loss is 'iter_loss'. A val/train pair given together is "
                             "drawn in one axes. Default: epe/train_epe/loss/train_loss when "
                             "present, else epe.")
    parser.add_argument("--out", default=None,
                        help="output image path (default: <log dir>/val_<metric>.png)")
    parser.add_argument("--print-only", action="store_true", help="print the table, skip the plot")
    args = parser.parse_args()

    records = parse_log(args.log)
    if not records:
        raise SystemExit("no 'Epoch N metrics: {{...}}' or 'Training Epoch: ... Loss:...' "
                         "lines found in {}".format(args.log))

    if args.metric is None:
        preferred = ["epe", "train_epe", "loss", "train_loss"]
        metrics = [m for m in preferred if m in records] or ["epe"]
    elif args.metric == ["all"]:
        metrics = sorted(records)
    else:
        metrics = args.metric
    unknown = [m for m in metrics if m not in records]
    if unknown:
        raise SystemExit("metric(s) {} not in log; available: {}".format(
            ", ".join(unknown), ", ".join(sorted(records))))

    # table to stdout
    # "loss" exists for every epoch while validation runs only every EVAL_INTERVAL
    # epochs, so take the union of epochs and leave "-" where a metric has no value
    header = ["epoch"] + metrics
    print("  ".join(h.rjust(9) for h in header))
    lookup = {m: dict(records[m]) for m in metrics}
    epochs = sorted({e for m in metrics for e in lookup[m]})
    for epoch in epochs:
        row = [str(epoch)] + ["{:.4f}".format(lookup[m][epoch]) if epoch in lookup[m] else "-"
                              for m in metrics]
        print("  ".join(c.rjust(9) for c in row))

    for m in metrics:
        best_e, best_v = min(records[m], key=lambda p: p[1])
        print("best {}: {:.4f} @ epoch {}".format(m, best_v, best_e))

    if args.print_only:
        return

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.log)),
                                   "curves_{}.png".format("_".join(metrics)))
    plot(records, metrics, out, title=os.path.basename(args.log))


if __name__ == "__main__":
    main()

'''
python tools/plot_val_metrics.py \
    --log train_outputs/SceneFlowDataset/LightStereo/lightstereo_m_train_sceneflow/pretrain_trainBS12_valBS12_LR0.0004/train_20260830-131502_0.log
'''