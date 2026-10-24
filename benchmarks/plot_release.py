# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Speed plot of release_tables.py's JSON: this version against the previous release.

Three panels: one SDM layer (forward + backward) and the 3:1 SWA:SDM model (training step), both
at 32K tokens per step (B x T = 1 x 32K, 2 x 16K, 4 x 8K), and decoding (time per token, batch 1)
at three context lengths. In each group: the previous release (eager, its only mode) at 1.0x, this
version eager, and this version with CUDA graphs (the ``cuda_graph=True`` option, left out when the
run did not time it; for decoding, the generator's CUDA-graph decode step). Bar height: speedup =
previous release time / this version's time; above each bar, its time in ms.

When the JSON's host-bound cells follow the uncontended rule (``host_selection.rule_applied``),
a quantity that rule marks noisy is drawn hatched, at the median over the processes of every run
in ``attachments.cross_runs``, with a whisker from the smallest to the largest of those runs'
medians. A cross run lists only the cells and versions of the code it timed; with one run of a
quantity's code, it is drawn at that median without a whisker.
``--table FILE`` writes the plotted values as a markdown table. Needs matplotlib.

Usage: python benchmarks/plot_release.py results/release/release_compare.json \\
           --out benchmarks/speedup_gb300 [--window 128] [--table FILE]
"""

import argparse
import json
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SHAPES = ((1, 32768), (2, 16384), (4, 8192))
COLORS = ("#b8b8b8", "#e8a33d", "#4a3aa7")  # previous release, this version eager, CUDA graphs
LABELS = ("previous release", "this version, eager", "this version, CUDA graphs")


def quantity(doc, cell, v, key):
    """(value, (low, high) or None, noisy) of one plotted quantity."""
    hs = doc.get("host_selection") or {}
    if hs.get("rule_applied") == "uncontended" and f"{cell} {v} {key}" in hs.get("noisy", []):
        cross = (doc.get("attachments") or {}).get("cross_runs") or {}
        # the runs of this quantity's code (a run lists only the cells and versions of the code it timed)
        runs = [r["cells"][cell][v][key] for r in cross.values()
                if "processes" in r.get("cells", {}).get(cell, {}).get(v, {}).get(key, {})]
        if not runs:  # a single run: its median over all processes (the tables' value), hatched, no range
            return doc["merged"][cell][v]["median"][key], None, True
        pooled = statistics.median([x for r in runs for x in r["processes"]])
        meds = [r["all_median"] for r in runs]
        return pooled, ((min(meds), max(meds)) if len(runs) > 1 else None), True
    return doc["merged"][cell][v]["median"][key], None, False


def ms_per_token(q):
    """A tokens/s quantity as ms per token (the range flips)."""
    x, rng, noisy = q
    return 1e3 / x, (None if rng is None else (1e3 / rng[1], 1e3 / rng[0])), noisy


def rows(doc, window):
    """Per panel, [(label, [previous, eager, CUDA graphs] as (ms, ms range or None, noisy))]."""
    layer, model = [], []
    timed = doc["merged"]
    for B, T in SHAPES:
        e, o = f"layer_B{B}_T{T}_C64_eager", f"layeropt_B{B}_T{T}_C64"
        layer.append((f"{B} × {T // 1024}K", [quantity(doc, e, "old", "eager fwd_bwd"),
                                              quantity(doc, e, "new", "eager fwd_bwd")]
                      + ([quantity(doc, o, "new", "eager fwd_bwd")] if o in timed else [])))
        c = f"W{window}_B{B}_T{T}"
        model.append((f"{B} × {T // 1024}K", [quantity(doc, f"model_{c}_eager", "old", "step"),
                                              quantity(doc, f"model_{c}_eager", "new", "step")]
                      + ([quantity(doc, f"modelopt_{c}", "new", "step")] if f"modelopt_{c}" in timed else [])))
    cell = f"decode_W{window}"
    ctxs = sorted({int(k.rsplit("T", 1)[1]) for k in doc["merged"][cell]["new"]["median"] if " tok/s T" in k})
    decode = [(f"{c // 1024}K", [ms_per_token(quantity(doc, cell, "old", f"eager tok/s T{c}")),
                                 ms_per_token(quantity(doc, cell, "new", f"eager tok/s T{c}")),
                                 ms_per_token(quantity(doc, cell, "new", f"graph tok/s T{c}"))]) for c in ctxs]
    return layer, model, decode


def fmt_ms(x):
    return f"{x:.0f}" if x >= 100 else f"{x:.1f}" if x >= 10 else f"{x:.2f}"


def panel(ax, groups, title, xlabel):
    w = 0.29
    top = max(g[1][0][0] / (q[1][0] if q[1] else q[0]) for g in groups for q in g[1])
    gap, step = top * 0.015, top * 0.085  # label offset above a bar; label height (data units)
    for i, (_, qs) in enumerate(groups):
        prev = qs[0][0]
        y_prev = None
        for j, (t, rng, noisy) in enumerate(qs):
            x, s = i + (j - (len(qs) - 1) / 2) * w, prev / t
            ax.bar(x, s, w * 0.94, color="white" if noisy else COLORS[j], edgecolor=COLORS[j],
                   hatch="////" if noisy else None, linewidth=1.0 if noisy else 0, zorder=3)
            hi = s
            if rng is not None:
                lo_s, hi_s = prev / rng[1], prev / rng[0]
                ax.plot([x, x], [lo_s, hi_s], color="0.25", lw=1.0, zorder=4)
                ax.plot([x - w * 0.18, x + w * 0.18], [lo_s, lo_s], color="0.25", lw=1.0, zorder=4)
                ax.plot([x - w * 0.18, x + w * 0.18], [hi_s, hi_s], color="0.25", lw=1.0, zorder=4)
                hi = max(s, hi_s)
            text = "1×" if j == 0 else f"{s:.2f}×" if s < 2 else f"{s:.1f}×"
            labels = [fmt_ms(t)]
            if (s >= 0.16 * top or j == 0) and not noisy:  # the speedup inside the bar, else above it
                ax.text(x, s / 2, text, ha="center", va="center", fontsize=9, zorder=4,
                        color="0.15" if j == 0 else "white", fontweight="bold")
            elif j:
                labels.insert(0, text)
            y = hi + gap  # above the bar, and above the left neighbour's labels when they would touch
            if y_prev is not None and abs(y - y_prev) < step:
                y = y_prev + step
            for n, lab in enumerate(labels):
                bold = len(labels) == 2 and n == 0
                ax.text(x, y + n * step, lab, ha="center", va="bottom", fontsize=9 if bold else 8.5,
                        color=COLORS[j] if bold and not noisy else "0.2",
                        fontweight="bold" if bold else None, zorder=5)
            y_prev = y + (len(labels) - 1) * step
        top = max(top, y_prev + step - gap)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9.5, color="0.3", labelpad=2)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylim(0, top * 1.17)
    ax.set_ylabel("speedup", fontsize=10)
    ax.set_title(title, fontsize=11.5, loc="left")
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", color="0.9", lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def table(panels):
    """The plotted values as markdown: ms (speedup over the previous release); hatched values marked."""
    lines = ["| | previous release, eager | this version, eager | this version, CUDA graphs |",
             "|---|---|---|---|"]
    for name, groups in panels:
        for label, qs in groups:
            prev = qs[0][0]

            def cell(q, first=False):
                t, rng, noisy = q
                txt = fmt_ms(t) if first else f"{fmt_ms(t)} (**{prev / t:.2f}x**)"
                if noisy:
                    txt += " †" + (f", runs {fmt_ms(rng[0])}-{fmt_ms(rng[1])}" if rng else "")
                return txt
            third = cell(qs[2]) if len(qs) > 2 else "not timed"
            lines.append(f"| {name}, {label} | {cell(qs[0], True)} | {cell(qs[1])} | {third} |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json")
    p.add_argument("--out", required=True, help="output path without extension")
    p.add_argument("--window", type=int, default=128, help="SWA window of the model and decoding")
    p.add_argument("--gpu", default="NVIDIA GB300")
    p.add_argument("--table", help="also write the plotted values as a markdown table to this file")
    a = p.parse_args(argv)
    doc = json.load(open(a.json))
    layer, model, decode = rows(doc, a.window)
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6, "hatch.linewidth": 0.8})
    fig, axes = plt.subplots(3, 1, figsize=(5.8, 7.9))
    panel(axes[0], layer, "SDM layer, forward + backward (ms)", "batch × sequence length (32K tokens)")
    panel(axes[1], model, "3:1 SWA:SDM model, training step (ms)", "batch × sequence length (32K tokens)")
    panel(axes[2], decode, "Decoding, batch 1 (ms per token)", "context length")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    labels = list(LABELS)
    if any(q[2] for groups in (layer, model, decode) for _, qs in groups for q in qs):
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0.4", hatch="////"))
        cross = (doc.get("attachments") or {}).get("cross_runs")
        labels.append("hatched: CPU-noisy; whisker: runs of the same code" if cross
                      else "hatched: CPU-noisy, median of all processes")
    fig.legend(handles, labels, loc="upper center", ncol=3 if len(labels) == 3 else 2, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=1.0,
               handletextpad=0.4)
    uncontended = (doc.get("host_selection") or {}).get("rule_applied") == "uncontended"
    note = (f"{a.gpu}, bf16. Host-bound cells: median of uncontended processes (RESULTS.md)." if uncontended
            else f"{a.gpu}, bf16. Speedup = previous release time / this version's time.")
    fig.text(0.5, 0.003, note, ha="center", va="bottom", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 0.965 if len(labels) == 3 else 0.945), h_pad=1.2)
    fig.savefig(f"{a.out}.png", dpi=200, bbox_inches="tight", metadata={"Software": None})
    print(f"{a.out}.png")
    if a.table:
        with open(a.table, "w") as f:
            f.write(table([("layer", layer), ("model", model), ("decoding", decode)]) + "\n")


if __name__ == "__main__":
    main()
