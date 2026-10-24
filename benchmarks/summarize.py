# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Markdown tables of bench_sdm.py and bench_sdm_gdn.py results.

Runs with identical arguments (repeats, e.g. fresh processes) are merged by the median of
their medians; "spread" is (max - min) / median over the repeats. With --baseline, every row
also shows the ratio to the baseline run with the same arguments (e.g. another checkout),
below 1 when the results are faster.

Usage:
  python benchmarks/summarize.py benchmarks/results/ [--baseline other_results/] [--out summary.md]
"""

import argparse
import glob
import json
import os
import statistics
import sys

# bench_sdm.py arguments that identify a row, and the timings reported per mode
SDM_KEYS = ("mode", "dim", "hd", "W", "R", "C", "B", "T", "sph", "dtype", "quant", "kwd",
            "tf32", "indices", "sac", "sdm_cuda_graph")
GDN_KEYS = ("mixer", "mode", "dim", "batch", "seqlen", "dtype", "n_blocks", "swa_window",
            "swa_heads", "sdm_heads", "slots_per_head", "writes", "reads", "chunk",
            "snapshot_quant", "sdm_cuda_graph", "sac", "gdn_heads", "gdn_head_dim",
            "gdn_expand_v", "optimizer")


def load(paths):
    runs = []
    for path in paths:
        files = (sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
                 if os.path.isdir(path) else [path])
        for f in files:
            try:
                with open(f) as fh:
                    r = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(r, dict) and r.get("bench") in ("sdm", "sdm_gdn"):
                runs.append(r)
    return runs


def key(r):
    names = SDM_KEYS if r["bench"] == "sdm" else GDN_KEYS
    return (r["bench"],) + tuple((n, r["args"].get(n)) for n in names)


def metrics(r):
    """{column: value} of one run."""
    out = {}
    if r["bench"] == "sdm":
        for mode in ("eager", "graph"):
            for name, v in (r.get(mode) or {}).items():
                if isinstance(v, dict) and "median" in v and name != "host_issue_ms":
                    out[f"{mode} {name.removesuffix('_ms')}"] = v["median"]
        for name, v in (r.get("kernel_ms") or {}).items():
            if name != "launches":
                out[f"kernel {name}"] = v
        if "peak_gib" in r:
            out["peak GiB"] = r["peak_gib"]
    else:
        for name, v in (r.get("timing") or {}).items():
            if name != "host_issue_ms":
                out[name.removesuffix("_ms")] = v["median"]
        out["peak GiB"] = (r.get("memory") or {}).get("peak_allocated_gib")
    return out


def merge(runs):
    rows = {}
    for r in runs:
        row = rows.setdefault(key(r), {"runs": 0, "failed": [], "values": {}})
        if r.get("status") != "ok":
            row["failed"].append(r.get("status"))
            continue
        row["runs"] += 1
        for name, v in metrics(r).items():
            if v is not None:
                row["values"].setdefault(name, []).append(v)
    return rows


def fmt(x):
    return "-" if x is None else f"{x:.3f}" if abs(x) < 10 else f"{x:.2f}"


def table(bench, rows, base):
    keys = [k for k in rows if k[0] == bench]
    if not keys:
        return []
    columns = sorted({c for k in keys for c in rows[k]["values"]})
    names = [n for n, _ in keys[0][1:]]
    head = names + [f"{c} ms" if "GiB" not in c else c for c in columns] + ["spread", "runs"]
    if base:
        head.append("vs baseline")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for k in sorted(keys, key=lambda k: [str(v) for _, v in k[1:]]):
        row = rows[k]
        med = {c: statistics.median(v) for c, v in row["values"].items()}
        main = next((c for c in ("graph fwd_bwd", "fwd_bwd", "graph fwd", "graph step",
                                 "eager fwd_bwd", "eager fwd", "eager step") if c in med), None)
        vals = row["values"].get(main, [])
        spread = (max(vals) - min(vals)) / statistics.median(vals) if len(vals) > 1 else None
        cells = [str(v) for _, v in k[1:]] + [fmt(med.get(c)) for c in columns]
        cells += [fmt(spread), f"{row['runs']}" + (f" ({len(row['failed'])} failed)"
                                                    if row["failed"] else "")]
        if base:
            b = base.get(k, {}).get("values", {}).get(main)
            cells.append(f"{med[main] / statistics.median(b):.3f} ({main})" if b and main
                         else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+")
    p.add_argument("--baseline", nargs="*", default=[])
    p.add_argument("--out")
    a = p.parse_args(argv)
    runs = load(a.paths)
    if not runs:
        print("no results found", file=sys.stderr)
        return 1
    rows = merge(runs)
    base = merge(load(a.baseline)) if a.baseline else None
    env = runs[0].get("environment") or {}
    out = [f"GPU: {env.get('gpu')}, torch {env.get('torch')}, CUDA {env.get('cuda')}", ""]
    for bench, title in (("sdm", "SDM layer and op (bench_sdm.py)"),
                         ("sdm_gdn", "SDM vs GDN model (bench_sdm_gdn.py)")):
        lines = table(bench, rows, base)
        if lines:
            out += [f"## {title}", ""] + lines + [""]
    text = "\n".join(out)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
