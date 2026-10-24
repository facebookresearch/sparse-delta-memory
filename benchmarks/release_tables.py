# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tables of run_release_compare.sh timings: the previous release against this version.

Reads OUT/{old,new}/*.json (one file per process, named <cell>_r<round>.json), checks that
the equivalence verdict is EQUIVALENT and of the records' two commits and harness, that every
expected record is there, that each record matches its cell, that the two versions' runs
differ only in the checkout and that every record of a version ran the same code (one
lingua/ tree, one harness), merges the rounds of a cell by the median of the per-process
medians, and prints the README tables (speedup = previous release time / this version's
time):

  main          per cell, the previous release eager (it has no CUDA-graph option) against
                this version eager and with its ``cuda_graph=True`` option (when the verdict
                verifies the option; otherwise its cells are neither expected nor shown)
  eager_layer   the layer's forward / backward split, eager, both versions
  graph_layer   the same with the whole step captured as one CUDA graph by the benchmark,
                both versions
  model_detail  model step, tokens/s/GPU and peak memory of every run
  decode        decoding tokens/s at SWA window 128: the previous release eager (it has no
                CUDA-graph decode step) against this version eager and with its CUDA-graph
                decode step

Layer and model cells run at 32K tokens per step: B x T = 1 x 32K, 2 x 16K and 4 x 8K.

``--host-rounds quiet-socket`` merges the host-bound cells (eager, the ``cuda_graph`` option,
decoding) over the rounds of the host-bound lanes (one per CPU socket) whose processes all
stayed within HOST_SPREAD_LIMIT of each other: per lane and per host-bound cell, the spread
((max - min) / median over that lane's rounds) of the cell's timed metric (eager or option
time; eager decoding tokens/s). The selection is by lane, the same for every host-bound cell;
the graph-only cells always use every round. ``--host-rounds all`` (the default) uses every
round of every cell. ``--host-rounds quiet-socket-or-uncontended`` applies the quiet-socket rule
and, when no lane meets it, the uncontended rule: per host-bound cell, version and timed metric
(eager or option time; per context, eager and CUDA-graph decoding tokens/s), the median over the
processes of both lanes within UNCONTENDED of the cell's fastest process; a version with fewer
than 3 such processes makes the cell noisy (``host_selection.noisy``; its value is then the
median over all processes). The other metrics of a cell come from the processes kept for its
timed metric.

``--json FILE`` also writes every per-process record with its lane, the merged values (and,
under a selection, the values over all rounds), the run-to-run spread ((max - min) / median
over the rounds), the per-round speedups of the eager rows, the code versions, the hashes of
the measurement harness and of this analysis, and the equivalence verdict (``--verdict``).
Exits with status 1 when a record is missing or mismatched, and prints no tables then.

``--attach KEY=FILE`` (repeatable) stores FILE's JSON under ``attachments.KEY`` of ``--json``:
records of checks made outside these scripts.

Usage: python benchmarks/release_tables.py OUT [--rounds 6] [--host-rounds all|quiet-socket]
           [--verdict VERDICT.json] [--attach KEY=FILE ...] [--json FILE]
"""

import argparse
import glob
import hashlib
import json
import math
import os
import re
import statistics
import sys

VERSIONS = ("old", "new")


def _ck(s, key):
    """A checkout record's code_tree / code_status, or the lingua_tree / lingua_status of records
    written before the package moved to the top level (the same fingerprint of the SDM code)."""
    return s.get(key, s.get(key.replace("code_", "lingua_")))


SHAPES = ((1, 32768), (2, 16384), (4, 8192))  # B x T of the layer and model cells
WINDOWS = (128,)
DECODE = {"config": "apps/main/configs/sdm_flagship.yaml", "contexts": "1024,8192,32768",
          "gen": 256, "warm": 8, "repeats": 3, "weights": "decode_weights.pt"}
DECODE_WINDOWS = (128,)


HOST_SPREAD_LIMIT = 0.10  # --host-rounds quiet-socket
UNCONTENDED = 1.15  # --host-rounds quiet-socket-or-uncontended, when no lane meets the quiet-socket rule
MIN_UNCONTENDED = 3
SLOWED = 1.15  # a host-bound process this much slower than its cell's fastest round counts as slowed


OPTION = {"verified": True}  # set in main from the verdict


def expected_cells():
    """{cell: versions} of the default schedule of run_release_compare.sh."""
    cells = {f"layer_B{B}_T{T}_C64_{k}": VERSIONS for B, T in SHAPES for k in ("eager", "graph")}
    if OPTION["verified"]:
        cells.update({f"layeropt_B{B}_T{T}_C64": ("new",) for B, T in SHAPES})
    for w in WINDOWS:
        for B, T in SHAPES:
            cells.update({f"model_W{w}_B{B}_T{T}_{m}": VERSIONS for m in ("eager", "graph")})
            if OPTION["verified"]:
                cells[f"modelopt_W{w}_B{B}_T{T}"] = ("new",)
    cells.update({f"decode_W{w}": VERSIONS for w in DECODE_WINDOWS})
    return cells


def load(out):
    """{cell: {version: [record, ...]}} and {cell: {version: [round, ...]}}."""
    runs, rounds = {}, {}
    for v in VERSIONS:
        for f in sorted(glob.glob(os.path.join(out, v, "*.json"))):
            m = re.fullmatch(r"(.+)_r(\d+)", os.path.basename(f)[:-5])
            cell, rnd = (m.group(1), int(m.group(2))) if m else (os.path.basename(f)[:-5], -1)
            with open(f) as fh:
                runs.setdefault(cell, {}).setdefault(v, []).append(json.load(fh))
            rounds.setdefault(cell, {}).setdefault(v, []).append(rnd)
    return runs, rounds


def option_problems(stats, layers):
    """Where the cuda_graph option's counters show that a timed run did not replay its graphs
    on every call: each SDM layer captured once, replays, and no eager fallback or
    backward-time forward replay."""
    stats = stats or {}
    p = [f"{k}={stats.get(k)} (expected {layers})" for k in ("forward_capture", "backward_capture")
         if stats.get(k) != layers]
    p += [f"no {k}" for k in ("forward_replay", "backward_replay") if not stats.get(k)]
    p += [f"{k}={x}" for k, x in stats.items() if k.startswith(("eager", "stale")) and x]
    return p


def decode_problems(r, graph):
    """The decode checks: contexts; for a run with the CUDA-graph decode step (this version),
    the same tokens and untimed logits as eager decoding and the graph's counters; for an
    eager-only run (the previous release), the same tokens in every repetition."""
    p = []
    for row in (r.get("rows") or []) if not graph else []:
        if row.get("eager_tokens_identical_across_repeats") is not True:
            p.append(f"context {row.get('context')}: the eager repetitions' tokens differ or "
                     f"were not compared")
    for row in (r.get("rows") or []) if graph else []:
        ctx, st = row.get("context"), row.get("graph_stats") or {}
        steps = r["args"]["repeats"] * (r["args"]["warm"] + r["args"]["gen"])
        if not row.get("tokens_identical"):
            p.append(f"context {ctx}: tokens differ between eager and graph decoding")
        if not row.get("untimed_logits_bitwise"):
            p.append(f"context {ctx}: logits of the untimed steps differ")
        if st.get("capture") != 1 or st.get("eager") != 1 or \
                st.get("eager", 0) + st.get("replay", 0) != steps:
            p.append(f"context {ctx}: decode graph stats {st}, expected 1 eager step, 1 capture "
                     f"and {steps - 1} replays")
    want = [int(c) for c in DECODE["contexts"].split(",")]
    if [row.get("context") for row in r.get("rows") or []] != want:
        p.append(f"contexts {[row.get('context') for row in r.get('rows') or []]}, expected {want}")
    return p


def definition_problems(cell, r, version="new"):
    """Where a successful record does not match its cell: benchmark, the arguments the cell
    name encodes, missing or non-finite timing fields, the cuda_graph option's counters, or
    the decode checks."""
    base = cell
    lay = re.fullmatch(r"layer_B(\d+)_T(\d+)_C(\d+)_(eager|graph)", base)
    dec = re.fullmatch(r"decode_W(\d+)", base)
    opt = re.fullmatch(r"layeropt_B(\d+)_T(\d+)_C(\d+)", base)
    mod = (re.fullmatch(r"model_W(\d+)_B(\d+)_T(\d+)_(eager|graph)", base)
           or re.fullmatch(r"modelopt_W(\d+)_B(\d+)_T(\d+)()", base))
    extra = []
    if lay or opt:
        B, T, C = (int(x) for x in (lay or opt).group(1, 2, 3))
        kind = lay.group(4) if lay else "eager"
        bench = "sdm"
        want = {"B": B, "T": T, "C": C, "mode": "train", "sdm_cuda_graph": bool(opt),
                "no_graph": kind == "eager", "no_eager": kind == "graph"}
        timed = (f"{kind} fwd_bwd", f"{kind} fwd", f"{kind} bwd", "peak GiB")
        if opt:
            extra = option_problems(r.get("cuda_graph_stats"), 1)
    elif mod:
        bench, timed = "sdm_gdn", ("fwd_bwd", "fwd", "step", "peak GiB")
        want = {"swa_window": int(mod.group(1)), "batch": int(mod.group(2)),
                "seqlen": int(mod.group(3)), "mode": mod.group(4) or "eager",
                "sdm_cuda_graph": not mod.group(4)}
        if not mod.group(4):
            extra = option_problems(r.get("cuda_graph_stats"), (r.get("args") or {}).get("n_blocks"))
    elif dec:
        w, graph = int(dec.group(1)), version == "new"
        bench, want = "decode", dict(DECODE, swa_window=w)
        timed = tuple(f"{k} tok/s T{c}" for c in DECODE["contexts"].split(",")
                      for k in (("eager", "graph") if graph else ("eager",)))
        extra = decode_problems(r, graph)
        if r.get("swa_window") != w:
            extra.append(f"the model's SWA window is {r.get('swa_window')}, not {w}")
        if r.get("weights") != "loaded":
            extra.append(f"weights {r.get('weights')!r}, not loaded from the shared file")
    else:
        return ["unknown cell"]
    p = [] if r.get("bench") == bench else [f"bench {r.get('bench')!r} is not {bench!r}"]
    args = r.get("args") or {}
    p += [f"{k}={args.get(k)!r} is not {x!r}" for k, x in want.items() if args.get(k) != x]
    vals = medians(r)
    p += [f"{k} missing or not finite" for k in timed
          if not isinstance(vals.get(k), (int, float)) or not math.isfinite(vals[k])]
    return p + extra


def medians(r):
    """{metric: per-process median} of one record (ms; tokens/s for decoding)."""
    out = {}
    if r["bench"] == "sdm":
        for mode in ("eager", "graph"):
            for k, s in (r.get(mode) or {}).items():
                if isinstance(s, dict) and "median" in s and k != "host_issue_ms":
                    out[f"{mode} {k[:-3]}"] = s["median"]
        out["peak GiB"] = r.get("peak_gib")
    elif r["bench"] == "decode":
        for row in r.get("rows") or []:
            out[f"eager tok/s T{row['context']}"] = row.get("eager_tokens_per_s")
            if "graph_tokens_per_s" in row:  # not in the previous release
                out[f"graph tok/s T{row['context']}"] = row["graph_tokens_per_s"]
    else:
        for k, s in r["timing"].items():
            if k != "host_issue_ms":
                out[k[:-3]] = s["median"]
        out["peak GiB"] = r["memory"]["peak_allocated_gib"]
    return out


def validate(runs, rounds, round_ids=None):
    problems = []
    for cell, versions in expected_cells().items():
        recs = []
        for v in versions:
            got = runs.get(cell, {}).get(v, [])
            ok = [r for r in got if r.get("status") == "ok"]
            if len(ok) != rounds or len(got) != rounds:
                problems.append(f"{cell} {v}: {len(ok)} ok of {len(got)} records, {rounds} expected "
                                f"({sorted({str(r.get('status')) for r in got})})")
            ids = sorted((round_ids or {}).get(cell, {}).get(v, []))
            if round_ids is not None and ids != list(range(rounds)):
                problems.append(f"{cell} {v}: rounds {ids}, expected 0..{rounds - 1}")
            problems += list(dict.fromkeys(  # each distinct problem once
                f"{cell} {v}: {x}" for r in ok for x in definition_problems(cell, r, v)))
            for r in ok:
                bad = [k for k, x in medians(r).items()
                       if not (isinstance(x, (int, float)) and math.isfinite(x) and x > 0)]
                if bad:
                    problems.append(f"{cell} {v}: non-finite or non-positive {bad}")
            recs += ok
        for field, get in (
                ("arguments", lambda r: {k: x for k, x in r["args"].items() if k != "out"}),
                ("environment", lambda r: r.get("environment")),
                ("weights", lambda r: r.get("weights_sha256") or r.get("x_layers_weights_sha256")),
                ("harness", lambda r: (r.get("sdm_checkout") or {}).get("harness_sha256"))):
            if len({json.dumps(get(r), sort_keys=True, default=str) for r in recs}) > 1:
                problems.append(f"{cell}: {field} differ between runs")
        if any(r.get("x_layers_weights_sha256_after") not in (None, r.get("x_layers_weights_sha256"))
               for r in recs):
            problems.append(f"{cell}: weights changed during timing")
    extra = sorted(set(runs) - set(expected_cells()))
    if extra:
        problems.append(f"unexpected cells {extra}")
    # One code version per checkout and one harness for the whole run.
    for v in VERSIONS:
        src = [r.get("sdm_checkout") or {} for by_v in runs.values() for r in by_v.get(v, [])]
        for key in ("commit", "code_tree"):
            if len({_ck(s, key) for s in src}) > 1:
                problems.append(f"{v}: several {key} values {sorted({str(_ck(s, key)) for s in src})}")
        if any(_ck(s, "code_status") for s in src):
            problems.append(f"{v}: a modified code directory")
    harness = {(r.get("sdm_checkout") or {}).get("harness_sha256")
               for by_v in runs.values() for recs in by_v.values() for r in recs}
    if len(harness) > 1:
        problems.append(f"several harness hashes {sorted(map(str, harness))}")
    return problems


def verdict_problems(v, runs):
    """The records must be of the two commits and the harness of an EQUIVALENT verdict."""
    if not v:
        return ["no equivalence verdict"]
    p = [] if v.get("verdict") == "EQUIVALENT" else [f"equivalence verdict {v.get('verdict')!r}"]
    want = {"old": (v.get("a") or {}).get("commit"), "new": (v.get("b") or {}).get("commit")}
    for ver in VERSIONS:
        got = {(r.get("sdm_checkout") or {}).get("commit")
               for by_v in runs.values() for r in by_v.get(ver, [])}
        if got - {want[ver]}:
            p.append(f"{ver}: commits {sorted(map(str, got))}, the verdict's is {want[ver]}")
    harness = {(r.get("sdm_checkout") or {}).get("harness_sha256")
               for by_v in runs.values() for recs in by_v.values() for r in recs}
    if harness - {v.get("harness_sha256")}:
        p.append(f"harness {sorted(map(str, harness))}, the verdict's is {v.get('harness_sha256')}")
    opt = v.get("cuda_graph_option") or {}
    src = opt.get("checkout") or {}
    if opt.get("verdict") == "VERIFIED" and (src.get("commit") != want["new"]
                                             or src.get("harness_sha256") != v.get("harness_sha256")):
        p.append("the cuda_graph option's verdict is not of the new commit and the verdict's harness")
    return p


def is_host(cell):
    """Host-bound cells: eager, the cuda_graph option, decoding."""
    return cell.endswith("_eager") or "opt_" in cell or cell.startswith("decode")


def host_metrics(cell):
    if cell.startswith("decode"):
        return [f"eager tok/s T{c}" for c in DECODE["contexts"].split(",")]
    return ["step"] if cell.startswith("model") else ["eager fwd_bwd"]


def record_name(r):
    return r["args"]["out"][:-5]


def placements(out):
    """{(record name, version): lane} from run_release_compare.sh's command lists and lane logs."""
    labels = {}
    for phase, prefix in (("graph", "graph"), ("host-bound", "eager")):
        lanes = {}
        for f in glob.glob(os.path.join(out, "logs", f"{prefix}_lane*.log")):
            with open(f) as fh:
                m = re.match(r"lane (\d+): gpu (\S+), (.*)", fh.readline().strip())
            if m:
                lanes[int(m.group(1))] = {"gpu": m.group(2), "pin": m.group(3)}
        path = os.path.join(out, "logs", f"{prefix}_commands.txt")
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    k, name, v = line.split()[:3]
                    labels[(name, v)] = {"phase": phase, "lane": int(k), **lanes.get(int(k), {})}
    return labels


def select_host_rounds(runs, labels, how):
    """The records to merge, and the lane statistics behind the selection."""
    if how == "all":
        return runs, None, []
    best = {}
    for cell, by_v in runs.items():
        for v, recs in by_v.items():
            if is_host(cell):
                for k in host_metrics(cell):
                    xs = [medians(r)[k] for r in recs]
                    best[(cell, v, k)] = min(xs) if k.endswith("fwd_bwd") or k == "step" else max(xs)
    lanes, problems = {}, []
    for cell, by_v in runs.items():
        if not is_host(cell):
            continue
        for v, recs in by_v.items():
            for r in recs:
                lab = labels.get((record_name(r), v))
                if lab is None:
                    problems.append(f"{cell} {v}: no lane for {record_name(r)}")
                    continue
                d = lanes.setdefault(lab["lane"], {"gpu": lab.get("gpu"), "pin": lab.get("pin"),
                                                   "values": {}, "processes": 0, "slowed": []})
                d["processes"] += 1
                worst = 1.0
                for k in host_metrics(cell):
                    x = medians(r)[k]
                    d["values"].setdefault((cell, v, k), []).append(x)
                    b = best[(cell, v, k)]
                    worst = max(worst, x / b if k.endswith("fwd_bwd") or k == "step" else b / x)
                if worst > SLOWED:
                    d["slowed"].append(worst)
    info = {"rule": f"host-bound cells (eager, cuda_graph option, decoding) use the rounds of the host-bound "
                    f"lanes (one per CPU socket) whose cells all spread by at most {HOST_SPREAD_LIMIT:.0%} over "
                    f"that lane's rounds; graph-only cells use every round",
            "limit": HOST_SPREAD_LIMIT, "slowed_threshold": SLOWED, "lanes": {}}
    for k, d in sorted(lanes.items()):
        spreads = {key: (max(x) - min(x)) / statistics.median(x) for key, x in d["values"].items()}
        worst = max(spreads.items(), key=lambda kv: kv[1])
        info["lanes"][k] = {"gpu": d["gpu"], "pin": d["pin"], "processes": d["processes"],
                            "max_spread": worst[1], "max_spread_cell": " ".join(worst[0]),
                            "slowed": len(d["slowed"]),
                            "slowdown_range": [min(d["slowed"]), max(d["slowed"])] if d["slowed"] else None}
    chosen = [k for k, d in info["lanes"].items() if d["max_spread"] <= HOST_SPREAD_LIMIT]
    info["selected_lanes"] = chosen
    if not chosen:
        problems.append(f"no host-bound lane within {HOST_SPREAD_LIMIT:.0%}: " + ", ".join(
            f"lane {k} {d['max_spread']:.0%}" for k, d in info["lanes"].items()))
        return runs, info, problems
    sel = {}
    for cell, by_v in runs.items():
        for v, recs in by_v.items():
            keep = [r for r in recs if not is_host(cell)
                    or (labels.get((record_name(r), v)) or {}).get("lane") in chosen]
            if not keep:
                problems.append(f"{cell} {v}: no record on lanes {chosen}")
            sel.setdefault(cell, {})[v] = keep
    info["rounds_used"] = {cell: sorted(int(record_name(r).rsplit("_r", 1)[1]) for r in by_v[next(iter(by_v))])
                           for cell, by_v in sel.items() if is_host(cell)}
    return sel, info, problems


def faster(k):
    """Whether smaller values of metric k are faster (times; not tokens/s)."""
    return "tok/s" not in k


def merge_uncontended(runs):
    """merge() with the uncontended rule for the host-bound cells, and the noisy cells."""
    merged, noisy, kept_counts = merge({c: by_v for c, by_v in runs.items() if not is_host(c)}), [], {}
    for cell, by_v in runs.items():
        if not is_host(cell):
            continue
        for v, recs in by_v.items():
            vals = [medians(r) for r in recs]
            primary = host_metrics(cell) if not cell.startswith("decode") else sorted(
                k for k in vals[0] if "tok/s" in k)
            out = {"n": len(recs), "median": {}, "spread": {}, "kept": {}}
            for k in primary:
                xs = [x[k] for x in vals]
                best = min(xs) if faster(k) else max(xs)
                keep = [i for i, x in enumerate(xs) if (x <= UNCONTENDED * best if faster(k)
                                                        else x >= best / UNCONTENDED)]
                out["kept"][k] = len(keep)
                kept_counts[f"{cell} {v} {k}"] = [len(keep), len(xs)]
                if len(keep) < MIN_UNCONTENDED:
                    noisy.append(f"{cell} {v} {k}")
                    keep = list(range(len(xs)))
                use = keep
                keys = [k] if cell.startswith("decode") else list(vals[0])  # a layer or model cell's metrics
                for kk in keys:
                    ys = [vals[i][kk] for i in use]
                    out["median"][kk] = statistics.median(ys)
                    out["spread"][kk] = (max(ys) - min(ys)) / statistics.median(ys) if len(ys) > 1 else None
            merged.setdefault(cell, {})[v] = out
    return merged, noisy, kept_counts


def harness_hashes(runs):
    """The harness hash the records were measured with, and that of the files now."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from provenance import HARNESS, harness_sha256  # this directory

    files = {}
    for name in HARNESS:
        with open(os.path.join(here, name), "rb") as f:
            files[name] = hashlib.sha256(f.read()).hexdigest()
    measured = sorted({(r.get("sdm_checkout") or {}).get("harness_sha256") or ""
                       for by_v in runs.values() for recs in by_v.values() for r in recs})
    now = harness_sha256()
    out = {"measurement": measured[0] if len(measured) == 1 else measured, "now": now,
           "files_now": files}
    if measured != [now]:
        out["note"] = ("the harness changed after the measurement; files_now has the sha256 of "
                       "each harness file as analysed")
    return out


def code_versions(runs):
    out = {}
    for v in VERSIONS:
        for by_v in runs.values():
            for r in by_v.get(v, []):
                s = r.get("sdm_checkout") or {}
                out[v] = {"commit": s.get("commit"), "code_tree": _ck(s, "code_tree")}
                out["harness_sha256"] = s.get("harness_sha256")
    return out


def merge(runs):
    merged = {}
    for cell, by_v in runs.items():
        for v, recs in by_v.items():
            vals = {}
            for r in recs:
                for k, x in medians(r).items():
                    vals.setdefault(k, []).append(x)
            merged.setdefault(cell, {})[v] = {
                "n": len(recs), "median": {k: statistics.median(x) for k, x in vals.items()},
                "spread": {k: (max(x) - min(x)) / statistics.median(x) if len(x) > 1 else None
                           for k, x in vals.items()}}
    return merged


def paired_speedups(runs):
    """Per round, the eager speedup of this version over the previous release (both versions
    of a cell ran back to back on one GPU): previous release / this version of the times,
    this version / previous release of the decoding tokens/s. {row kind: [min, max]} and the
    values per cell."""
    cells = {"layer": [(f"layer_B{B}_T{T}_C64_eager", "eager fwd_bwd") for B, T in SHAPES],
             "model": [(f"model_W{w}_B{B}_T{T}_eager", "step") for w in WINDOWS
                       for B, T in SHAPES],
             "decode": [(f"decode_W{w}", f"eager tok/s T{c}") for w in DECODE_WINDOWS
                        for c in DECODE["contexts"].split(",")]}
    out = {"per_cell": {}}
    for kind, lst in cells.items():
        allv = []
        for cell, key in lst:
            by_round = {}
            for v in VERSIONS:
                for r in runs[cell][v]:
                    rd = re.search(r"_r(\d+)\.json$", r["args"]["out"]).group(1)
                    by_round.setdefault(rd, {})[v] = medians(r)[key]
            vals = [(x["new"] / x["old"]) if kind == "decode" else (x["old"] / x["new"])
                    for _, x in sorted(by_round.items()) if len(x) == 2]
            out["per_cell"][cell if kind != "decode" else f"{cell} {key}"] = vals
            allv += vals
        out[kind] = [min(allv), max(allv)]
    return out


def get(m, cell, v, key):
    return m[cell][v]["median"][key]


def ms(x):
    return f"{x:.1f}" if x >= 10 else f"{x:.2f}"


def pair(o, n):
    return f"{ms(o)} → {ms(n)} (**{o / n:.2f}x**)"


def ktok(tokens, t_ms):
    return f"{tokens / (t_ms / 1e3) / 1e3:.0f}k"


def layer_rows(m, mode):
    lines = ["| B x T | forward ms | backward ms | fwd + bwd ms | tokens/s (fwd + bwd) |",
             "|---|---|---|---|---|"]
    for B, T in SHAPES:
        c = f"layer_B{B}_T{T}_C64_{mode}"
        o = {k: get(m, c, "old", f"{mode} {k}") for k in ("fwd", "bwd", "fwd_bwd")}
        n = {k: get(m, c, "new", f"{mode} {k}") for k in ("fwd", "bwd", "fwd_bwd")}
        lines.append(f"| {B} x {T // 1024}K | {pair(o['fwd'], n['fwd'])} | {pair(o['bwd'], n['bwd'])} | "
                     f"{pair(o['fwd_bwd'], n['fwd_bwd'])} | {ktok(B * T, o['fwd_bwd'])} → "
                     f"{ktok(B * T, n['fwd_bwd'])} |")
    return lines


def main_rows(m):
    """The headline table (the plot's values): per cell, the previous release (eager, its
    only mode) against this version eager and with its cuda_graph option (once verified)."""
    opt = OPTION["verified"]
    lines = ["| | previous release, eager | this version, eager | "
             + ("this version, `cuda_graph=True` |" if opt else ""),
             "|---|---|---|" + ("---|" if opt else "")]

    def cell(o, n):
        return f"{ms(n)} (**{o / n:.2f}x**)"

    for B, T in SHAPES:
        e = f"layer_B{B}_T{T}_C64_eager"
        o = get(m, e, "old", "eager fwd_bwd")
        lines.append(f"| layer, {B} x {T // 1024}K tokens, forward + backward | {ms(o)} | "
                     f"{cell(o, get(m, e, 'new', 'eager fwd_bwd'))} |"
                     + (f" {cell(o, get(m, f'layeropt_B{B}_T{T}_C64', 'new', 'eager fwd_bwd'))} |" if opt else ""))
    for w in WINDOWS:
        for B, T in SHAPES:
            c = f"W{w}_B{B}_T{T}"
            o = get(m, f"model_{c}_eager", "old", "step")
            lines.append(f"| model, {B} x {T // 1024}K tokens, training step | "
                         f"{ms(o)} | {cell(o, get(m, f'model_{c}_eager', 'new', 'step'))} |"
                         + (f" {cell(o, get(m, f'modelopt_{c}', 'new', 'step'))} |" if opt else ""))
    return lines


def model_detail_rows(m):
    lines = ["| SWA window | B x T | run | forward + backward ms | step ms | tokens/s/GPU | peak GiB |",
             "|---|---|---|---|---|---|---|"]
    for w in WINDOWS:
        for B, T in SHAPES:
            c = f"W{w}_B{B}_T{T}"
            runs = [("previous release, eager", f"model_{c}_eager", "old"),
                    ("this version, eager", f"model_{c}_eager", "new"),
                    ("this version, `cuda_graph=True`", f"modelopt_{c}", "new"),
                    ("previous release, whole-step graph", f"model_{c}_graph", "old"),
                    ("this version, whole-step graph", f"model_{c}_graph", "new")]
            for label, cell, v in [r for r in runs if OPTION["verified"] or "opt_" not in r[1]]:
                st = get(m, cell, v, "step")
                lines.append(f"| {w} | {B} x {T // 1024}K | {label} | "
                             f"{ms(get(m, cell, v, 'fwd_bwd'))} | {ms(st)} | {ktok(B * T, st)} | "
                             f"{get(m, cell, v, 'peak GiB'):.0f} |")
    return lines


def decode_rows(m, w=128):
    """Tokens/s (speedup over the previous release, which decodes eagerly only)."""
    lines = ["| context | previous release, eager | this version, eager | "
             "this version, CUDA-graph decode step |", "|---|---|---|---|"]
    for c in DECODE["contexts"].split(","):
        o = get(m, f"decode_W{w}", "old", f"eager tok/s T{c}")
        e = get(m, f"decode_W{w}", "new", f"eager tok/s T{c}")
        g = get(m, f"decode_W{w}", "new", f"graph tok/s T{c}")
        lines.append(f"| {int(c) // 1024}K | {o:.0f} | {e:.0f} (**{e / o:.2f}x**) | "
                     f"{g:.0f} (**{g / o:.1f}x**) |")
    return lines


def spread_by_kind(m):
    """Largest spread per kind: eager / option / graph, layer / model / decode."""
    kinds = {}
    for cell, by_v in m.items():
        for v, d in by_v.items():
            for k, s in d["spread"].items():
                if s is None or "GiB" in k:
                    continue
                part = cell.split("_")[0].replace("opt", "")
                mode = ("option" if "opt_" in cell else
                        "graph" if (k.startswith("graph") or cell.endswith("_graph")) else "eager")
                key = f"{part} {mode}"
                if s > kinds.get(key, (0,))[0]:
                    kinds[key] = (s, cell, v, k)
    return kinds


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--host-rounds", choices=("all", "quiet-socket", "quiet-socket-or-uncontended"), default="all",
                   help="rounds merged for the host-bound cells (see above)")
    p.add_argument("--verdict", help="equivalence verdict to bundle (default OUT/equivalence/verdict.json)")
    p.add_argument("--attach", action="append", default=[], metavar="KEY=FILE",
                   help="store FILE's JSON under attachments.KEY of --json")
    p.add_argument("--json")
    a = p.parse_args(argv)
    verdict = a.verdict or os.path.join(a.out, "equivalence", "verdict.json")
    v = json.load(open(verdict)) if os.path.exists(verdict) else None
    # The cuda_graph option's cells are expected (and shown) only when its check passed.
    OPTION["verified"] = bool(v) and v.get("cuda_graph_option", {}).get("verdict") == "VERIFIED"
    runs, round_ids = load(a.out)
    problems = verdict_problems(v, runs) + validate(runs, a.rounds, round_ids)
    labels = placements(a.out)
    for by_v in runs.values():
        for ver, recs in by_v.items():
            for r in recs:
                r["placement"] = labels.get((record_name(r), ver))
    how = "quiet-socket" if a.host_rounds == "quiet-socket-or-uncontended" else a.host_rounds
    sel, host_info, sel_problems = select_host_rounds(runs, labels, how)
    uncontended = (a.host_rounds == "quiet-socket-or-uncontended" and host_info is not None
                   and not host_info["selected_lanes"])
    if uncontended:  # no lane met the quiet-socket rule: the uncontended rule
        sel_problems = [x for x in sel_problems if not x.startswith("no host-bound lane within")]
        m, noisy, kept = merge_uncontended(runs)
        host_info.update(rule_applied="uncontended", uncontended_factor=UNCONTENDED,
                         min_processes=MIN_UNCONTENDED, noisy=noisy, kept=kept, rule_uncontended=(
                             f"no host-bound lane met the {HOST_SPREAD_LIMIT:.0%} rule: per host-bound cell, "
                             f"version and timed metric, the median over both lanes' processes within "
                             f"{UNCONTENDED}x of the cell's fastest process (at least {MIN_UNCONTENDED})"))
    else:
        m = merge(sel)
        if host_info is not None:
            host_info["rule_applied"] = "quiet-socket"
    problems += sel_problems
    doc = {"merged": m, "problems": problems, "processes": runs, "code": code_versions(runs),
           "harness": harness_hashes(runs), "host_rounds": a.host_rounds,
           "cuda_graph_option_verified": OPTION["verified"]}
    if host_info:
        doc["host_selection"] = host_info
    if problems:
        print("INCOMPLETE OR MISMATCHED RUN, no tables:\n" + "\n".join(f"- {x}" for x in problems))
    else:
        kinds = spread_by_kind(m)
        sections = {
            "main": main_rows(m), "eager_layer": layer_rows(m, "eager"),
            "graph_layer": layer_rows(m, "graph"), "model_detail": model_detail_rows(m),
            "decode": decode_rows(m, 128)}
        paired = paired_speedups(sel)
        text = "\n\n".join(f"### {k}\n\n" + "\n".join(v) for k, v in sections.items())
        text += "\n\n### spread\n\n" + "\n".join(
            f"- {k}: {100 * s:.1f}% ({c}, {v}, {key})" for k, (s, c, v, key) in sorted(kinds.items()))
        text += "\n\n### per-round eager speedups\n\n" + "\n".join(
            [f"- {k}: {lo:.2f}x to {hi:.2f}x" for k, (lo, hi) in
             ((k, paired[k]) for k in ("layer", "model", "decode"))]
            + [f"- {c}: " + ", ".join(f"{x:.2f}" for x in vals)
               for c, vals in paired["per_cell"].items()])
        print(text)
        doc.update(tables_markdown=text, sections={k: "\n".join(v) for k, v in sections.items()},
                   spread={k: {"max": s, "cell": c, "version": v, "metric": key}
                           for k, (s, c, v, key) in kinds.items()},
                   paired_eager_speedups=paired)
        if host_info:  # the same over every round, so that the selection can be checked
            m_all = merge(runs)
            doc.update(merged_all_rounds=m_all,
                       sections_all_rounds={"main": "\n".join(main_rows(m_all)),
                                            "decode": "\n".join(decode_rows(m_all, 128))},
                       spread_all_rounds={k: {"max": s, "cell": c, "version": v, "metric": key}
                                          for k, (s, c, v, key) in spread_by_kind(m_all).items()},
                       paired_eager_speedups_all_rounds=paired_speedups(runs))
            print("\n### host-bound rounds\n\n" + "\n".join(
                f"- lane {k} (gpu {d['gpu']}, {d['pin']}): {d['processes']} processes, max spread "
                f"{d['max_spread']:.1%} ({d['max_spread_cell']}), {d['slowed']} slowed by more than "
                f"{SLOWED}x" + (f" ({d['slowdown_range'][0]:.2f}-{d['slowdown_range'][1]:.2f}x)"
                                if d["slowdown_range"] else "")
                for k, d in host_info["lanes"].items()) + f"\n- selected lanes: {host_info['selected_lanes']}"
                + (f"\n- {host_info['rule_uncontended']}; noisy: {host_info['noisy'] or 'none'}"
                   if host_info.get("rule_applied") == "uncontended" else ""))
    if a.json:
        doc["equivalence"] = v
        for item in a.attach:
            key, _, path = item.partition("=")
            with open(path) as f:
                doc.setdefault("attachments", {})[key] = json.load(f)
        with open(a.json, "w") as f:
            json.dump(doc, f, indent=1, default=str)
        print(f"wrote {a.json}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
