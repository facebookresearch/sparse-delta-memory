# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Mutation check of the determinism tests: does the default tier of test_determinism.py see
a nondeterministic accumulation put back into the kernels?

Each mutation is one or more anchored text edits of a scratch copy of the code tree (sparse_delta_memory/,
lingua/, apps/, tests/). An anchor must occur exactly once in its file, and each target statement a
mutation declares must occur as often as the mutation edits it, else the mutation is INVALID
(never SURVIVED); tests/test_det_mutations_manifest.py checks both in the ordinary test suite.
The default tier then runs on the copy in a fresh pytest process (its own
PYTHONPATH, python -B, a Triton cache in the temporary directory shared by the runs of this
harness: Triton keys its cache on the kernel source, and the CUDA extensions are
content-addressed). The unmutated copy runs first through the same path, with the same
extra targets as any mutation: it must pass every test (no failure, no skip; a strict xfail
counts as passing, its XPASS as a failure; the one non-strict xfail, whose equality depends on
the inputs, may pass or xfail), else no verdict is given.

Verdicts per mutation:
  DETECTED  determinism assertions failed ("not bitwise reproducible", or a fresh process
            produced other bytes), and nothing else failed
  SURVIVED  the run passed exactly the control's tests
  NO-EFFECT a mutation expected to be caught survived without moving any fingerprinted
            value (its edit is masked; it tests nothing).
  BROKEN    anything else: an exception, a compile error, a failed premise, a timeout, a
            different set of tests, or determinism failures mixed with other failures (the
            detections are listed): no verdict
  INVALID   an anchor did not match exactly once

Each run also hashes the outputs of test_determinism's fresh-process cases (its --digests
--wide worker: op cases on bf16 and fp32 banks with none / fp8 / int4 snapshots, per-write
gates, chunks 16 to 256 and head_dim 64 to 1024, Phase 1 alone, a layer) and records
whether the mutation changed any value of the unmutated tree ("changed"; only when two
control runs give the same digests). Mutations marked ``expect="SURVIVED"`` are negative
controls: an order change that stays deterministic must change the arithmetic (else it
tests nothing) and still pass, so the tests do not pin results to one summation order.
A deterministic mutation that computes wrong values must fail the float64 value gates
(``extra`` targets among ORACLE_TESTS) and nothing else: "NUMERICS". The
"selftest_*" mutations check the harness on any tree.

Usage:
  python -B tests/det_mutation_harness.py --src ROOT --work DIR --mutations FILE.json
                                          [--only REGEX] [--list] [--keep]
  python -B tests/det_mutation_harness.py --src ROOT --work DIR --selftest-only
                                          [--pytest-args "..."]
ROOT may be read-only (a frozen snapshot): the copies are made writable.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from typing import NamedTuple

TIER = ["-m", "not slow", "tests/test_determinism.py"]
DETECTION_MARKERS = ("not bitwise reproducible", "tensors that differ from this process")
# The float64 value gates whose own assertion failures count as numerics (a test named as a
# mutation's extra target; any other failure there, or a wrapped exception, is BROKEN).
ORACLE_TESTS = (
    "tests/test_autograd_reference.py::test_long_slot_groups_match_the_float64_reference",
    "tests/test_f64_oracle.py::test_fp32_forward_and_every_gradient_match_the_oracle")
EXCEPTION_MARKERS = ("Error", "Exception", "Traceback", "status")
# The only expected failures of the tier (xfails that expect one exception type, as
# test_the_default_tier_covers_every_axis checks): any other xfail is a problem. The strict
# ones must fail in the control, and their outcome may not change between the control and a
# mutant; XFAIL_INPUTS (not strict: equality depends on the inputs) may pass or xfail in
# either.
_PIN = "tests.test_determinism::test_op_saving_forward_is_bitwise_the_no_grad_forward"
XFAIL_STRICT = (f"{_PIN}[fp32_fp8]", "tests.test_determinism::"
                "test_cuda_graph_eager_fallback_with_fp8_snapshots_is_the_documented_exception")
XFAIL_INPUTS = (f"{_PIN}[bf16_fp8]",)
XFAIL_ALLOWED = XFAIL_STRICT + XFAIL_INPUTS


def _own(tid):
    """A test of the tier or of the float64 value gates (others are another module's)."""
    return tid.startswith("tests.test_determinism::") or any(
        _node(tid).startswith(x) for x in ORACLE_TESTS)
# A value-gate failure names a deviation ("grad_v=3.10e-02", "forward 2.02e-02"); a failed
# premise of an oracle test (distinct slots, the IEEE fixture, ...) does not.
VALUE_MARKER = re.compile(r"[\w.]+[= ]\d\.\d+e[-+]\d+")
EXPECTS = ("DETECTED", "SURVIVED", "NUMERICS")
COPIED = ("sparse_delta_memory", "lingua", "apps", "tests")


class Edit(NamedTuple):
    path: str          # relative to the tree root
    anchor: str        # literal text, or a regex when ``regex``
    replacement: str
    regex: bool = False


class Target(NamedTuple):
    """A statement a mutation edits, counted on its own: ``pattern`` (a regex) occurs exactly
    ``count`` times in ``path`` (in top-level function ``within`` when given). A one-match
    anchor alone does not establish that every copy of a target statement is edited (a new
    branch can duplicate it, and an anchor that starts at the function's def stops at the
    first copy)."""
    path: str
    pattern: str
    count: int
    within: str = ""


class Mutation(NamedTuple):
    name: str
    what: str
    edits: tuple
    expect: str = "DETECTED"   # DETECTED, SURVIVED or NUMERICS
    extra: tuple = ()          # more pytest targets (e.g. the float64 tests)
    changes: object = None     # must the outputs change (True / False / None: no check)
    targets: tuple = ()        # Target statements, checked before the edits apply


MEMORY_OPS = "sparse_delta_memory/memory_ops.py"
FORWARD = "sparse_delta_memory/forward.py"
_UNSTAGE = "    readings = mo.unstage_time_major(readings, P, T, C, dtype, steps, out)\n"
SELFTESTS = [
    Mutation("selftest_noise", "harness check: random noise on the readings (any tree)",
             (Edit(FORWARD, _UNSTAGE,
                   _UNSTAGE + "    readings.add_(torch.randn_like(readings), alpha=1e-3)\n"),)),
    Mutation("selftest_noop", "harness check: a comment line (any tree)",
             (Edit(MEMORY_OPS, "MIN_CHUNK, MAX_CHUNK = 16, 256\n",
                   "MIN_CHUNK, MAX_CHUNK = 16, 256  # mutation harness no-op\n"),),
             expect="SURVIVED", changes=False),
]

# The kernel mutations depend on the deterministic kernels' code: they are loaded from a JSON
# file (--mutations) pinned to the tree under test. Format: a list of {"name", "what",
# "expect", "extra": [...], "changes": true|false|null, "edits": [{"path", "anchor",
# "replacement", "regex"}], "targets": [{"path", "pattern", "count", "within"}]}.


def load_mutations(path):
    out = []
    for m in json.load(open(path)):
        mut = Mutation(m["name"], m["what"],
                       tuple(Edit(e["path"], e["anchor"], e["replacement"],
                                  e.get("regex", False)) for e in m["edits"]),
                       m.get("expect", "DETECTED"), tuple(m.get("extra", ())), m.get("changes"),
                       tuple(Target(t["path"], t["pattern"], int(t["count"]), t.get("within", ""))
                             for t in m.get("targets", ())))
        if not mut.edits or mut.expect not in EXPECTS:
            raise ValueError(f"{mut.name}: needs edits and an expect in {EXPECTS}")
        if not mut.targets or any(t.count < 1 for t in mut.targets):
            raise ValueError(f"{mut.name}: needs its target statements (targets, count >= 1)")
        if any(x not in ORACLE_TESTS for x in mut.extra):
            raise ValueError(f"{mut.name}: extra targets must be among ORACLE_TESTS")
        if mut.expect == "SURVIVED" and (mut.changes is not True or not mut.extra):
            raise ValueError(f"{mut.name}: a negative control must change values (changes: "
                             "true) and pass the float64 value gates (extra), else it tests "
                             "nothing")
        if mut.expect == "NUMERICS" and not mut.extra:
            raise ValueError(f"{mut.name}: NUMERICS needs float64 value gates (extra)")
        out.append(mut)
    names = [m.name for m in out]
    if len(set(names)) != len(names) or set(names) & {m.name for m in SELFTESTS}:
        raise ValueError(f"duplicate mutation names: {names}")
    return out


def make_copy(src, dst):
    """Copies the code subtrees of ``src`` to ``dst`` and makes every file writable (a copy
    of a read-only snapshot keeps read-only directories otherwise)."""
    if os.path.exists(dst):
        _make_writable(dst)
        shutil.rmtree(dst)
    os.makedirs(dst)
    for sub in COPIED:
        shutil.copytree(os.path.join(src, sub), os.path.join(dst, sub),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    _make_writable(dst)


def tree_digest(root):
    """SHA-256 over the paths and contents of the copied subtrees of ``root``."""
    h = hashlib.sha256()
    for sub in COPIED:
        for d, dirs, files in sorted(os.walk(os.path.join(root, sub))):
            dirs.sort()
            for f in sorted(files):
                p = os.path.join(d, f)
                h.update(os.path.relpath(p, root).encode() + b"\0")
                with open(p, "rb") as fh:
                    h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


def _make_writable(root):
    for d, dirs, files in os.walk(root):
        os.chmod(d, os.stat(d).st_mode | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR)
        for f in files:
            p = os.path.join(d, f)
            if not os.path.islink(p):
                os.chmod(p, os.stat(p).st_mode | stat.S_IWUSR | stat.S_IRUSR)


def _scope(text, within):
    """The source of top-level function ``within`` (its def to the next top-level line), the
    whole text when ``within`` is empty, or None when there is no such function."""
    if not within:
        return text
    m = re.search(rf"^def {re.escape(within)}\(", text, re.M)
    if not m:
        return None
    body = text.find("\n", m.end()) + 1 or len(text)  # the lines after the def's first line
    end = re.compile(r"^[^\s)#]", re.M).search(text, body)
    return text[m.start():end.start() if end else len(text)]


def check_targets(root, targets):
    """None, or why the target statements do not match their declared counts (INVALID)."""
    for t in targets:
        scope = _scope(open(os.path.join(root, t.path)).read(), t.within)
        if scope is None:
            return f"{t.path}: no function {t.within}"
        n = len(re.findall(t.pattern, scope, flags=re.M))
        if n != t.count:
            where = f" in {t.within}" if t.within else ""
            return f"{t.path}{where}: {n} target statements, the mutation is written for " \
                   f"{t.count}: {t.pattern[:100]!r}"
    return None


def apply_edits(root, edits):
    """Applies ``edits``; returns None, or why the mutation is INVALID."""
    for e in edits:
        path = os.path.join(root, e.path)
        text = open(path).read()
        if e.regex:
            new, n = re.subn(e.anchor, e.replacement, text, flags=re.M)
        else:
            n = text.count(e.anchor)
            new = text.replace(e.anchor, e.replacement)
        if n != 1:
            return f"{e.path}: anchor matched {n} times: {e.anchor[:120]!r}"
        if new == text:
            return f"{e.path}: the edit changes nothing"
        with open(path, "w") as f:
            f.write(new)
    return None


TRITON_CACHE = os.path.join(tempfile.gettempdir(), f"mutation_harness_triton_{os.getpid()}")


def _env(root, *extra_paths):
    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)  # nothing outside this harness selects the tests
    env.update(PYTHONPATH=os.pathsep.join((root,) + extra_paths), PYTHONDONTWRITEBYTECODE="1",
               TRITON_CACHE_DIR=TRITON_CACHE)
    return env


def _run(cmd, root, env, log, timeout):
    """Runs ``cmd`` in its own process group; kills the whole group on timeout. Returns the
    exit code, or "timeout"."""
    with open(log, "w") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        # Its own process group: a timeout kills pytest and everything it started (the
        # fresh-process workers stay in pytest's group).
        proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = "timeout"
        # Whatever is left of the group (a worker or compiler pytest did not reap) goes too.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return rc


def check_import_path(root):
    """The copy, not another tree or an installed package, provides ``sparse_delta_memory`` in the runs."""
    r = subprocess.run([sys.executable, "-B", "-c",
                        "import sparse_delta_memory.memory_ops as m; print(m.__file__)"],
                       cwd=root, env=_env(root), capture_output=True, text=True, timeout=600)
    got = r.stdout.strip()
    if r.returncode or not os.path.realpath(got).startswith(os.path.realpath(root) + os.sep):
        raise RuntimeError(f"sparse_delta_memory imports from {got!r}, not the copy {root} "
                           f"(rc={r.returncode}): {r.stderr[-1000:]}")


def run_tier(root, work, name, targets, timeout):
    """Runs pytest on the copy; returns {test id: (outcome, message)}, rc and the log."""
    xml = os.path.join(work, f"{name}.xml")
    log = os.path.join(work, f"{name}.log")
    for stale in (xml, log):
        if os.path.exists(stale):
            os.remove(stale)
    cmd = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rfEs",
           f"--junitxml={xml}", *targets]
    t0 = time.time()
    rc = _run(cmd, root, _env(root), log, timeout)
    results, dupes = {}, []
    if os.path.exists(xml):
        for tc in ET.parse(xml).getroot().iter("testcase"):
            tid = f"{tc.get('classname')}::{tc.get('name')}"
            # The message only: the traceback text quotes source lines (labels, the
            # "not bitwise reproducible" raise), which says nothing about this failure.
            found = [("xfailed" if el.get("type") == "pytest.xfail" else el.tag,
                      el.get("message") or "")
                     for el in tc if el.tag in ("failure", "error", "skipped")]
            if tid in results:
                dupes.append(tid)
            if not found:
                results[tid] = ("passed", "")
            elif len(found) == 1:
                results[tid] = found[0]
            else:  # e.g. a failure and a teardown error: never a clean outcome
                results[tid] = ("error", " | ".join(f"{k}: {m}" for k, m in found))
    return {"rc": rc, "seconds": round(time.time() - t0, 1), "log": log, "tests": results,
            "duplicates": dupes}


def digests(root, work, name, timeout):
    """The fresh-process digests of the tree (test_determinism's worker), or None."""
    out = os.path.join(work, f"{name}.digests.json")
    if os.path.exists(out):
        os.remove(out)
    cmd = [sys.executable, "-B", "-c", "import test_determinism as m; m.main()",
           "--digests", out, "--wide"]
    rc = _run(cmd, root, _env(root, os.path.join(root, "tests")),
              os.path.join(work, f"{name}.digests.log"), timeout)
    return json.load(open(out)) if rc == 0 and os.path.exists(out) else None


def _node(tid):
    """'tests.test_x::test_y[p]' -> 'tests/test_x.py::test_y[p]' (pytest's node id)."""
    mod, _, name = tid.partition("::")
    return mod.replace(".", "/") + ".py::" + name


def collect(root, work, targets, timeout):
    """The node ids the tier selects in the unmutated tree (pytest --collect-only)."""
    log = os.path.join(work, "collect.log")
    rc = _run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "--collect-only", *targets], root, _env(root), log, timeout)
    ids = [ln.strip() for ln in open(log) if "::" in ln and not ln.startswith(sys.executable)]
    if rc != 0 or not ids:
        raise RuntimeError(f"collection failed (rc={rc}), see {log}")
    return ids


def classify(run):
    """(determinism failures, numerics failures, premise failures, other problems)."""
    det, num, prem, other = [], [], [], []
    for t, (o, m) in sorted(run["tests"].items()):
        if o == "xfailed":
            # Another module's expected failure (a custom tier) is its own; the transitions
            # between the control and a mutant are checked in ``verdict``.
            if (_own(t) and t not in XFAIL_ALLOWED) or m.lstrip().startswith("[NOTRUN]"):
                other.append(f"{t} xfailed (not an allowed expected failure, or not run)")
            continue  # the strict xfail's XPASS is a failure, not this
        if o == "skipped":
            other.append(f"{t} skipped")
        elif o in ("failure", "error"):
            assertion = m.lstrip().startswith("AssertionError") and o == "failure"
            if assertion and "premise" in m:
                prem.append(t)
            elif (assertion and "test_determinism" in t
                  and any(k in m for k in DETECTION_MARKERS)):
                det.append(t)
            elif any(_node(t).startswith(x) for x in ORACLE_TESTS) and assertion and any(
                    k in m for k in DETECTION_MARKERS):
                continue  # an oracle test's own repeat check: not the tier under evaluation
            elif (assertion and any(_node(t).startswith(x) for x in ORACLE_TESTS)
                  and not any(k in m.split("\n", 1)[0][len("AssertionError"):]
                              for k in EXCEPTION_MARKERS)
                  and VALUE_MARKER.search(m.split("\n", 1)[0])):
                num.append(t)
            else:
                other.append(f"{t} {o}: {m.strip()[:160]}")
    if run["rc"] not in (0, 1):
        other.append(f"rc={run['rc']}")
    other += [f"{t} reported twice" for t in run.get("duplicates", ())]
    return det, num, prem, other


def verdict(run, control):
    """DETECTED / NUMERICS / SURVIVED only for a run that completed exactly the control's
    tests with nothing but determinism (or float64-oracle) assertions failing."""
    det, num, prem, other = classify(run)
    missing = sorted(set(control["tests"]) - set(run["tests"]))
    unexpected = sorted(set(run["tests"]) - set(control["tests"]))
    # A test the control ran to completion may not turn into an expected failure, nor the
    # control's expected failure pass (a failing one is judged like any failure).
    both = (set(run["tests"]) & set(control["tests"])) - set(XFAIL_INPUTS)
    unexpected += sorted(f"{t} xfailed only in the mutant" for t in both
                         if run["tests"][t][0] == "xfailed"
                         and control["tests"][t][0] != "xfailed")
    unexpected += sorted(f"{t} expected to fail, passed" for t in both
                         if control["tests"][t][0] == "xfailed"
                         and run["tests"][t][0] == "passed")
    if other or prem or missing or unexpected:
        return "BROKEN", {"other": other, "premise": prem, "missing": missing,
                          "unexpected": unexpected, "detections": det, "numerics": num}
    if det:
        return "DETECTED", {"determinism": det, "numerics": num}
    if num:
        return "NUMERICS", num
    if run["rc"] != 0:
        return "BROKEN", {"rc": run["rc"]}
    return "SURVIVED", []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="code tree to mutate (not modified)")
    ap.add_argument("--work", required=True, help="scratch directory for the copies")
    ap.add_argument("--mutations", help="JSON file of kernel mutations (see load_mutations)")
    ap.add_argument("--selftest-only", action="store_true",
                    help="run the harness self-tests only (no kernel mutations needed)")
    ap.add_argument("--only", help="regex on mutation names")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the mutated copies")
    ap.add_argument("--pytest-args", default="",
                    help="replaces the default tier (self-tests, or with --allow-custom-tier)")
    ap.add_argument("--allow-custom-tier", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per pytest run")
    ap.add_argument("--out", help="results JSON (default WORK/results.json)")
    a = ap.parse_args(argv)

    kernel = load_mutations(a.mutations) if a.mutations else []
    if not kernel and not a.selftest_only:
        ap.error("no kernel mutations: pass --mutations FILE (or --selftest-only)")
    muts = SELFTESTS + kernel
    if a.only:
        muts = [m for m in muts if re.search(a.only, m.name)]
    if not muts or not (a.selftest_only or any(m in kernel for m in muts)):
        ap.error("the selection holds no kernel mutation")
    if a.list:
        for m in muts:
            print(f"{m.name:28s} expect {m.expect:9s} {m.what}")
        return 0
    if a.pytest_args and not (a.selftest_only or a.allow_custom_tier):
        ap.error("kernel mutations run the default tier (pass --allow-custom-tier to change it)")
    targets = shlex.split(a.pytest_args) if a.pytest_args else TIER
    # A custom tier, a selection of mutations or the self-tests alone give verdicts, but no
    # qualification: only the full manifest on the default tier qualifies.
    qualified = not (a.pytest_args or a.only or a.selftest_only)
    a.work, a.src = os.path.abspath(a.work), os.path.abspath(a.src)  # runs change directory
    os.makedirs(a.work, exist_ok=True)
    out = a.out or os.path.join(a.work, "results.json")
    res = {"src": a.src, "targets": targets, "control": {}, "mutations": []}

    # Every run uses this one copy of the source (the mutants are copied from it), so a source
    # edited during a long run cannot mix versions; its content hash is recorded.
    root = os.path.join(a.work, "tree")
    make_copy(a.src, root)
    res["tree_sha256"] = tree_digest(root)
    check_import_path(root)
    mandatory = collect(root, a.work, targets, a.timeout)
    res["mandatory_tests"] = len(mandatory)
    lost = sorted(set(_node(t) for t in XFAIL_ALLOWED) - set(mandatory))
    if not a.pytest_args and lost:  # the default tier must keep its pinned expected failures
        res["verdict"] = f"CONTROL FAILED: the tier no longer collects {lost}"
        json.dump(res, open(out, "w"), indent=1)
        print(res["verdict"])
        return 2
    controls = {}
    for extra in sorted({m.extra for m in muts}):
        name = "control" + "".join("_" + re.sub(r"\W+", "", e) for e in extra)
        # The oracle targets are collected without the tier's marker filter: every
        # parameterization of each must run in this control (and so in every mutant).
        required = mandatory + (collect(root, a.work, list(extra), a.timeout) if extra else [])
        run = run_tier(root, a.work, name, targets + list(extra), a.timeout)
        det, num, prem, other = classify(run)
        bad = det + num + prem + other
        res["control"][name] = {"rc": run["rc"], "seconds": run["seconds"],
                                "tests": len(run["tests"]), "problems": bad, "log": run["log"]}
        print(f"{name}: rc={run['rc']} {len(run['tests'])} tests, {len(bad)} problems, "
              f"{run['seconds']} s", flush=True)
        ran = {_node(t) for t in run["tests"]}
        missing = sorted(set(required) - ran)
        bad += [f"{t} not run" for t in missing]
        xfailed = sorted(t for t, (o, _) in run["tests"].items()
                         if o == "xfailed" and _own(t) and t not in XFAIL_INPUTS)
        want = sorted(t for t in XFAIL_STRICT if t in run["tests"])
        if xfailed != want:
            bad.append(f"expected failures {xfailed}, want {want}")
        if bad or run["rc"] != 0 or not run["tests"]:
            res["verdict"] = "CONTROL FAILED: no verdicts"
            json.dump(res, open(out, "w"), indent=1)
            print(res["verdict"], bad[:20])
            return 2
        controls[extra] = run
    # Twice: on nondeterministic kernels the digests differ between runs of one tree, and
    # "changed" then carries no information.
    base_digests = digests(root, a.work, "control", a.timeout)
    again = digests(root, a.work, "control_again", a.timeout)
    if base_digests is None or base_digests != again:
        base_digests = None
    res["control_digests_reproducible"] = base_digests is not None
    print(f"control digests reproducible: {base_digests is not None}", flush=True)
    if base_digests is None and kernel:
        res["verdict"] = "CONTROL FAILED: the control's fingerprints are not reproducible"
        json.dump(res, open(out, "w"), indent=1)
        print(res["verdict"])
        return 2

    for m in muts:
        mroot = os.path.join(a.work, f"tree_{m.name}")
        make_copy(root, mroot)
        if tree_digest(mroot) != res["tree_sha256"]:
            raise RuntimeError(f"the copy for {m.name} differs from the control's tree")
        why = check_targets(mroot, m.targets) or apply_edits(mroot, m.edits)
        rec = {"name": m.name, "what": m.what, "expect": m.expect}
        if why:
            rec.update(verdict="INVALID", detail=why)
        else:
            run = run_tier(mroot, a.work, m.name, targets + list(m.extra), a.timeout)
            v, which = verdict(run, controls[m.extra])
            dg = digests(mroot, a.work, m.name, a.timeout)
            changed = None if dg is None or base_digests is None else dg != base_digests
            if v == "SURVIVED" and m.expect != "SURVIVED" and changed is False:
                v = "NO-EFFECT"  # no fingerprinted value moved: the mutation tests nothing
            if dg is None and base_digests is not None:
                v, which = "BROKEN", {"digests": "the mutated tree's digest run failed",
                                      "verdict": v, "detail": which}
            rec.update(verdict=v, detail=which, rc=run["rc"], seconds=run["seconds"],
                       log=run["log"], tests=len(run["tests"]), changed=changed)
        rec["as_expected"] = (rec["verdict"] == m.expect
                              and (m.changes is None or base_digests is None
                                   or rec.get("changed") == m.changes))
        res["mutations"].append(rec)
        print(f"{m.name:28s} {rec['verdict']:9s} (expect {m.expect}) changed="
              f"{rec.get('changed')} {rec.get('seconds', '')} s  {str(rec['detail'])[:300]}",
              flush=True)
        if not a.keep and os.path.exists(mroot):
            _make_writable(mroot)
            shutil.rmtree(mroot)
        json.dump(res, open(out, "w"), indent=1)
    ok = bool(res["mutations"]) and all(r["as_expected"] for r in res["mutations"])
    res["verdict"] = ("ALL AS EXPECTED" if ok else "UNEXPECTED VERDICTS") + (
        "" if qualified else " (partial run: not a qualification)")
    json.dump(res, open(out, "w"), indent=1)
    print(res["verdict"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
