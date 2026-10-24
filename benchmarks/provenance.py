# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Which SDM checkout a benchmark process ran, and fingerprints of its weights."""

import hashlib
import importlib
import os
import subprocess
import sys

import torch


def _path_id(path):
    """A short fingerprint of a directory path (shows whether two runs shared it, without
    recording the path)."""
    return None if not path else hashlib.sha256(os.path.abspath(path).encode()).hexdigest()[:12]


def sdm_checkout():
    """(package, checkout root, code directory) of the SDM package a benchmark runs: the first
    sys.path entry holding ``sparse_delta_memory/`` (this release; also an installed package)
    or ``lingua/sparse_delta_memory/`` (the previous release), so two checkouts compare by
    switching PYTHONPATH. The code directory is ``sparse_delta_memory`` or ``lingua``."""
    for entry in sys.path:
        base = os.path.realpath(entry or os.getcwd())
        for name, code_dir in (("sparse_delta_memory", "sparse_delta_memory"),
                               ("lingua.sparse_delta_memory", "lingua")):
            where = os.path.join(base, *name.split("."))
            if os.path.isfile(os.path.join(where, "__init__.py")):
                sdm = importlib.import_module(name)
                got = os.path.dirname(os.path.realpath(sdm.__file__))
                if got != where:
                    raise RuntimeError(f"{name} imports from {got}, not from {where}")
                return sdm, base, code_dir
    raise ModuleNotFoundError("no sparse_delta_memory (or lingua/sparse_delta_memory) on sys.path")


def sdm_package():
    """The SDM package a benchmark runs (see sdm_checkout)."""
    return sdm_checkout()[0]


def checkout_info():
    """Which SDM package was imported: the git commit of its checkout, the git tree id of its
    code directory (the code version itself, the same for any commit with the same code) and
    the git status of that directory, fingerprints of the extension and Triton cache
    directories, and the harness hash. No paths, host or user names are recorded."""
    sdm, root, code_dir = sdm_checkout()

    def git(*args):
        try:
            return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True,
                                  timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError) as e:
            return f"unavailable: {type(e).__name__}"

    return {"module": os.path.relpath(os.path.abspath(sdm.__file__), root),
            "commit": git("rev-parse", "HEAD"),
            "code_dir": code_dir,
            "code_tree": git("rev-parse", f"HEAD:{code_dir}"),
            "code_status": git("status", "--porcelain", "--", code_dir),
            "torch_extensions_dir_id": _path_id(os.environ.get("TORCH_EXTENSIONS_DIR")),
            "triton_cache_dir_id": _path_id(os.environ.get("TRITON_CACHE_DIR")),
            "harness_sha256": harness_sha256()}


def chunk_field(args_cls):
    """The name of the SDM layer's chunk-size field in the imported checkout: ``chunk_size``,
    or ``memory_block_size`` in the previous release."""
    return ("chunk_size" if "chunk_size" in getattr(args_cls, "__dataclass_fields__", {})
            else "memory_block_size")


def public_args(a):
    """The arguments of a run as recorded: file arguments by their base name only."""
    out = dict(vars(a))
    for k in ("out", "inputs", "dir"):
        if isinstance(out.get(k), str):
            out[k] = os.path.basename(out[k])
    if isinstance(out.get("indices"), str) and out["indices"].startswith("trace:"):
        out["indices"] = "trace:" + os.path.basename(out["indices"][len("trace:"):])
    return out


HARNESS = ("bench_sdm.py", "bench_sdm_gdn.py", "bench_decode.py", "equivalence.py",
           "provenance.py", "run_release_compare.sh", "release_tables.py")


def harness_sha256():
    """sha256 of the benchmark harness files, so results can be tied to one revision."""
    h = hashlib.sha256()
    here = os.path.dirname(os.path.abspath(__file__))
    for name in HARNESS:
        with open(os.path.join(here, name), "rb") as f:
            h.update(name.encode() + b"\0" + f.read())
    return h.hexdigest()


def sha256_tensors(named):
    """sha256 over (name, shape, dtype, bytes) of ``named`` (an iterable of (name, tensor))."""
    h = hashlib.sha256()
    for name, t in named:
        t = t.detach().contiguous().cpu()
        h.update(f"{name}:{tuple(t.shape)}:{t.dtype};".encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
