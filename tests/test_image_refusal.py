# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The loaded-image check (sparse_delta_memory.cuda._image): PTX-only builds for older GPUs are refused on Hopper /
Blackwell GPUs at every CUDA extension load, before any kernel of the extension runs; an RTX-class device warns once."""
import ast
import contextlib
import os
import re
import sys
import types
import warnings

import pytest
import torch

from sdm_testlib import requires_cuda

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sparse_delta_memory")
MODERN = ((9, 0), (10, 0), (10, 3))
_ARCH = re.compile(r"gencode|-arch|gpu-architecture|code=|compute_\d|sm_\d")


def _package_trees():
    for root, _, names in os.walk(PKG):
        for n in names:
            if n.endswith(".py"):
                path = os.path.join(root, n)
                yield os.path.relpath(path, PKG), ast.parse(open(path, encoding="utf-8").read())


def test_refusal_cannot_be_caught():
    """ImageRefusal is a BaseException outside Exception, and no handler of the package catches BaseException."""
    from sparse_delta_memory.cuda import _image
    assert issubclass(_image.ImageRefusal, BaseException) and not issubclass(_image.ImageRefusal, Exception)
    bad = [(rel, h.lineno) for rel, tree in _package_trees() for h in ast.walk(tree) if isinstance(h, ast.ExceptHandler)
           and (h.type is None or any(isinstance(n, ast.Name) and n.id in ("BaseException", "ImageRefusal")
                                      for n in ast.walk(h.type)))]
    assert bad == []


def test_every_extension_shares_one_target_list():
    """No extension passes architecture flags of its own and only _build reaches torch's extension builder, so the probe's
    loaded image is every extension's."""
    calls, bad = 0, []
    for rel, tree in _package_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "load_cuda_extension":
                calls += 1
                bad += [(rel, node.lineno) for k in node.keywords if k.arg is None or (
                    k.arg.startswith("extra_") and _ARCH.search(ast.unparse(k.value)))]
            if rel != os.path.join("cuda", "_build.py") and "cpp_extension" in ast.dump(node) and isinstance(
                    node, (ast.Import, ast.ImportFrom, ast.Attribute)):
                bad.append((rel, node.lineno))
    assert bad == [] and calls >= 12, (calls, bad)


@pytest.fixture
def img(monkeypatch):
    from sparse_delta_memory.cuda import _device, _image
    st = {"real": (9, 0), "probe": 90, "backend": "modern", "backend_calls": [], "probe_ptx": _image.probe_ptx}
    monkeypatch.setattr(_image, "_OK", set())
    monkeypatch.setattr(_image, "_WARNED", set())
    monkeypatch.setattr(_image, "probe_ptx", lambda i, env: st["probe"])
    monkeypatch.setattr(_device, "backend", lambda d=None: (st["backend_calls"].append(d), st["backend"])[1])
    monkeypatch.setattr(_device, "capability", lambda d=None: st["real"])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=None: st["real"], raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=None: "stand-in", raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1, raising=False)
    monkeypatch.setattr(sys.modules["sparse_delta_memory"], "__version__", "2.0.0", raising=False)
    return _image, st


def test_decision_modern_routes_need_a_modern_loaded_image(img):
    _image, st = img
    for real, ptx, backend, refused in (((9, 0), 80, "modern", True), ((10, 3), 86, "modern", True),
                                        ((10, 0), 89, "modern", True), ((9, 0), 90, "modern", False),
                                        ((10, 3), 103, "modern", False), ((10, 3), 90, "modern", False),
                                        ((8, 0), 80, "modern", True),      # a forced modern row on an A100
                                        ((10, 3), 80, "a100", False),      # a forced A100 row: the portable path
                                        ((8, 6), 86, "rtx", False), ((12, 0), 89, "rtx", False)):
        _image._OK.clear()
        st.update(real=real, probe=ptx, backend=backend)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", _image.UntestedHardwareWarning)
            if not refused:
                _image.require_supported_image(0)
                continue
            with pytest.raises(_image.ImageRefusal) as err:
                _image.require_supported_image(0)
        assert str(err.value).startswith(
            "sparse_delta_memory: PTX-only builds for older GPUs are refused on Hopper / Blackwell GPUs. The CUDA image "
            f"loaded on device 0 (stand-in, compute capability {real[0]}.{real[1]}) was compiled for compute_{ptx} "
            f"(cudaFuncGetAttributes ptxVersion {ptx}; TORCH_CUDA_ARCH_LIST=")
        assert "Rebuild with TORCH_CUDA_ARCH_LIST including your GPU (9.0 for H100, 10.0 for B200, 10.3 for GB300)" in str(
            err.value)
        assert not _image._OK                                  # refusals are never memoized


def test_a_load_checks_every_visible_device_once_current_first(img, monkeypatch):
    _image, st = img
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3, raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1, raising=False)
    _image.require_supported_image(2)
    assert st["backend_calls"] == [2]
    _image.require_supported_image()
    assert st["backend_calls"] == [2, 1, 0]                    # the current device first; device 2 memoized
    _image.require_supported_image()
    _image.require_supported_image(2)
    assert st["backend_calls"] == [2, 1, 0]                    # memoized: one set lookup per device


def test_every_extension_load_runs_the_check_but_the_probes(img, monkeypatch, tmp_path):
    _image, st = img
    from sparse_delta_memory.cuda import _build
    builds, checks = [], []

    def get_build_directory(name, verbose):
        (tmp_path / name).mkdir(exist_ok=True)
        return str(tmp_path / name)
    monkeypatch.setattr(_build, "cpp_extension", types.SimpleNamespace(
        CUDA_HOME=str(tmp_path), _get_build_directory=get_build_directory,
        load=lambda name, sources, **kw: (builds.append(name), types.SimpleNamespace(f=lambda: 0))[1]))
    monkeypatch.setattr(_build, "_require_toolchain", lambda: str(tmp_path))
    monkeypatch.setattr(_build, "_target_architectures", lambda: {"TORCH_CUDA_ARCH_LIST": "9.0", "capabilities": [[9, 0]]})
    monkeypatch.setattr(torch, "__version__", getattr(torch, "__version__", "stand-in"), raising=False)
    monkeypatch.setattr(torch, "version", getattr(torch, "version", types.SimpleNamespace(cuda="stand-in")), raising=False)
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0")
    check = _image.require_supported_image
    monkeypatch.setattr(_image, "require_supported_image", lambda device=None: (checks.append(device), check(device))[1])

    def load(name):
        return _build.load_cuda_extension(name, cpp_sources="int f();", cuda_sources="__global__ void k() {}",
                                          functions=("f",))
    assert load(_image.PROBE_BASE).f() == 0 and checks == []   # the probe's own load is not checked (it is the check)
    for _ in range(2):
        assert load("sdm_x").f() == 0
    assert checks == [None, None] and len(builds) == 3
    _image._OK.clear()
    st["probe"] = 80
    with pytest.raises(_image.ImageRefusal):                   # after the load, before the module is returned
        load("sdm_x")


def test_an_rtx_device_warns_once(img, monkeypatch):
    _image, st = img
    st.update(real=(8, 6), probe=86, backend="rtx")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2, raising=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for _ in range(3):
            _image.require_supported_image()
        got = [str(w.message) for w in rec if issubclass(w.category, _image.UntestedHardwareWarning)]
        assert len(got) == 2 and ("has not been tested on RTX-class GPUs: device 0 (stand-in, compute capability 8.6); "
                                  "results are unverified") in got[0]
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.6")          # another build environment: checked again, no warning
        _image.require_supported_image()
        assert len([w for w in rec if issubclass(w.category, _image.UntestedHardwareWarning)]) == 2
        del rec[:]
        for backend in ("a100", "modern"):
            st.update(backend=backend, probe=90)
            _image._OK.clear()
            _image._WARNED.clear()
            _image.require_supported_image()
        assert not [w for w in rec if issubclass(w.category, _image.UntestedHardwareWarning)]
    st.update(backend="rtx", probe=86)
    _image._OK.clear()
    _image._WARNED.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error", _image.UntestedHardwareWarning)
        with pytest.raises(_image.UntestedHardwareWarning):
            _image.require_supported_image(0)
    assert not _image._OK and not _image._WARNED
    assert issubclass(_image.UntestedHardwareWarning, UserWarning)


def test_a_changed_build_environment_is_checked_again(img, monkeypatch):
    """The approval and the probe are keyed by the build environment and the toolkit, so an
    extension built under another one (TORCH_CUDA_ARCH_LIST 8.0+PTX, or another CUDA_HOME, after a native load) is checked
    again with a probe built under it; builds inside the test-only hook are not checked."""
    _image, st = img
    from sparse_delta_memory.cuda import _build
    monkeypatch.setattr(_image, "probe_ptx", st["probe_ptx"])
    monkeypatch.setattr(_image, "_PROBE", {})
    monkeypatch.setattr(_image, "_PTX", {})
    builds = []

    def load(name, **kw):   # the probe built now: compute_80 PTX for 8.0+PTX or the old toolkit, else native
        builds.append(os.environ.get("TORCH_CUDA_ARCH_LIST") or _build.cpp_extension.CUDA_HOME)
        ptx = 80 if builds[-1] in ("8.0+PTX", "/old/cuda") else 90
        return types.SimpleNamespace(sdm_image_ptx=lambda: ptx)
    monkeypatch.setattr(_build, "load_cuda_extension", load)
    monkeypatch.setattr(torch.cuda, "device", lambda i: contextlib.nullcontext(), raising=False)
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    _image.require_supported_image(0)                          # a native load: approved
    with _image._unchecked_build_env():
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0+PTX")
        _image.require_supported_image(0)                      # inside the hook: not checked
    with pytest.raises(_image.ImageRefusal, match="compiled for compute_80"):
        _image.require_supported_image(0)                      # a later 8.0+PTX build: checked again, refused
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST")
    monkeypatch.setattr(_build.cpp_extension, "CUDA_HOME", "/old/cuda", raising=False)
    with pytest.raises(_image.ImageRefusal, match="compiled for compute_80"):
        _image.require_supported_image(0)                      # a build with another toolkit: checked again, refused
    assert builds[1:] == ["8.0+PTX", "/old/cuda"]


def test_nvcc_environment_flags_enter_the_build_name(monkeypatch):
    """NVCC_APPEND_FLAGS / NVCC_PREPEND_FLAGS shape the targets, so a build under them is its own build (cache key)."""
    from sparse_delta_memory.cuda import _build
    monkeypatch.setattr(_build, "_require_toolchain", lambda: "/stand-in/cuda")
    monkeypatch.setattr(_build, "_target_architectures", lambda: {"TORCH_CUDA_ARCH_LIST": "9.0", "capabilities": [[9, 0]]})
    monkeypatch.setattr(torch, "__version__", getattr(torch, "__version__", "stand-in"), raising=False)
    monkeypatch.setattr(torch, "version", getattr(torch, "version", types.SimpleNamespace(cuda="stand-in")), raising=False)
    names = []
    for env in ({}, {"NVCC_APPEND_FLAGS": "-gencode=arch=compute_90,code=sm_90"},
                {"NVCC_PREPEND_FLAGS": "-gencode=arch=compute_90,code=sm_90"}, {}):
        for k in ("NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        names.append(_build._extension_name("sdm_x", cpp_sources="int f();", cuda_sources="__global__ void k() {}",
                                            functions=("f",)))
    assert len(set(names[:3])) == 3 and names[3] == names[0]


def _want():
    m = re.fullmatch(r"(\d+)\.(\d+)\+PTX", (os.environ.get("TORCH_CUDA_ARCH_LIST") or "").strip())
    return int(m.group(1)) * 10 + int(m.group(2)) if m else None


@requires_cuda
def test_portable_image_on_a_modern_gpu_is_refused_at_the_first_extension_load():
    """A PTX-only build for an older GPU (8.0+PTX / 8.6+PTX) on H100 / B200 / GB300: the first extension load raises the
    documented ImageRefusal (no Exception) and no fault follows (the context stays usable)."""
    from sparse_delta_memory.cuda import _device, _image, gather_cuda
    dev = torch.cuda.current_device()
    ptx = _image.probe_ptx(dev, _image.build_env())
    if _device.backend(dev) != "modern" or ptx in _image.MODERN_PTX:
        pytest.skip("not a portable image on a modern GPU")
    assert _want() is None or ptx == _want()
    with pytest.raises(_image.ImageRefusal) as err:
        gather_cuda._module()
    assert not isinstance(err.value, Exception)
    assert re.match(r"sparse_delta_memory: PTX-only builds for older GPUs are refused on Hopper / Blackwell GPUs\. "
                    rf"The CUDA image loaded on device \d+ \(.*\) was compiled for compute_{ptx} ", str(err.value))
    assert torch.ones(4, device="cuda").sum().item() == 4.0


@requires_cuda
def test_portable_rows_keep_their_portable_path():
    """sm_80 / sm_86 devices keep the portable path: an A100 row on an 8.0+PTX build (an RTX 30 row on 8.6+PTX), forced
    on this GPU, runs the op (no refusal)."""
    import test_determinism as td
    from sparse_delta_memory.cuda import _device
    row = {80: ((8, 0), 166912, 108), 86: ((8, 6), 101376, 82)}.get(_want())
    if row is None:
        pytest.skip("not an 8.0+PTX / 8.6+PTX build")
    c = next(c for c in td.OP_FAST if c.id == "hot_bf16_C64")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with _device._force_row(*row):
            res = td.run_op(td.op_inputs(c), c)
            torch.cuda.synchronize()
    assert all(torch.isfinite(t.float()).all() for t in res.values())
