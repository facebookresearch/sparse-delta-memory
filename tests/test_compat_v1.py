# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Compatibility with v1, the previous public release: its import path and its configs.

v1 shipped the package as ``lingua.sparse_delta_memory``; v2 is the top-level ``sparse_delta_memory``.
In a checkout, ``lingua/sparse_delta_memory/__init__.py`` keeps the old name as a deprecated alias: the
old names are the same module objects as the new ones (never second copies, which would duplicate the
autograd functions and break ``isinstance``), importing it emits one FutureWarning per process, v1's
modules that v2 removed raise ImportError and any other unknown name ModuleNotFoundError. Each import
check runs in a fresh interpreter, so the alias is imported there for the first time. The alias is
part of the checkout, not of the pip package: outside a checkout these tests skip. CPU only.

v1's training configs that use SDM (v1's apps/main/configs: debug_sdm.yaml, kept as
data/debug_sdm_previous_release.yaml, and debug_hybrid.yaml, sdm_7B.yaml and sdm_flagship.yaml in
data/v1_configs/; byte-identical copies, checked by SHA-256) load through lingua's config helper and
the training entry point's loader into a SparseDeltaMemoryArgs with v1's values (memory_block_size
becomes chunk_size, with its DeprecationWarning).
"""
import ast
import hashlib
import os
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
ALIAS = os.path.join(ROOT, "lingua", "sparse_delta_memory", "__init__.py")
checkout_only = pytest.mark.skipif(not os.path.isfile(ALIAS),
                                   reason="needs the repository checkout (lingua/sparse_delta_memory)")

# v1's modules (lingua/sparse_delta_memory in v1), relative to the package.
V1_MODULES = ("cache", "cuda", "cuda.scatter_write_cuda", "cuda.sparse_ip_cuda",
              "cuda.warp_coop_gather_async_cuda", "cuda.warp_cooperative_gather_cuda", "layer",
              "memory_ops", "memory_stats", "triton_argsort")
# v1's SDM training configs: name in apps/main/configs -> (copy under tests/data, SHA-256 of v1's bytes).
V1_CONFIGS = {
    "debug_sdm.yaml": ("debug_sdm_previous_release.yaml",
                       "e995f37b2c343b963b28879d3bc0eed8d5cada4652565b3506ad81307a763211"),
    "debug_hybrid.yaml": (os.path.join("v1_configs", "debug_hybrid.yaml"),
                          "c940eba6e27f678d9eb2d04d6fcbc569b190ba40c4a8219c3b6bb63bd265853c"),
    "sdm_7B.yaml": (os.path.join("v1_configs", "sdm_7B.yaml"),
                    "6ac880405763958531780f97a091df67de174ad753e07a616976d6e6feb3ca45"),
    "sdm_flagship.yaml": (os.path.join("v1_configs", "sdm_flagship.yaml"),
                          "3f362860554110c1ba267f3efec79cef0c9e87cc01602c3ebf41fefbec9df1fa"),
}

# Helpers of the scripts: the alias's FutureWarnings among the warnings ``caught``
# (warnings.catch_warnings(record=True)), and all of them for a failure message.
_HELPERS = """
def alias_warnings(caught):
    return [str(w.message) for w in caught if issubclass(w.category, FutureWarning)
            and "lingua.sparse_delta_memory" in str(w.message)]

def shown(caught):
    return [f"{w.category.__name__}: {w.message}" for w in caught]
"""


def _run(script, *flags):
    """``script`` (after _HELPERS) in a fresh interpreter with the checkout first on its path (no
    bytecode written)."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([ROOT, os.environ.get("PYTHONPATH", "")]))
    env.pop("PYTHONWARNINGS", None)
    code = _HELPERS + textwrap.dedent(script)
    return subprocess.run([sys.executable, "-B", *flags, "-c", code], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=900)


def _ok(r, marker):
    assert r.returncode == 0 and marker in r.stdout, (r.stdout[-2000:] + r.stderr[-6000:])


def _removed():
    """The alias's _REMOVED, read from its source (importing the alias replaces it in sys.modules)."""
    with open(ALIAS) as f:
        for node in ast.parse(f.read()).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_REMOVED" for t in node.targets):
                return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"{ALIAS} defines no _REMOVED")


@checkout_only
def test_every_old_module_name_is_the_new_module():
    """With the package imported first, then the alias (twice, a submodule, a from-import): the
    old names are the new module objects, for every module of the package (by import, in
    sys.modules and by attribute), nothing is loaded twice, and one FutureWarning is emitted."""
    _ok(_run("""
        import importlib
        import pkgutil
        import sys
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import sparse_delta_memory as new
            import lingua.sparse_delta_memory as old
            import lingua.sparse_delta_memory as again
            import lingua.sparse_delta_memory.memory_ops
            from lingua.sparse_delta_memory.cuda import phase1_cuda
            import lingua
            assert old is new and again is new and lingua.sparse_delta_memory is new, (old, new)
            assert lingua.sparse_delta_memory.memory_ops is new.memory_ops
            assert phase1_cuda is new.cuda.phase1_cuda
            names = sorted(m.name for m in pkgutil.walk_packages(new.__path__, new.__name__ + "."))
            assert {"sparse_delta_memory.memory_ops", "sparse_delta_memory.layer",
                    "sparse_delta_memory.cuda.phase1_cuda"} <= set(names), names
            for name in names:
                module = importlib.import_module(name)
                assert importlib.import_module("lingua." + name) is module, name
                assert sys.modules["lingua." + name] is module, name
                obj = old
                for part in name.split(".")[1:]:
                    obj = getattr(obj, part)
                assert obj is module, name
            copies = [n for n, m in sys.modules.items() if n.startswith("lingua.sparse_delta_memory")
                      and sys.modules.get(n[len("lingua."):]) is not m]
            assert not copies, copies
        assert len(alias_warnings(caught)) == 1, shown(caught)
        print("SAME_MODULES_OK", len(names))
    """), "SAME_MODULES_OK")


@checkout_only
def test_v1_imports_build_layers_of_the_new_classes():
    """The alias imported first through a submodule, then again: one FutureWarning; v1's
    ``from lingua.sparse_delta_memory import SparseDeltaMemory, SparseDeltaMemoryArgs`` gives the
    new classes, a layer built through the old names is an instance of the new class, and args
    pickled under v1's class path load as the new class."""
    _ok(_run("""
        import pickle
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import lingua.sparse_delta_memory.layer as old_layer
            from lingua.sparse_delta_memory import SparseDeltaMemory, SparseDeltaMemoryArgs
            import lingua.sparse_delta_memory
            import sparse_delta_memory
            import sparse_delta_memory.layer as new_layer
        assert len(alias_warnings(caught)) == 1, shown(caught)
        assert old_layer is new_layer
        assert SparseDeltaMemory is sparse_delta_memory.SparseDeltaMemory
        assert SparseDeltaMemoryArgs is sparse_delta_memory.SparseDeltaMemoryArgs
        args = SparseDeltaMemoryArgs(dim=64, num_heads=2, slots_per_head=256, num_writes=1, num_reads=1, chunk_size=64)
        layer = SparseDeltaMemory(args, layer_id=0)
        assert isinstance(layer, sparse_delta_memory.SparseDeltaMemory)
        assert isinstance(layer, new_layer.SparseDeltaMemory)
        assert isinstance(layer.args, sparse_delta_memory.SparseDeltaMemoryArgs)
        data = pickle.dumps(args, protocol=2)
        new_path, old_path = b"csparse_delta_memory.layer\\n", b"clingua.sparse_delta_memory.layer\\n"
        assert data.count(new_path) == 1, data[:200]
        loaded = pickle.loads(data.replace(new_path, old_path))
        assert type(loaded) is sparse_delta_memory.SparseDeltaMemoryArgs and loaded == args, loaded
        print("OLD_CLASSES_OK")
    """), "OLD_CLASSES_OK")


@checkout_only
def test_v1_pickled_args_keep_their_chunk_size():
    """Args pickled by v1 (v1's class path, v1's fields only: memory_block_size and no
    chunk_size) load as the new class with chunk_size taken from memory_block_size, as
    construction maps it (with its DeprecationWarning), and v2's new fields at their defaults."""
    _ok(_run("""
        import pickle
        import warnings

        warnings.simplefilter("ignore", FutureWarning)
        import lingua.sparse_delta_memory  # noqa: F401  (registers v1's class path)
        from sparse_delta_memory import SparseDeltaMemoryArgs

        V2_ONLY = {"chunk_size", "cuda_graph", "grad_bank_fp32", "product_key_factor_dims"}

        class V1Args:  # stands in for v1's class; pickled under v1's class path below
            pass

        here = b"c__main__\\nV1Args\\n"
        v1_path = b"clingua.sparse_delta_memory.layer\\nSparseDeltaMemoryArgs\\n"
        for block in (64, 256):
            new = SparseDeltaMemoryArgs(dim=64, num_heads=2, slots_per_head=256, num_writes=1, num_reads=1)
            old = V1Args.__new__(V1Args)
            old.__dict__.update({k: v for k, v in vars(new).items() if k not in V2_ONLY})
            old.memory_block_size = block
            data = pickle.dumps(old, protocol=2)
            assert data.count(here) == 1, data[:200]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = pickle.loads(data.replace(here, v1_path))
            assert type(loaded) is SparseDeltaMemoryArgs, type(loaded)
            assert loaded.chunk_size == block and loaded.memory_block_size is None, vars(loaded)
            assert loaded == SparseDeltaMemoryArgs(dim=64, num_heads=2, slots_per_head=256, num_writes=1,
                                                   num_reads=1, chunk_size=block), vars(loaded)
            assert any(issubclass(w.category, DeprecationWarning)
                       and "memory_block_size" in str(w.message) for w in caught), \\
                [str(w.message) for w in caught]
        print("V1_PICKLE_OK")
    """), "V1_PICKLE_OK")


@checkout_only
def test_only_the_old_name_warns():
    """The package and lingua import without a deprecation warning of their own (warnings that
    torch or other libraries emit at import are not ours and are ignored), and the alias's import
    fails when its FutureWarning is turned into an error."""
    _ok(_run("""
        import warnings

        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            import sparse_delta_memory, lingua  # noqa: F401, E401
        ours = [f"{w.category.__name__}: {w.message} ({w.filename})" for w in seen
                if issubclass(w.category, (FutureWarning, DeprecationWarning))
                and ("sparse_delta_memory" in f"{w.filename} {w.message}" or "/lingua/" in w.filename)]
        assert not ours, ours
        print("SILENT_OK")
    """), "SILENT_OK")
    r = _run("import lingua.sparse_delta_memory", "-W", "error:lingua.sparse_delta_memory is deprecated:FutureWarning")
    assert r.returncode != 0 and "FutureWarning: lingua.sparse_delta_memory is deprecated" in r.stderr, \
        r.stderr[-4000:]


@checkout_only
def test_removed_and_unknown_old_names_raise():
    """Every v1 module is either the new module of the same name or, exactly for the alias's
    _REMOVED, raises ImportError naming it ("removed in v2"); any other name under the old prefix
    raises ModuleNotFoundError naming it, never a second copy of a module."""
    removed = _removed()
    _ok(_run(f"""
        import importlib
        import warnings

        warnings.simplefilter("ignore", FutureWarning)
        import sparse_delta_memory  # noqa: F401
        import lingua.sparse_delta_memory  # noqa: F401
        gone = []
        for rel in {V1_MODULES!r}:
            name = "lingua.sparse_delta_memory." + rel
            try:
                module = importlib.import_module(name)
            except ImportError as e:
                assert type(e) is ImportError and e.name == name and "removed in v2" in str(e), (rel, e)
                gone.append(rel)
            else:
                assert module is importlib.import_module("sparse_delta_memory." + rel), rel
        assert sorted(gone) == sorted({removed!r}), gone
        for rel in {removed!r}:
            pkg, _, leaf = ("lingua.sparse_delta_memory." + rel).rpartition(".")
            try:
                exec(f"from {{pkg}} import {{leaf}}")
            except ImportError as e:
                assert type(e) is ImportError and "removed in v2" in str(e), (rel, type(e), e)
            else:
                raise AssertionError(f"from {{pkg}} import {{leaf}} imported")
        for rel in ("no_such_module", "cuda.no_such_module", "memory_op"):
            name = "lingua.sparse_delta_memory." + rel
            try:
                importlib.import_module(name)
            except ModuleNotFoundError as e:
                assert e.name == name, (name, e.name)
            else:
                raise AssertionError(name + " imported")
        print("OLD_NAMES_OK")
    """), "OLD_NAMES_OK")


@pytest.mark.parametrize("name", sorted(V1_CONFIGS))
def test_v1_sdm_configs_load_with_their_values(name):
    """v1's SDM config (a byte-identical copy) loads through lingua's dataclass_from_dict and the
    training entry point's loader (apps/main/train.py main) with v1's sdm_args values."""
    rel, digest = V1_CONFIGS[name]
    path = os.path.join(DATA, rel)
    with open(path, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == digest, f"{rel} is not v1's apps/main/configs/{name}"
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    train = pytest.importorskip("apps.main.train", reason="needs the repository checkout (apps/, lingua/)")
    from apps.main.transformer import LMTransformerArgs
    from lingua.args import dataclass_from_dict
    from sparse_delta_memory import SparseDeltaMemoryArgs

    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    want = dict(cfg["model"]["sdm_args"])
    want["chunk_size"] = want.pop("memory_block_size")
    with pytest.warns(DeprecationWarning, match="memory_block_size"):
        model = dataclass_from_dict(LMTransformerArgs, cfg["model"])
    with pytest.warns(DeprecationWarning, match="memory_block_size"):
        args = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(train.TrainArgs()),
                                                   OmegaConf.load(path)))
    for got in (model.sdm_args, args.model.sdm_args):
        assert isinstance(got, SparseDeltaMemoryArgs)
        assert {k: getattr(got, k) for k in want} == want
        assert got.memory_block_size is None
