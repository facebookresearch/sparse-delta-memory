# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The mutation manifest of tests/det_mutation_harness.py (tests/det_mutations.json) still
applies to this tree.

Each mutation edits the kernels at anchors: regexes on the code's structure. A kernel change
can leave an anchor matching nothing (the harness then reports that mutation INVALID, and the
suite's mutation verdict silently stops covering the kernel it targeted), or duplicate a
statement the mutation edits (a new branch), where a one-match anchor edits only one copy.
Here every mutation's target statements must occur as often as the mutation is written for,
every anchor must match exactly once, and every edited Python file must still parse, so such
drift fails the ordinary test suite. The harness is loaded from its file (it is a script, not
an importable module of this package). CPU only.

In this repository (sparse_delta_memory/ next to tests/) the manifest and every file it
edits must exist. In another layout (a vendored copy of these tests, which ships no manifest
and keeps the package elsewhere) the test is skipped.
"""
import ast
import importlib.util
import json
import os
import shutil

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MANIFEST = os.path.join(HERE, "det_mutations.json")
THIS_REPOSITORY = os.path.isdir(os.path.join(ROOT, "sparse_delta_memory"))
pytestmark = pytest.mark.skipif(not THIS_REPOSITORY,
                                reason="the manifest edits this repository's layout")
NAMES = ([m["name"] for m in json.load(open(MANIFEST))] if THIS_REPOSITORY
         and os.path.exists(MANIFEST) else ["<no manifest>"])


def _harness():
    spec = importlib.util.spec_from_file_location(
        "det_mutation_harness", os.path.join(HERE, "det_mutation_harness.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", NAMES)
def test_every_mutation_still_applies_to_this_tree(name, tmp_path):
    assert os.path.exists(MANIFEST), f"{MANIFEST} is missing"
    harness = _harness()
    mutation = {m.name: m for m in harness.load_mutations(MANIFEST)}[name]
    paths = sorted({x.path for x in mutation.edits + mutation.targets})
    missing = [p for p in paths if not os.path.exists(os.path.join(ROOT, p))]
    assert not missing, f"{name} edits files this tree does not have: {missing}"
    for p in paths:
        os.makedirs(os.path.dirname(os.path.join(tmp_path, p)), exist_ok=True)
        shutil.copyfile(os.path.join(ROOT, p), os.path.join(tmp_path, p))
    why = harness.check_targets(str(tmp_path), mutation.targets) or harness.apply_edits(
        str(tmp_path), mutation.edits)
    assert why is None, f"{name} no longer applies to this tree: {why}"
    for p in paths:
        if p.endswith(".py"):
            with open(os.path.join(tmp_path, p)) as f:
                ast.parse(f.read(), p)
