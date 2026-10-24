# Copyright (c) Meta Platforms, Inc. and affiliates.
"""cuda/_devmemo serves _device's reads (the same values and exceptions), memoized per device ordinal and forced row:
_devmemo.backend returns _device.backend's value on every row of the matrix and raises its RuntimeError (same message)
outside it, queries once per (ordinal, row), never caches a failure."""
import torch

ROWS = (((9, 0), 232448), ((10, 0), 232448), ((10, 3), 232448), ((8, 0), 166912), ((8, 6), 101376), ((8, 9), 101376),
        ((12, 0), 101376), ((7, 5), 65536), ((8, 0), 101376), ((9, 0), 101376))


def test_the_memoized_backend_is_the_device_backend(monkeypatch):
    from sparse_delta_memory.cuda import _device, _devmemo
    monkeypatch.setattr(_devmemo, "_DEVCLS", {})
    monkeypatch.setattr(_device, "_FORCED", None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=None: "stand-in", raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0, raising=False)
    row = {}
    queries = []
    monkeypatch.setattr(_device, "capability", lambda device=None: (queries.append("cc"), row["cc"])[1])
    monkeypatch.setattr(_device, "_optin", lambda device=None: row["optin"])

    def outcome(fn):
        try:
            return fn()
        except RuntimeError as e:
            return ("RuntimeError", str(e))
    for i, (cc, optin) in enumerate(ROWS):
        row.update(cc=cc, optin=optin)
        want = outcome(lambda: _device.backend(i))
        queries.clear()
        got = [outcome(lambda: _devmemo.backend(torch.device("cuda", i))) for _ in range(3)]
        assert got == [want] * 3, (cc, optin, got, want)
        assert len(queries) == (1 if isinstance(want, str) else 3), (cc, optin, queries)   # successes memoized only
