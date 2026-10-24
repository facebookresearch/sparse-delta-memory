# Copyright (c) Meta Platforms, Inc. and affiliates.
"""CUDA-graph compatibility of the SDM layer.

A whole training step (forward + backward) is captured after an eager warmup on a side
stream, then replayed on new inputs. A replay runs the eager kernels, which are
deterministic, on the same input, so it must equal an eager step bitwise; replays on
different inputs must differ (the graph really reads its input buffer). The eager step must
not synchronize with the host.
"""
import pytest
import torch

from sdm_testlib import (
    assert_runs_bitwise,
    layer_fwd_bwd,
    layer_inputs,
    make_layer,
    rel_rms,
    requires_cuda,
)

pytestmark = requires_cuda

# (label, layer overrides, B, T, dtype)
SHAPES = [
    ("hd128_w64_c64", dict(dim=512, num_heads=4, slots_per_head=4096, num_writes=64,
                           num_reads=64, chunk_size=64, output_gate=True),
     2, 256, torch.bfloat16),
    ("hd128_w16_c64", dict(dim=256, num_heads=2, slots_per_head=1024, num_writes=16,
                           num_reads=16, chunk_size=64),
     2, 256, torch.bfloat16),
    ("fp32_hd64_c32_ragged", dict(dim=128, num_heads=2, slots_per_head=256, num_writes=8,
                                  num_reads=8, chunk_size=32),
     2, 120, torch.float32),
    ("kwd_fp8_batchnorm", dict(dim=256, num_heads=2, slots_per_head=1024, num_writes=16,
                               num_reads=16, chunk_size=64, key_weighted_decay=True,
                               snapshot_quant="fp8", query_batchnorm=True),
     2, 256, torch.bfloat16),
    ("f8_w64", dict(dim=256, num_heads=2, slots_per_head=57600,
                    product_key_factor_dims=[5, 5, 4, 4, 4, 4, 3, 3], num_writes=64,
                    num_reads=48, chunk_size=64),
     2, 256, torch.float32),
    ("fp32_f1_ragged", dict(dim=128, num_heads=2, slots_per_head=2048,
                            product_key_factor_dims=[2048], num_writes=16, num_reads=8,
                            chunk_size=32),
     2, 120, torch.float32),
]


class CapturedStep:
    """A captured forward + backward of ``layer`` on static input buffers."""

    def __init__(self, layer, shape, dtype, warmup=3):
        self.layer = layer
        self.params = dict(layer.named_parameters())
        self.x = torch.zeros(*shape, device="cuda", dtype=dtype, requires_grad=True)
        self.gy = torch.zeros(*shape, device="cuda", dtype=dtype)
        x0, gy0 = layer_inputs(*shape, dtype=dtype, seed=100)
        with torch.no_grad():
            self.x.copy_(x0)
            self.gy.copy_(gy0)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self._zero()
                out, _ = layer(self.x)
                out.backward(self.gy)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        # Capture with every grad unset, so the captured backward owns (and overwrites)
        # its gradient buffers instead of accumulating into eager ones.
        self._zero()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out, _ = layer(self.x)
            self.out.backward(self.gy)
        torch.cuda.synchronize()
        self.grads = {n: p.grad for n, p in self.params.items()}
        self.x_grad = self.x.grad
        assert all(g is not None for g in self.grads.values())

    def _zero(self):
        for p in self.params.values():
            p.grad = None
        self.x.grad = None

    def replay(self, x, gy):
        with torch.no_grad():
            self.x.copy_(x)
            self.gy.copy_(gy)
        self.graph.replay()
        torch.cuda.synchronize()
        res = {"out": self.out.detach().clone(), "grad_x": self.x_grad.detach().clone()}
        res.update({"grad." + n: g.detach().clone() for n, g in self.grads.items()})
        return res


@pytest.mark.parametrize("label,cfg,B,T,dtype", SHAPES, ids=[s[0] for s in SHAPES])
def test_captured_training_step_matches_eager_on_new_inputs(label, cfg, B, T, dtype):
    from sparse_delta_memory import memory_ops as mo

    layer = make_layer(seed=0, dtype=dtype, **cfg)
    shape = (B, T, cfg["dim"])
    mo.reset_dispatch()
    step = CapturedStep(layer, shape, dtype)
    assert mo.last_dispatch()["save"]

    replays = []
    for seed in (201, 202, 203):
        x, gy = layer_inputs(*shape, dtype=dtype, seed=seed)
        r1, r2 = step.replay(x, gy), step.replay(x, gy)
        eager = [layer_fwd_bwd(layer, x, gy) for _ in range(2)]
        # A replay that accumulated its gradients instead of overwriting them (2x on the
        # second replay) differs.
        assert_runs_bitwise(eager + [r1, r2], label=f"{label} seed={seed}")
        assert torch.equal(r1["grad.Wo.bias"], r2["grad.Wo.bias"]), "second replay accumulated"
        replays.append(r1)
    for a, b in ((0, 1), (1, 2)):
        assert rel_rms(replays[a]["out"], replays[b]["out"]) > 0.5, "replay ignored its input"
        assert rel_rms(replays[a]["grad.Wv_write.weight"],
                       replays[b]["grad.Wv_write.weight"]) > 0.1


def test_captured_training_step_with_the_graph_option():
    """With cuda_graph=True the layer replays its own graphs during the uncaptured warmup and
    runs eager inside the whole-step capture, whose replays match the layer without graphs."""
    from sparse_delta_memory.cuda_graph import reset_cuda_graphs

    label, cfg, B, T, dtype = SHAPES[1]
    shape = (B, T, cfg["dim"])
    reset_cuda_graphs()
    eager = make_layer(seed=0, dtype=dtype, **cfg)
    layer = make_layer(seed=0, dtype=dtype, cuda_graph=True, **cfg)
    step = CapturedStep(layer, shape, dtype)
    warmup = dict(forward_capture=1, forward_replay=3, backward_capture=1, backward_replay=2)
    assert dict(layer.graph_runner.stats) == warmup
    for seed in (201, 202):
        x, gy = layer_inputs(*shape, dtype=dtype, seed=seed)
        replays = [step.replay(x, gy) for _ in range(2)]
        assert_runs_bitwise([layer_fwd_bwd(eager, x, gy) for _ in range(2)] + replays,
                            label=f"{label} seed={seed}")
    assert dict(layer.graph_runner.stats) == warmup
    reset_cuda_graphs()


@pytest.mark.parametrize("label,cfg,B,T,dtype", SHAPES, ids=[s[0] for s in SHAPES])
def test_eager_training_step_never_synchronizes(label, cfg, B, T, dtype):
    layer = make_layer(seed=0, dtype=dtype, **cfg)
    x, gy = layer_inputs(B, T, cfg["dim"], dtype=dtype)
    layer_fwd_bwd(layer, x, gy)          # builds extensions, kernels and lazy caches
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for p in layer.parameters():
            p.grad = None
        xi = x.detach().requires_grad_(True)
        out, _ = layer(xi)
        out.backward(gy)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all() and torch.isfinite(xi.grad).all()


def test_sync_debug_mode_detects_a_synchronization():
    """Control: the detector used above does raise on a host sync."""
    t = torch.ones(4, device="cuda")
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(RuntimeError):
            t.sum().item()
    finally:
        torch.cuda.set_sync_debug_mode(0)


def test_stats_logging_does_not_break_capture():
    """With log_memory_access_stats on, a training step captures (stats are eval-only) and
    an eval forward captures (stats are skipped while capturing); eager eval records them."""
    cfg = dict(SHAPES[1][1], log_memory_access_stats=True)
    layer = make_layer(seed=0, dtype=torch.bfloat16, **cfg)
    shape = (2, 256, cfg["dim"])
    step = CapturedStep(layer, shape, torch.bfloat16)
    x, gy = layer_inputs(*shape, seed=5)
    r = step.replay(x, gy)
    assert torch.isfinite(r["out"]).all()
    assert layer._read_idx_counts.sum().item() == 0, "stats were recorded in training"

    layer.eval()
    static_x = x.clone()
    with torch.no_grad():
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            eager_out, _ = layer(static_x)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        recorded = layer._read_idx_counts.sum().item()
        assert recorded > 0, "eager eval forward did not record stats"
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out, _ = layer(static_x)
        graph.replay()
        torch.cuda.synchronize()
    assert layer._read_idx_counts.sum().item() == recorded, "stats were recorded while capturing"
    assert rel_rms(eager_out, graph_out) < 2e-2


@pytest.mark.parametrize("label,cfg", [(s[0], s[1]) for s in SHAPES[:2] + SHAPES[4:]],
                         ids=[s[0] for s in SHAPES[:2] + SHAPES[4:]])
def test_captured_decode_step_matches_eager(label, cfg):
    """One T == 1 decode step on a cache, captured and replayed from a fixed bank state."""
    layer = make_layer(seed=0, dtype=torch.bfloat16, **cfg)
    layer.eval()
    B, dim = 2, cfg["dim"]
    cache = layer.create_kv_cache(bsz=B, seq_len=0, dtype=torch.bfloat16, device="cuda")
    prefix, _ = layer_inputs(B, 96, dim, seed=11)
    static_x = torch.zeros(B, 1, dim, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        layer(prefix, cache=cache)
        bank0 = cache.memory.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                layer(static_x, cache=cache)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out, _ = layer(static_x, cache=cache)
    bank = cache.memory
    outs = []
    for seed in (21, 22):
        x_t, _ = layer_inputs(B, 1, dim, seed=seed)
        with torch.no_grad():
            bank.copy_(bank0)
            static_x.copy_(x_t)
            graph.replay()
            torch.cuda.synchronize()
            replay_out, replay_bank = static_out.clone(), bank.clone()
            bank.copy_(bank0)
            eager_out, _ = layer(x_t, cache=cache)
            eager_bank = bank.clone()
        assert rel_rms(eager_out, replay_out) < 1e-2, "decode replay differs from eager"
        assert rel_rms(eager_bank - bank0, replay_bank - bank0) < 1e-2, "bank update differs"
        assert not torch.equal(replay_bank, bank0), "the replay did not write the bank"
        outs.append(replay_out)
    assert rel_rms(outs[0], outs[1]) > 0.1, "replay ignored its input"

