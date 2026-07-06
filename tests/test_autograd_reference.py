# Copyright (c) Meta Platforms, Inc. and affiliates.
"""SDM correctness vs a pure-torch autograd baseline: fp32, chunk invariance, inference.

Ported from the standalone SDM repo. Tests the SparseDeltaMemory kernel/layer against a
pure-torch autograd reference (an implementation-independent oracle). Covers:
  - fp32 and bf16 forward + ALL backward gradients vs the autograd reference
  - Chunk-size invariance (cs=16/32/64/128) for forward + backward
  - Layer inference-vs-training forward match + token-by-token decode
  - B=1,2,4 and H=1,2

Run in the release env:  PYTHONPATH=. python tests/test_autograd_reference.py
"""
import torch

from lingua.sparse_delta_memory import (
    SparseDeltaMemory as SDMLayer,
    SparseDeltaMemoryArgs as SDMLayerArgs,
)
from lingua.sparse_delta_memory.memory_ops import GatedSparseMemoryWriteRead


PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))


def _rel(a, b):
    return (a - b).abs().max().item() / max(a.abs().max().item(), 1e-30)


def _abs(a, b):
    return (a - b).abs().max().item()


# ============================================================
# Pure-torch autograd reference (token-by-token SDM recurrence)
# ============================================================

def sdm_reference(memory, k_idx, k_val, v, beta, g, q_idx, q_val):
    """Token-by-token SDM forward. No in-place ops, pure autograd."""
    B_T = k_idx.shape[0]
    mem = memory
    readings = []
    for t in range(B_T):
        mem_at_k = mem[k_idx[t]]
        decay = torch.exp(g[t]).unsqueeze(-1)
        mem_read = mem_at_k * decay
        retrieved = (k_val[t].unsqueeze(-1) * mem_read).sum(0)
        delta_v = beta[t] * (v[t] - retrieved)
        write_vals = k_val[t].unsqueeze(-1) * delta_v.unsqueeze(0)
        delta_at_k = mem_read - mem_at_k + write_vals
        update = torch.zeros_like(mem)
        idx_expanded = k_idx[t].unsqueeze(-1).expand_as(delta_at_k)
        update.scatter_add_(0, idx_expanded, delta_at_k)
        mem = mem + update
        mem_at_q = mem[q_idx[t]]
        reading = (q_val[t].unsqueeze(-1) * mem_at_q).sum(0)
        readings.append(reading)
    return torch.stack(readings, dim=0)


def make_raw_inputs(ns, dim, B, T, W, R, device, dtype, seed=42):
    """Generate raw kernel-level inputs with unique sorted indices."""
    torch.manual_seed(seed)
    total_slots = B * ns
    B_T = B * T
    ki_list, qi_list = [], []
    for b in range(B):
        offset = b * ns
        ki_list.append(torch.stack([torch.randperm(ns, device=device)[:W].sort().values
                                    for _ in range(T)]) + offset)
        qi_list.append(torch.stack([torch.randperm(ns, device=device)[:R].sort().values
                                    for _ in range(T)]) + offset)
    return (
        torch.randn(total_slots, dim, device=device, dtype=dtype) * 0.1,  # memory
        torch.cat(ki_list), torch.cat(qi_list),                            # indices
        torch.randn(B_T, W, device=device, dtype=dtype) * 0.3,            # k_val
        torch.randn(B_T, dim, device=device, dtype=dtype) * 0.1,          # v
        torch.rand(B_T, 1, device=device, dtype=dtype) * 0.3,             # beta
        torch.empty(B_T, 1, device=device, dtype=dtype).uniform_(-0.5, -0.01),  # g
        torch.randn(B_T, R, device=device, dtype=dtype) * 0.3,            # q_val
    )


def run_ref_and_kernel(memory, k_idx, q_idx, k_val, v, beta, g, q_val, cs, ns, B):
    """Run both reference and kernel, return grads dict for each."""
    params = (k_val, v, beta, g, q_val)
    names = ("k_val", "v", "beta", "g", "q_val")

    # Reference
    mem_r = memory.clone().requires_grad_(True)
    pr = [x.clone().requires_grad_(True) for x in params]
    out_r = sdm_reference(mem_r, k_idx, *pr[:1], pr[1], pr[2], pr[3], q_idx, pr[4])
    out_r.sum().backward()
    ref = {"out": out_r.detach(), "memory": mem_r.grad}
    for n, p in zip(names, pr):
        ref[n] = p.grad

    # Kernel (extra trailing GatedSparseMemoryWriteRead args use their defaults).
    mem_k = memory.clone().requires_grad_(True)
    pk = [x.clone().requires_grad_(True) for x in params]
    out_k, _ = GatedSparseMemoryWriteRead.apply(
        mem_k + 0, k_idx, pk[0], pk[1], pk[2], pk[3], q_idx, pk[4],
        cs, True, ns * B, B)
    out_k.sum().backward()
    kern = {"out": out_k.detach(), "memory": mem_k.grad}
    for n, p in zip(names, pk):
        kern[n] = p.grad

    return ref, kern


# ============================================================
# Tests
# ============================================================


def test_autograd_reference():
    """Kernel forward + ALL backward gradients vs pure-torch autograd reference."""
    print("=== Autograd reference comparison (forward + all grads) ===")
    device = "cuda"
    grad_names = ["out", "memory", "k_val", "v", "beta", "g", "q_val"]

    configs = [
        # (ns, dim, B, T, W, R, dtype, tol, label)
        (512, 64, 1, 64, 4, 4, torch.float32, 2e-2, "fp32 B=1"),
        (512, 64, 2, 64, 4, 4, torch.float32, 2e-2, "fp32 B=2"),
        (512, 64, 4, 32, 4, 4, torch.float32, 2e-2, "fp32 B=4"),
        (512, 64, 1, 64, 4, 4, torch.bfloat16, 3e-2, "bf16 B=1"),
        (512, 64, 2, 64, 4, 4, torch.bfloat16, 3e-2, "bf16 B=2"),
        (512, 64, 4, 32, 4, 4, torch.bfloat16, 3e-2, "bf16 B=4"),
    ]

    for ns, dim, B, T, W, R, dtype, tol, label in configs:
        memory, k_idx, q_idx, k_val, v, beta, g, q_val = make_raw_inputs(
            ns, dim, B, T, W, R, device, dtype)
        for cs in [16, 64]:
            ref, kern = run_ref_and_kernel(memory, k_idx, q_idx, k_val, v, beta, g, q_val, cs, ns, B)
            for name in grad_names:
                rel = _rel(ref[name], kern[name])
                absd = _abs(ref[name], kern[name])
                lbl = "forward" if name == "out" else f"grad_{name}"
                check(f"{label} cs={cs} {lbl}", rel < tol,
                      f"rel={rel:.2e} abs={absd:.2e}")
    print()


def test_fp32_forward_precision():
    """fp32 output must not be truncated to bf16."""
    print("=== fp32 forward precision ===")
    device = "cuda"
    memory, k_idx, q_idx, k_val, v, beta, g, q_val = make_raw_inputs(
        512, 64, 1, 64, 4, 4, device, torch.float32)
    mem = memory.clone()
    out, _ = GatedSparseMemoryWriteRead.apply(
        mem, k_idx, k_val.clone(), v.clone(), beta.clone(), g.clone(),
        q_idx, q_val.clone(), 32, True, 512, 1)
    rt = out.to(torch.bfloat16).to(torch.float32)
    diff = (out - rt).abs().max().item()
    check("fp32 output != bf16 roundtrip", diff > 1e-5, f"diff={diff:.2e}")
    print()


def test_chunk_invariance():
    """Chunk-size invariance: cs=16 vs cs=32/64/128, forward + all grads."""
    print("=== Chunk-size invariance ===")
    device = "cuda"
    grad_names = ["out", "memory", "k_val", "v", "beta", "g", "q_val"]

    configs = [
        (512, 64, 1, 128, 4, 4, torch.float32, 1e-2, "fp32 B=1"),
        (512, 64, 2, 128, 4, 4, torch.float32, 5e-4, "fp32 B=2"),
        (512, 64, 2, 128, 4, 4, torch.bfloat16, 1.5e-2, "bf16 B=2"),
    ]

    for ns, dim, B, T, W, R, dtype, tol, label in configs:
        memory, k_idx, q_idx, k_val, v, beta, g, q_val = make_raw_inputs(
            ns, dim, B, T, W, R, device, dtype)

        def _run(cs):
            mem = memory.clone().requires_grad_(True)
            ps = [x.clone().requires_grad_(True) for x in (k_val, v, beta, g, q_val)]
            o, _ = GatedSparseMemoryWriteRead.apply(
                mem + 0, k_idx, *ps[:1], ps[1], ps[2], ps[3], q_idx, ps[4],
                cs, True, ns * B, B)
            o.sum().backward()
            d = {"out": o.detach(), "memory": mem.grad}
            for n, p in zip(["k_val", "v", "beta", "g", "q_val"], ps):
                d[n] = p.grad
            return d

        ref = _run(16)
        for cs in [32, 64, 128]:
            r = _run(cs)
            for name in grad_names:
                rel = _rel(ref[name], r[name])
                absd = _abs(ref[name], r[name])
                lbl = "forward" if name == "out" else f"grad_{name}"
                check(f"{label} cs={cs} vs 16 {lbl}", rel < tol,
                      f"rel={rel:.2e} abs={absd:.2e}")
    print()


def test_layer_inference_training_match():
    """SDMLayer training forward must match inference forward + token-by-token decode."""
    print("=== Layer inference vs training forward (H=1, H=2) ===")
    device = "cuda"

    for H in [1, 2]:
        torch.manual_seed(42)
        cfg = dict(dim=128, slots_per_head=1024, num_writes=16, num_reads=16,
                   memory_block_size=64, num_heads=H, log_memory_access_stats=False,
                   normalize_readings=True)
        layer = SDMLayer(SDMLayerArgs(**cfg), layer_id=0).cuda().bfloat16()
        layer.init_weights()

        torch.manual_seed(7)
        for B in [2, 4]:
            x = torch.randn(B, 256, cfg["dim"], device=device, dtype=torch.bfloat16) * 0.1

            with torch.no_grad():
                layer.train()
                out_train = layer(x.clone())[0]
                layer.eval()
                out_inf = layer(x.clone())[0]

            diff = _abs(out_train, out_inf)
            scale = out_train.abs().max().item()
            rel = diff / max(scale, 1e-10)
            check(f"H={H} B={B} train vs inference", rel < 0.015,
                  f"rel={rel:.2e} abs={diff:.2e}")

            # Token-by-token decode vs single-pass
            with torch.no_grad():
                cache = None
                outs = []
                for t in range(x.shape[1]):
                    o, cache = layer(x[:, t:t+1], cache=cache)
                    outs.append(o)
                out_tok = torch.cat(outs, dim=1)

            diff_tok = _abs(out_inf, out_tok)
            rel_tok = diff_tok / max(scale, 1e-10)
            check(f"H={H} B={B} full vs token-by-token", rel_tok < 0.015,
                  f"rel={rel_tok:.2e} abs={diff_tok:.2e}")
    print()


def test_layer_backward():
    """SDMLayer backward produces finite gradients for all parameters."""
    print("=== Layer backward (finite grads, H=1/2) ===")
    device = "cuda"

    for H in [1, 2]:
        for B in [2, 4]:
            torch.manual_seed(42)
            cfg = dict(dim=128, slots_per_head=1024, num_writes=16, num_reads=16,
                       memory_block_size=64, num_heads=H, log_memory_access_stats=False,
                       normalize_readings=True)
            layer = SDMLayer(SDMLayerArgs(**cfg), layer_id=0).cuda().bfloat16()
            layer.init_weights()
            layer.train()

            x = torch.randn(B, 64, cfg["dim"], device=device, dtype=torch.bfloat16) * 0.1
            x.requires_grad_(True)
            out, _ = layer(x)
            out.sum().backward()

            out_ok = out.isfinite().all().item()
            x_ok = x.grad is not None and x.grad.isfinite().all().item()
            params_ok = all(p.grad is not None and p.grad.isfinite().all().item()
                           for p in layer.parameters() if p.requires_grad)
            check(f"H={H} B={B} finite grads", out_ok and x_ok and params_ok)
    print()


def main():
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}\n")

    test_fp32_forward_precision()
    test_autograd_reference()
    test_chunk_invariance()
    test_layer_inference_training_match()
    test_layer_backward()

    print(f"Results: {PASS} passed, {FAIL} failed")
    return 1 if FAIL > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
