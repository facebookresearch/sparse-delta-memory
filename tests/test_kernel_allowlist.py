# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Every GPU kernel the layer launches, in each way it can be called, is on the allowlist:
the op's kernels (``sdm_testlib.OP_KERNELS``), the layer's own kernels (product-key top-k,
decode step, head permutes, bank expand and its reduction) and torch's kernels.

Calls: a training step (forward + backward), an eval-mode forward and backward under grad,
a forward under no_grad in eval and in train mode, and token-by-token decode. The first two
record a saving op call and the next two a non-saving one; decode does not call the op.
"""
import pytest
import torch

from sdm_testlib import (
    OP_KERNELS,
    assert_quant_tail_refused,
    kernels_outside,
    layer_inputs,
    make_layer,
    profiled_kernel_names,
    requires_cuda,
    quant_tail_refusal,
)

pytestmark = requires_cuda

LAYER_KERNELS = OP_KERNELS + (
    r"\bproduct_key_topk_(anyf_|one_)?(fwd|bwd)_kernel<",
    r"^_sdm_decode_kernel$",
    r"^_bht_permute(_packed)?_kernel$",
    r"^_bcast_rows_kernel$",
    r"^_sum_batch_kernel$",
    # torch kernels of the projections, activations, normalizations and stream offsets
    r"^void at::native::(\(anonymous namespace\)::)?(vectorized_|unrolled_)?elementwise_kernel",
    r"^void \(anonymous namespace\)::elementwise_kernel_with_index<",
    r"^void \(anonymous namespace\)::softmax_warp_(forward|backward)<",
    r"^void at::native::\(anonymous namespace\)::\w*(LayerNorm|layer_norm|RowwiseMoments|GammaBeta)",
    r"^void at::native::(\(anonymous namespace\)::)?batch_norm_",
)

CONFIGS = [
    ("bf16_hd128_w64", torch.bfloat16, dict(dim=512, num_heads=4, slots_per_head=4096,
                                            num_writes=64, num_reads=64, chunk_size=64,
                                            output_gate=True, normalize_readings=True)),
    ("bf16_w48_c48_int4_kwd", torch.bfloat16, dict(dim=256, num_heads=2, slots_per_head=1024,
                                                   num_writes=48, num_reads=24,
                                                   chunk_size=48, snapshot_quant="int4",
                                                   key_weighted_decay=True)),
    ("fp32_hd65_fp8_batchnorm", torch.float32, dict(dim=130, num_heads=2, slots_per_head=256,
                                                    num_writes=16, num_reads=100,
                                                    chunk_size=17, snapshot_quant="fp8",
                                                    query_batchnorm=True)),
    ("fp32_w256_c256_hd1024", torch.float32, dict(dim=1024, num_heads=1, slots_per_head=4096,
                                                  num_writes=256, num_reads=256,
                                                  chunk_size=256)),
    ("bf16_f8_w64", torch.bfloat16, dict(dim=256, num_heads=4, slots_per_head=57600,
                                         product_key_factor_dims=[5, 5, 4, 4, 4, 4, 3, 3],
                                         num_writes=64, num_reads=48, chunk_size=64)),
    ("fp32_f1_batchnorm", torch.float32, dict(dim=128, num_heads=2, slots_per_head=4096,
                                              product_key_factor_dims=[4096], num_writes=32,
                                              num_reads=8, chunk_size=32,
                                              query_batchnorm=True)),
]


def _calls(layer, x, gy):
    def train():
        layer.train()
        out, _ = layer(x)
        out.backward(gy)

    def eval_with_grad():
        layer.eval()
        out, _ = layer(x)
        out.backward(gy)

    def no_grad(training):
        def run():
            layer.train(training)
            with torch.no_grad():
                layer(x)
        return run

    def decode():
        layer.eval()
        with torch.no_grad():
            cache = layer.create_kv_cache(x.shape[0], 0, x.dtype, x.device)
            for t in range(3):
                layer(x[:, t:t + 1], cache=cache)

    return {"train": (train, True), "eval_with_grad": (eval_with_grad, True),
            "no_grad_eval": (no_grad(False), False), "no_grad_train": (no_grad(True), False),
            "decode": (decode, None)}


@pytest.mark.parametrize("label,dtype,cfg", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_the_layer_launches_only_allowlisted_kernels(label, dtype, cfg):
    from sparse_delta_memory.memory_ops import last_dispatch, reset_dispatch

    layer = make_layer(seed=0, dtype=dtype, **cfg)
    x, gy = layer_inputs(2, 150, cfg["dim"], dtype=dtype)
    x.requires_grad_(True)
    cls = quant_tail_refusal(dtype, layer.head_dim, layer.args.snapshot_quant)
    for name, (call, saves) in _calls(layer, x, gy).items():
        if saves and cls is not None:   # fp8 / int4 rows of a quantized tail head: training calls are refused
            assert_quant_tail_refused(call, cls)
            continue
        call()  # compile outside the profiled run
        reset_dispatch()
        _, kernels = profiled_kernel_names(call)
        outside = kernels_outside(kernels, LAYER_KERNELS)
        assert not outside, f"{label} {name}: {outside}"
        assert last_dispatch().get("save") is saves, f"{label} {name}: {last_dispatch()}"
        layer.zero_grad(set_to_none=True)
        x.grad = None
