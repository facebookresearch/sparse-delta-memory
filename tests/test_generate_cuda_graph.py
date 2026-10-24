# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The CUDA-graph decode step of SDM models in the generator (``sdm_decode_cuda_graph``).

A replay runs the kernels of the eager decode step on the same inputs. Starting from one
prefilled state (KV caches, SDM banks, positions), the graphed steps therefore produce the
eager steps' logits bit for bit, so greedy decoding picks the same tokens. (The prefill
is bitwise reproducible as well; test_determinism.py checks it.)
"""
import pytest
import torch
from torch import nn

pytest.importorskip("apps.main.generate", reason="needs the repository checkout (apps/, lingua/)")
from apps.main.generate import (  # noqa: E402
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    pack_prompts,
    sample_tokens,
)
from apps.main.transformer import LMTransformer, LMTransformerArgs  # noqa: E402
from lingua.tokenizer import build_tokenizer  # noqa: E402
from lingua.transformer import Attention  # noqa: E402
from sdm_testlib import bits_equal, requires_cuda  # noqa: E402
from sparse_delta_memory import SparseDeltaMemory, SparseDeltaMemoryArgs  # noqa: E402

pytestmark = requires_cuda

PROMPTS = ["The quick brown fox", "hi", "A longer prompt that spans several chunks. " * 6]
STEPS = 24


@pytest.fixture(scope="module")
def tok():
    return build_tokenizer("bytes")


def hybrid_model(tok, sdm_layers):
    """SWA, full attention and SDM layers (with a learned bank), random weights, bf16."""
    sdm = SparseDeltaMemoryArgs(num_heads=2, slots_per_head=1024, num_reads=16, num_writes=16,
                                chunk_size=64, output_gate=True, normalize_readings=True)
    args = LMTransformerArgs(
        dim=256, n_layers=2 + len(sdm_layers), n_heads=4, vocab_size=tok.n_words,
        max_seqlen=512, init_base_std=0.02, swa_at=[0], attn_at=[1], sdm_at=sdm_layers,
        swa_window=16, swa_n_heads=4, swa_head_dim=64, sdm_args=sdm)
    torch.manual_seed(0)
    model = LMTransformer(args).cuda()
    model.init_weights()
    return model.to(torch.bfloat16).eval()


def generator(model, tok, graph=True, max_gen_len=STEPS + 1):
    args = PackedCausalTransformerGeneratorArgs(
        temperature=0.0, max_gen_len=max_gen_len, max_tokens=512, dtype="bf16",
        sdm_decode_cuda_graph=graph)
    return PackedCausalTransformerGenerator(args, model, tok)


def mutable_state(gen):
    """Every tensor a decode step updates in place."""
    state = [gen.current_tok_id]
    for m in gen.model.modules():
        if isinstance(m, Attention):
            state += [m.kv_cache.k_cache, m.kv_cache.v_cache]
        elif isinstance(m, SparseDeltaMemory):
            state.append(m._gen_cache.memory)
    return state


def decode(gen, step, token, n):
    """Greedy tokens and logits of ``n`` decode steps from ``token``."""
    tokens, logits = [], []
    for _ in range(n):
        out = step(token)
        logits.append(out.detach().clone())
        token = sample_tokens(out.clone(), 0.0)
        tokens.append(token.item())
    return tokens, logits


def sdm_states(gen):
    return [m._gen_cache for m in gen.model.modules() if isinstance(m, SparseDeltaMemory)]


@torch.inference_mode()
def eager_vs_graph(gen, prompt):
    """Prefills ``prompt``, then decodes STEPS tokens eagerly and, from the same state, through
    ``decode_step`` as generate() calls it; returns both (tokens, logits). Both paths advance
    each SDM state's length by one per step."""
    tokens, lengths = pack_prompts([gen.tokenizer.encode(prompt, add_bos=True, add_eos=False)])
    tokens, lengths = tokens.cuda(), lengths.cuda()
    first = sample_tokens(gen.prefill(tokens.unsqueeze(0), lengths), 0.0)
    first = first[:, lengths.cumsum(0) - 1]
    saved = [t.clone() for t in mutable_state(gen)]
    prefilled = [state._cache_len for state in sdm_states(gen)]
    eager = decode(gen, gen.generate_next_token, first, STEPS)
    assert [state._cache_len for state in sdm_states(gen)] == [n + STEPS for n in prefilled]
    for t, s in zip(mutable_state(gen), saved):
        t.copy_(s)
    for state, n in zip(sdm_states(gen), prefilled):
        state._cache_len = n
    decode_tensors = gen._decode_tensors()  # once per prompt, as generate()
    graphed = decode(gen, lambda t: gen.decode_step(t, decode_tensors), first, STEPS)
    assert [state._cache_len for state in sdm_states(gen)] == [n + STEPS for n in prefilled]
    for m in gen.model.modules():
        if hasattr(m, "_gen_cache"):
            del m._gen_cache
    return eager, graphed


@pytest.mark.parametrize("sdm_layers", [[2, 3], [2]], ids=["two_sdm", "one_sdm"])
def test_graph_decode_is_bitwise_eager(tok, sdm_layers):
    model = hybrid_model(tok, sdm_layers)
    gen = generator(model, tok)
    for prompt in PROMPTS:
        (tok_e, log_e), (tok_g, log_g) = eager_vs_graph(gen, prompt)
        assert tok_e == tok_g, prompt
        assert all(bits_equal(a, b) for a, b in zip(log_e, log_g)), prompt
    # The first graphed step runs eager; the next is captured, then every step of every
    # prompt replays one graph.
    assert gen.decode_graph.stats == dict(eager=1, capture=1, replay=len(PROMPTS) * STEPS - 1)


def test_a_settings_change_recaptures(tok, monkeypatch):
    """The step is captured again when a setting it bakes in (here TF32 matmuls) changes."""
    model = hybrid_model(tok, [2, 3])
    gen = generator(model, tok)
    eager_vs_graph(gen, PROMPTS[0])
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32",
                        not torch.backends.cuda.matmul.allow_tf32)
    (tok_e, log_e), (tok_g, log_g) = eager_vs_graph(gen, PROMPTS[1])
    assert tok_e == tok_g and all(bits_equal(a, b) for a, b in zip(log_e, log_g))
    assert gen.decode_graph.stats["capture"] == 2


def test_each_prompt_starts_from_the_initial_bank(tok):
    """The decode state that the graph reads at a fixed address is reset in place to the
    layer's initial bank for every prompt."""
    model = hybrid_model(tok, [2, 3])
    gen = generator(model, tok)
    eager_vs_graph(gen, PROMPTS[0])
    with torch.inference_mode():  # as generate()
        for m in model.modules():
            if isinstance(m, SparseDeltaMemory):
                state = gen._sdm_state(m)
                fresh = m.create_kv_cache(bsz=1, seq_len=0, dtype=gen.dtype, device=gen.device)
                assert state is gen._sdm_states[m] and state._cache_len == 0
                assert bits_equal(state.memory, fresh.memory)


def test_a_parameter_replaced_between_prompts_recaptures(tok):
    """generate() takes the layout of the step's tensors once per prompt: a parameter replaced
    between two calls is read at its new address."""
    model = hybrid_model(tok, [2, 3])
    gen = generator(model, tok)
    gen.generate(PROMPTS[:1])
    lin = next(m for m in model.modules() if isinstance(m, nn.Linear))
    lin.weight = nn.Parameter(lin.weight.detach().clone())
    gen.generate(PROMPTS[1:2])
    assert gen.decode_graph.stats["capture"] == 2


def test_autocast_decodes_eager(tok):
    """A step under autocast runs eager (a capture would bake in its casts)."""
    model = hybrid_model(tok, [2, 3])
    gen = generator(model, tok)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        (tok_e, log_e), (tok_g, log_g) = eager_vs_graph(gen, PROMPTS[0])
    assert tok_e == tok_g and gen.decode_graph.stats == dict(eager=STEPS)


def test_generate_replays_one_graph(tok):
    """generate() over several prompts and calls captures once and leaves no decode state on
    the model."""
    model = hybrid_model(tok, [2, 3])
    gen = generator(model, tok)
    for _ in range(2):
        out, _, _ = gen.generate(PROMPTS)
        assert len(out) == len(PROMPTS) and all(isinstance(g, str) for g in out)
    st = gen.decode_graph.stats
    assert st["eager"] == 1 and st["capture"] == 1 and st["replay"] > len(PROMPTS)
    assert not any(hasattr(m, "_gen_cache") for m in model.modules())


def _refusing_capture(monkeypatch):
    class FailingGraph(torch.cuda.graph):
        def __enter__(self):
            raise RuntimeError("capture refused")
    monkeypatch.setattr(torch.cuda, "graph", FailingGraph)


def _out_of_memory_graph(monkeypatch):
    def graph():
        raise torch.cuda.OutOfMemoryError("injected")
    monkeypatch.setattr(torch.cuda, "CUDAGraph", graph)


@pytest.mark.parametrize("failure", [_refusing_capture, _out_of_memory_graph],
                         ids=["refused", "out_of_memory"])
def test_failed_capture_decodes_eager(tok, monkeypatch, failure):
    """A decode step that cannot be captured (refused, or no memory for the graph) leaves
    decoding eager."""
    model = hybrid_model(tok, [2, 3])
    gen = generator(model, tok)
    failure(monkeypatch)
    (tok_e, log_e), (tok_g, log_g) = eager_vs_graph(gen, PROMPTS[0])
    assert tok_e == tok_g and all(bits_equal(a, b) for a, b in zip(log_e, log_g))
    assert gen.decode_graph.failed and gen.decode_graph.stats == dict(eager=STEPS)


def test_models_without_sdm_have_no_decode_graph(tok):
    args = LMTransformerArgs(dim=128, n_layers=2, n_heads=4, vocab_size=tok.n_words,
                             max_seqlen=128, attn_at=[0, 1])
    model = LMTransformer(args).cuda().to(torch.bfloat16).eval()
    assert generator(model, tok).decode_graph is None
