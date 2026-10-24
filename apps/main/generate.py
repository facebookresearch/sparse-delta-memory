# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import List, Optional

import torch
from torch import nn
from tqdm import tqdm

from omegaconf import OmegaConf
from torch.nn import functional as F
import xformers

from apps.main.transformer import (
    LMTransformer,
    LMTransformerArgs,
    sliding_window_causal_mask_mod,
)
from lingua.args import dataclass_from_dict
from lingua.checkpoint import CONSOLIDATE_NAME
from lingua.tokenizer import Tokenizer, build_tokenizer
from lingua.transformer import (
    Attention,
    causal_mask,
    generate_doc_mask_mod,
    lengths_to_local_ids,
    lengths_to_start_ids,
)
from sparse_delta_memory import SparseDeltaMemory
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask

logger = logging.getLogger(__name__)


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def sample_top_k(probs, k):
    topk_value, _ = torch.topk(probs, k)  # batch_sz x topk
    min_value_top_k = topk_value[:, [-1]]
    probs[probs < min_value_top_k] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None):
    shape = logits.shape
    logits = logits.flatten(end_dim=-2)
    if temperature > 0.0:
        probs = torch.softmax(logits / temperature, dim=-1)

        if top_p is not None:
            next_token = sample_top_p(probs, top_p)
        elif top_k is not None:
            next_token = sample_top_k(probs, top_k)
        else:
            next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(logits, dim=-1)
    return next_token.view(shape[:-1])


def pack_prompts(prompts: List[int]):
    res = []
    lengths = []
    for i, p in enumerate(prompts):
        p = torch.tensor(p, dtype=torch.long)
        l = p.size(0)
        res.append(p)
        lengths.append(l)
    lengths = torch.tensor(lengths, dtype=torch.long)
    res = torch.cat(res)
    return res, lengths


def batch_prompts(prompts, max_elements, lengths=None):
    batches = []
    current_batch = []
    current_count = 0

    for i in range(len(prompts)):
        prt = prompts[i]
        prompt_size = len(prt) if lengths is None else lengths[i]
        if current_count + prompt_size <= max_elements:
            current_batch.append(prt)
            current_count += prompt_size
        else:
            if current_batch:  # Add the current batch to batches
                batches.append(current_batch)
            # Start a new batch with the current prompt
            current_batch = [prt]
            current_count = prompt_size

    # Add the last batch if it contains any prompts
    if current_batch:
        batches.append(current_batch)

    return batches


class KVCache(nn.Module):
    def __init__(self, bsz, seqlen, n_heads, head_dim, dtype, device):
        super().__init__()
        shape = (bsz, seqlen, n_heads, head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.offset = 0

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.offset = 0

    def update(self, k_val, v_val, tok_idx):
        # input_pos: [B], k_val: [B, S, H, D]
        self.k_cache.index_copy_(1, self.offset + tok_idx, k_val)
        self.v_cache.index_copy_(1, self.offset + tok_idx, v_val)
        return self.k_cache, self.v_cache


def _decode_settings():
    """The global settings that a captured decode step bakes in: matmul precision and the
    enabled attention backends."""
    b = torch.backends
    return (b.cuda.matmul.allow_tf32, b.cuda.matmul.allow_bf16_reduced_precision_reduction,
            b.cudnn.allow_tf32, b.cuda.flash_sdp_enabled(), b.cuda.mem_efficient_sdp_enabled(),
            b.cuda.math_sdp_enabled(), b.cuda.cudnn_sdp_enabled())


class _DecodeGraph:
    """The decode step of an SDM model, replayed from one CUDA graph.

    SDM models are generated one prompt at a time, so every decode step has the same shapes,
    and the generator keeps the tensors a step reads or updates (KV caches and their offset,
    SDM banks, position and document ids) at fixed addresses across prompts. The first step
    runs eager on a side stream (it builds kernels and lazy state); the next one is captured
    and every later step copies its token in and replays. The step is captured again when
    one of those tensors or a parameter or buffer of the model has another address, shape,
    stride or dtype, or a setting of ``_decode_settings`` has changed (``layout``, taken with
    the autocast state once per prompt: neither may change while a prompt is generated).
    Prompts under autocast run eager, and when a capture fails, decoding stays eager.
    """

    def __init__(self):
        self.graph = self.key = self.static_token = self.static_out = None
        self.warm = self.failed = False
        self.stats = Counter()

    def step(self, fn, token, layout, sdm_states):
        if self.failed or layout[1]:  # autocast
            self.stats["eager"] += 1
            return fn(token)
        if not self.warm:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                out = fn(token)
            torch.cuda.current_stream().wait_stream(side)
            self.warm = True
            self.stats["eager"] += 1
            return out
        if layout is not self.key:  # a new prompt: its layout is compared once
            if self.graph is None or layout != self.key:
                if not self._capture(fn, token, sdm_states):
                    self.stats["eager"] += 1
                    return fn(token)
            self.key = layout
        self.static_token.copy_(token)
        self.graph.replay()
        for state in sdm_states:  # the replayed step does not run the Python bookkeeping
            state._cache_len += 1
        self.stats["replay"] += 1
        return self.static_out

    def _capture(self, fn, token, sdm_states):
        lengths = [state._cache_len for state in sdm_states]
        self.graph = None
        try:
            self.static_token, graph = token.clone(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):  # records the step; nothing runs until a replay
                self.static_out = fn(self.static_token)
        except Exception as exc:
            self.failed, self.graph = True, None
            logger.warning("Capturing the SDM decode step failed (%s: %s); decoding runs "
                           "eager.", type(exc).__name__, exc)
            return False
        finally:
            for state, n in zip(sdm_states, lengths):
                state._cache_len = n
        self.graph = graph
        self.stats["capture"] += 1
        return True


@dataclass
class PackedCausalTransformerGeneratorArgs:
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    max_gen_len: int = 512  # Maximum number of tokens to generate
    max_tokens: int = 1024  # Maximum number of tokens that can go through the model
    max_prompt_len: Optional[int] = None
    until: List[str] = field(default_factory=list)
    compile_prefilling: bool = False
    reduce_generation_overhead: bool = False
    # SDM models: replay the single-token decode step from a CUDA graph (see _DecodeGraph).
    sdm_decode_cuda_graph: bool = True
    # Decode attention on the memory-efficient SDPA kernel, math as the fallback, instead of PyTorch's choice: the
    # decode is then bitwise repeatable on H100 (PyTorch's cuDNN attention kernel there is not), but slower.
    deterministic_attention: bool = False
    show_progress: bool = False
    dtype: Optional[str] = "bf16"
    device: Optional[str] = "cuda"


class PackedCausalTransformerGenerator:
    def __init__(
        self,
        cfg: PackedCausalTransformerGeneratorArgs,
        model: nn.Module,
        tokenizer: Tokenizer,
    ):
        """
        This class wraps a causal transformer model with its corresponding tokenizer
        and provides an efficient way to pack prompts together and do generation on
        the packed sequence.

        For example, if we had the prompts "Hello, I am a " and "Initiating calibration "
        Then this class will concatenate those sequence (pack them together)
        "Hello, I am a Initiating calibration"
        And make the necessary attention masks such that a sequence only attends to itself
        during prefilling and generation.

        This class creates a fixed size cache of size max_tokens or sum of prompt sizes
        + the max number of generated tokens per sequence.
        """
        self.model = model
        # Sliding-window attention layers need windowed masks in both prefill and decode
        # (built in setup_prefilling / generate_next_token). The KV cache stays full size;
        # the window is enforced purely by the attention mask.
        self.has_swa = "swa" in getattr(model, "layer_types", [])
        self.swa_window = getattr(model, "swa_window", None)
        if self.has_swa:
            assert self.swa_window is not None and self.swa_window > 0, (
                "model has sliding-window (swa) layers but swa_window is unset"
            )
        # SDM's recurrent state is per-sequence and cannot be packed with the generator's
        # block-diagonal-mask trick, so SDM models are generated one prompt at a time (see
        # generate()); each SDM layer keeps a decode cache (installed in clear_cache).
        self.has_sdm = "sdm" in getattr(model, "layer_types", [])
        self.tokenizer = tokenizer
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.top_k = cfg.top_k

        self.max_gen_len = cfg.max_gen_len
        self.max_tokens = cfg.max_tokens
        self.max_prompt_len = cfg.max_prompt_len
        self.until = cfg.until
        self.max_until_size = max([len(e) for e in self.until]) if self.until else 1
        self.device = cfg.device
        self.deterministic_attention = cfg.deterministic_attention

        # Compile if necessary. The SDM layer updates its cache in place, which this
        # generator's reduce-overhead (CUDA-graph) compilation does not support, so
        # compilation is disabled for SDM models; their decode step replays a CUDA graph of
        # its own instead (sdm_decode_cuda_graph).
        self.decode_graph = (
            _DecodeGraph() if self.has_sdm and cfg.sdm_decode_cuda_graph else None
        )
        self._sdm_states = {}  # per SDM layer, the decode state reused across prompts
        self._static = {}  # per-prompt index tensors kept at fixed addresses for the graph
        self.prefill = torch.compile(
            self.prefill, disable=not cfg.compile_prefilling or self.has_sdm
        )
        self.generate_next_token = torch.compile(
            self.generate_next_token,
            mode="reduce-overhead",
            disable=not cfg.reduce_generation_overhead or self.has_sdm,
        )

        self.show_progress = cfg.show_progress
        self.dtype = dict(fp32=torch.float32, bf16=torch.bfloat16)[cfg.dtype]

        self.prefill_doc_id, self.prefill_tok_id = None, None
        self.padded_doc_id, self.padded_tok_id = None, None
        self.current_doc_id, self.current_tok_id = None, None
        self.padded_doc_start = None
        self.prefill_mask = None

    def clear_cache(self, offset):
        for module in self.model.modules():
            if isinstance(module, Attention):
                if not hasattr(module, "kv_cache"):
                    module.kv_cache = KVCache(
                        1,
                        self.max_tokens,
                        module.n_kv_heads,
                        module.head_dim,
                        self.dtype,
                        self.device,
                    )
                module.kv_cache.offset = offset
            elif isinstance(module, SparseDeltaMemory):
                # Fresh per-prompt decode state: memory bank reset to the learned initial
                # memory. The block threads this cache (as `_gen_cache`) through prefill
                # and each decode step. (SDM is generated one prompt at a time, so bsz=1.)
                module._gen_cache = self._sdm_state(module)

    def _sdm_state(self, module):
        """A decode state at the layer's initial bank. With a decode graph the same state,
        reset in place, serves every prompt: the graph reads its bank at a fixed address."""
        fresh = module.create_kv_cache(bsz=1, seq_len=0, dtype=self.dtype, device=self.device)
        if self.decode_graph is None:
            return fresh
        state = self._sdm_states.get(module)
        if state is None or state.memory.shape != fresh.memory.shape or (
                state.memory.dtype != fresh.memory.dtype):
            state = self._sdm_states[module] = fresh
        else:
            state.memory.copy_(fresh.memory)
            state._cache_len = 0
        return state

    def _persistent(self, name, value):
        """``value`` itself, or with a decode graph a tensor kept across prompts that holds a
        copy of it (the graph reads fixed addresses)."""
        if self.decode_graph is None:
            return value
        t = self._static.get(name)
        if t is None or t.shape != value.shape or t.dtype != value.dtype or (
                t.device != value.device):
            t = self._static[name] = value.clone()
        else:
            t.copy_(value)
        return t

    def _decode_tensors(self):
        """The layout: the ``_decode_settings``, the autocast state and the address, shape,
        stride and dtype of every tensor the decode step reads or updates (parameters and
        buffers, KV caches and their offset, SDM banks, position and document ids); and the SDM
        decode states."""
        tensors = [self.current_tok_id, self.current_doc_id, self.padded_doc_id,
                   self.padded_tok_id, *self.model.parameters(), *self.model.buffers()]
        states = []
        for module in self.model.modules():
            if isinstance(module, Attention):
                tensors.append(module.kv_cache.offset)
            elif isinstance(module, SparseDeltaMemory):
                states.append(module._gen_cache)
                tensors.append(module._gen_cache.memory)
        layout = (_decode_settings(), torch.is_autocast_enabled("cuda"),
                  tuple((t.data_ptr(), t.shape, t.stride(), t.dtype) for t in tensors))
        return layout, states

    def decode_step(self, current_token, decode_tensors=None):
        """``generate_next_token``, replayed from the decode graph when there is one and the
        step fits it (one sequence). ``decode_tensors`` is ``_decode_tensors()`` taken after
        this prompt's prefill (by default, taken now)."""
        if self.decode_graph is None or tuple(current_token.shape) != (1, 1):
            return self.generate_next_token(current_token)
        layout, states = decode_tensors or self._decode_tensors()
        return self.decode_graph.step(self.generate_next_token, current_token, layout, states)

    @torch.compiler.disable
    def setup_prefilling(self, lengths: torch.Tensor):
        # The KV cache is a fixed size tensor of size max_tokens that we need
        # to update in order to do correct autoregressive generation.

        # Here we will generate token by token but on multiple sequences
        # at once. To do so, we need to have an attention mask that makes
        # each sequence independent.

        # Each sequence will write to its allocated space in the KV Cache.
        # We allocate len(seq) + max_gen_len to each sequence in the cache.

        # We will generate max_gen_len for each document
        padded_lengths = lengths + self.max_gen_len
        max_tokens = self.max_tokens or padded_lengths.sum().item()
        # The last document might have more padding to fill up to max_tokens
        padded_lengths[-1] += max_tokens - padded_lengths.sum()

        # This is the start index in the cache for each document
        self.padded_doc_start = lengths_to_start_ids(padded_lengths)
        # For example with ab--123--cdef--
        # this would be 0, 4, 9 if max_gen_len is 2

        # We repeat interleave to align with tokens for prefilling
        # Ex: ab--123--cdef--
        #     000044444999999
        prefill_offset = torch.repeat_interleave(self.padded_doc_start, lengths)
        # This offset will make sure the tokens are written to the
        # correct positions in the cache during prefilling

        # We either init the cache or clear it by resetting the offset to prefill_offset
        self.clear_cache(prefill_offset)

        # The prefilling mask looks like the following for
        # the two packed sequences ab and 123 : ab123
        # Where spaces are empty cache positions
        #                 keys
        #                ab---123---
        #   queries    a 10000000000
        #              b 11000000000
        #              1 00000100000
        #              2 00000110000
        #              3 00000111000
        # We make sure to skip the empty cache positions
        # and only attend to positions within the same sequence
        doc_mask_mod = generate_doc_mask_mod(causal_mask, lengths, padded_lengths)
        attn_prefill_mask = create_block_mask(
            doc_mask_mod, 1, None, lengths.sum(), max_tokens
        )
        if self.has_swa:
            # SWA layers use a windowed variant of the same doc-causal mask.
            swa_doc_mask_mod = generate_doc_mask_mod(
                sliding_window_causal_mask_mod(self.swa_window), lengths, padded_lengths
            )
            swa_prefill_mask = create_block_mask(
                swa_doc_mask_mod, 1, None, lengths.sum(), max_tokens
            )
            self.prefill_mask = {"attn": attn_prefill_mask, "swa": swa_prefill_mask}
        else:
            self.prefill_mask = attn_prefill_mask

        # This creates the prefilling token ids which look like
        # the following for the packed sequence abcdefg1234
        # abcdefg1234
        # 01234560123
        # The token id gives us the position within each sequence
        # This is used to compute ROPE and to update the cache
        # At each forward pass the current tokens are written to
        # offset + tok_id
        self.prefill_doc_id, self.prefill_tok_id = lengths_to_local_ids(lengths)

        # This creates the padded token and document ids
        # which look like the following for the packed sequence ab123
        #               ab---123---               ab---123---
        # padded_doc_id 00000111111 padded_tok_id 01234012345
        # This will later be useful for the attention mask at generation
        padded_doc_id, padded_tok_id = lengths_to_local_ids(padded_lengths)
        self.padded_doc_id = self._persistent("padded_doc_id", padded_doc_id)
        self.padded_tok_id = self._persistent("padded_tok_id", padded_tok_id)

    @torch.compiler.disable
    def setup_generation(self, lengths):
        # KV Cache offset is set to the start of the padded documents
        offset = self._persistent("kv_offset", self.padded_doc_start)
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.kv_cache.offset = offset
        # The token ids during generations correspond to the lengths of each doc
        # current_tok_id will be incremented during generation
        self.current_tok_id = self._persistent("current_tok_id", lengths.clone())
        # Since we're generating one token per document
        # the document id is just an arange
        self.current_doc_id = self._persistent(
            "current_doc_id", torch.arange(lengths.size(0), device=lengths.device)
        )

    # From here on some methods for generation
    def prefill(self, tokens: torch.Tensor, lengths: torch.Tensor):
        # Prefilling is done by taking multiple packed sequences and
        # doing block diagonal attention on them so they remain independent
        self.setup_prefilling(lengths=lengths)
        prefill_out = self.model.forward(
            tokens,
            tok_idx=self.prefill_tok_id,
            mask=self.prefill_mask,
            attn_impl="flex_attention",
        )
        self.setup_generation(lengths=lengths)
        return prefill_out

    def generate_next_token(self, current_token):
        # Since we're doing generation with multiple sequences at once
        # we need to ignore tokens and cache entries from other sequences
        # or in the future.
        # Example mask :
        #                  keys
        #                abc--1234--
        #   queries    c 11100000000
        #              4 00000111100

        # mask shape : (n_seqs, cache_size)
        doc_mask = self.current_doc_id.unsqueeze(1) == self.padded_doc_id.unsqueeze(0)
        caus_mask = self.current_tok_id.unsqueeze(1) >= self.padded_tok_id.unsqueeze(0)
        mask = doc_mask & caus_mask
        if self.has_swa:
            # SWA layers additionally attend only to the `swa_window` most-recent keys.
            within_window = (
                self.current_tok_id.unsqueeze(1) - self.padded_tok_id.unsqueeze(0)
            ) < self.swa_window
            mask = {"attn": mask, "swa": mask & within_window}
        # deterministic_attention: PyTorch's backend switches are process-wide, so during a decode step they also apply
        # to other threads' SDPA calls (the generator decodes on one thread).
        if self.deterministic_attention:
            attention = sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH], set_priority=True)
        else:
            attention = nullcontext()
        with attention:
            out = self.model.forward(
                current_token,
                tok_idx=self.current_tok_id,  # n_seqs
                mask=mask,
                attn_impl="sdpa",
            )
        self.current_tok_id += 1
        return out

    @torch.inference_mode()
    def generate(self, prompts):
        # Tokenize
        prompts = [
            self.tokenizer.encode(p, add_bos=True, add_eos=False) for p in prompts
        ]
        # Truncate
        max_seqlen = (
            self.max_tokens
            if not hasattr(self.model, "max_seqlen")
            else self.model.max_seqlen
        )
        max_prompt_len = self.max_prompt_len or min(
            max_seqlen - self.max_gen_len, self.max_tokens - self.max_gen_len
        )
        prompts = [p[-max_prompt_len:] for p in prompts]
        # Account for the generation in lengths
        padded_lengths = [len(p) + self.max_gen_len for p in prompts]
        generation = []
        loglikelihood = []
        greedy = []
        if self.has_sdm:
            # SDM's per-sequence recurrent memory cannot be packed into one block-diagonal
            # sequence, so process a single prompt per batch (correct; lower throughput).
            it = ([p] for p in prompts)
        else:
            it = batch_prompts(prompts, self.max_tokens, lengths=padded_lengths)
        if self.show_progress:
            it = tqdm(it)
        try:
            for batch in it:
                n_seqs = len(batch)
                generated_tokens = [[] for _ in range(n_seqs)]
                is_done = [False for _ in range(n_seqs)]
                packed_batch, lengths = pack_prompts(batch)
                packed_batch, lengths = packed_batch.cuda(), lengths.cuda()
                n_seqs = lengths.size(0)

                # Prefilling cache
                prompt_logits = self.prefill(packed_batch.unsqueeze(0), lengths)
                # Selecting last token in each prompt
                all_tokens = sample_tokens(
                    prompt_logits, self.temperature, self.top_p, self.top_k
                )
                start_token = all_tokens[:, lengths.cumsum(0) - 1]

                for seq_id, tok in enumerate(start_token.squeeze(0).tolist()):
                    generated_tokens[seq_id].append(tok)

                current_token = start_token
                decode_tensors = (
                    self._decode_tensors() if self.decode_graph is not None else None
                )
                for i in range(1, self.max_gen_len):

                    next_logits = self.decode_step(current_token, decode_tensors)
                    next_token = sample_tokens(
                        next_logits.clone(), self.temperature, self.top_p, self.top_k
                    )

                    for seq_id, tok in enumerate(next_token.squeeze(0).tolist()):
                        if not is_done[seq_id]:
                            generated_tokens[seq_id].append(tok)
                            current_end_str = self.tokenizer.decode(
                                generated_tokens[seq_id][-self.max_until_size :]
                            )
                            contains_end_string = any(
                                [e in current_end_str for e in self.until]
                            )
                            is_done[seq_id] = (
                                contains_end_string or tok == self.tokenizer.eos_id
                            )
                    if all(is_done):
                        break

                    current_token = next_token

                generation.extend([self.tokenizer.decode(g) for g in generated_tokens])

                for p, logit in zip(
                    batch, prompt_logits.squeeze(0).split(lengths.tolist())
                ):
                    x = logit[:-1]
                    y = torch.tensor(p[1:], device=x.device)
                    loglikelihood.append(-F.cross_entropy(x, y, reduction="none").cpu())
                    greedy.append((x.argmax(dim=-1) == y).cpu())
        finally:
            # Drop per-layer SDM decode state so the model object is safe to reuse for a
            # plain forward/training pass afterwards (would otherwise resume from stale
            # generated memory). Attention KV caches follow the generator's own reset path.
            for module in self.model.modules():
                if isinstance(module, SparseDeltaMemory) and hasattr(module, "_gen_cache"):
                    del module._gen_cache

        return generation, loglikelihood, greedy


def load_consolidated_model_and_tokenizer(
    consolidated_path,
    model_cls=LMTransformer,
    model_args_cls=LMTransformerArgs,
):
    ckpt_path = Path(consolidated_path)
    config = ckpt_path / "params.json"
    config = OmegaConf.load(config)

    param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
        config.distributed.model_dtype
    ]
    model_args = dataclass_from_dict(model_args_cls, config.model, strict=False)
    tokenizer = build_tokenizer(config.data.tokenizer.name, config.data.tokenizer.path)
    model = model_cls(model_args)
    st_dict = torch.load(ckpt_path / CONSOLIDATE_NAME, weights_only=True)
    model.load_state_dict(st_dict["model"])
    model = model.cuda().eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)
    return model, tokenizer, config


def main():
    # Load CLI arguments (overrides) and combine with a YAML config
    cfg = OmegaConf.from_cli()
    gen_cfg = dataclass_from_dict(
        PackedCausalTransformerGeneratorArgs, cfg, strict=False
    )
    print(cfg)

    model, tokenizer, _ = load_consolidated_model_and_tokenizer(cfg.ckpt)

    generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)

    # Allow multiple prompts
    prompts = []
    while True:
        prompt = input("Enter a prompt (or press enter to finish): ")
        if not prompt:
            break
        prompts.append(prompt)

    # Start generation
    start_time = time.time()
    generation, loglikelihood, greedy = generator.generate(prompts)
    end_time = time.time()

    # Calculate tokens per second
    total_tokens = sum(len(tokenizer.encode(gen, False, False)) for gen in generation)
    tokens_per_second = total_tokens / (end_time - start_time)

    # Display the results
    for i, gen in enumerate(generation):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Generated Text: {gen}")

    print(f"\nTokens per second: {tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
