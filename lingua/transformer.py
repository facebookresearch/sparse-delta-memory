# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union, Tuple, List, Any

import torch
from torch import nn
from torch.nn import functional as F
from xformers.ops import fmha, AttentionBias
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
    _mask_mod_signature,
)

from lingua import probe
from lingua.sparse_delta_memory import SparseDeltaMemory, SparseDeltaMemoryArgs

flex_attention_comp = torch.compile(flex_attention)


class InitStdFactor(Enum):
    DISABLED = "disabled"  # Init std is divided by 1.0
    GLOBAL_DEPTH = "global_depth"  # Init std is divided by sqrt(2*n_layers)
    CURRENT_DEPTH = "current_depth"  # Init std is divided by sqrt(2*depth)
    DIM_RATIO = "dim_ratio"  # Init std is divided by model_dim/4096


@dataclass
class BaseTransformerArgs:
    dim: int = 512
    n_layers: int = 8
    head_dim: Optional[int] = None
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None

    ffn_dim_multiplier: Optional[float] = None

    multiple_of: int = 256

    norm_eps: float = 1e-5

    rope_theta: float = 10000.0

    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"

    max_seqlen: int = 1024

    # ---- Explicit per-layer sequence-mixer assignment ----
    # Every layer is exactly one of: full causal attention ("attn"), sliding-window
    # attention ("swa"), or Sparse-Delta-Memory ("sdm"). Placement is declared explicitly
    # via these three lists; each is either the string "all" or a list of layer indices
    # (a comma-separated string is also accepted). Together they must partition
    # range(n_layers) exactly — no gaps, no overlaps. When all three are empty,
    # BaseTransformer falls back to all-"attn" (legacy behaviour for callers that never set
    # them, e.g. apps/aunet, apps/mtp); apps/main.LMTransformer requires an explicit split.
    # Typed Any (not Union[str, List[int]]) so OmegaConf's structured configs accept either
    # a YAML list or the "all"/comma-string form; resolve_layer_types normalizes both.
    attn_at: Any = field(default_factory=list)
    swa_at: Any = field(default_factory=list)
    sdm_at: Any = field(default_factory=list)

    # Sliding-window attention ("swa") layers use their own window + head configuration,
    # independent of the full-attention layers. Unset head fields fall back to the
    # full-attention values (swa_head_dim defaults to dim // swa_n_heads).
    swa_window: Optional[int] = None
    swa_n_heads: Optional[int] = None
    swa_n_kv_heads: Optional[int] = None
    swa_head_dim: Optional[int] = None

    # Attention output gating (applies to ALL attention layers — both full and SWA — when
    # set): sigmoid(w_gate(x)) * attention_output before the output projection. Off by default.
    attn_gate: bool = False

    # Sparse-Delta-Memory hyper-parameters (placement is controlled by sdm_at above).
    sdm_args: SparseDeltaMemoryArgs = field(default_factory=SparseDeltaMemoryArgs)


LAYER_TYPES = ("attn", "swa", "sdm")


def _as_layer_index(x, name: str) -> int:
    """Strictly convert a single spec element to an int layer index.

    Rejects booleans, non-integral floats, and non-numeric strings so that a mistyped
    (Any-typed) config value fails loudly instead of being silently coerced.
    """
    if isinstance(x, bool):
        raise ValueError(f"{name}: boolean {x!r} is not a valid layer index")
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if not x.is_integer():
            raise ValueError(f"{name}: non-integer layer index {x!r}")
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "" or not s.lstrip("-").isdigit():
            raise ValueError(f"{name}: invalid layer index {x!r}")
        return int(s)
    raise ValueError(f"{name}: invalid layer index {x!r} of type {type(x).__name__}")


def _check_layer_bounds(idxs, n_layers: int, name: str) -> set:
    oob = sorted({i for i in idxs if i < 0 or i >= n_layers})
    if oob:
        raise ValueError(
            f"{name}: layer indices {oob} out of range for n_layers={n_layers}"
        )
    dups = sorted({i for i in idxs if idxs.count(i) > 1})
    if dups:
        raise ValueError(f"{name}: duplicate layer indices {dups}")
    return set(idxs)


def _normalize_layer_spec(spec, n_layers: int, name: str) -> set:
    """Normalize a layer-placement spec into a set of int indices.

    Accepts the sentinel "all" (-> every layer), a list/tuple of ints, a single int,
    a comma-separated string like "0,2,4", or an empty value (-> empty set).
    """
    if spec is None:
        return set()
    if isinstance(spec, bool):
        raise ValueError(f"{name}: boolean {spec!r} is not a valid layer spec")
    if isinstance(spec, str):
        s = spec.strip()
        if s == "":
            return set()
        if s.lower() == "all":
            return set(range(n_layers))
        parts = [tok.strip() for tok in s.split(",")]
        if any(p == "" for p in parts):
            raise ValueError(f"{name}: empty token in layer spec {spec!r}")
        idxs = [_as_layer_index(p, name) for p in parts]
        return _check_layer_bounds(idxs, n_layers, name)
    if isinstance(spec, int):
        return _check_layer_bounds([_as_layer_index(spec, name)], n_layers, name)
    try:
        elems = list(spec)  # list / tuple / OmegaConf ListConfig
    except TypeError as e:
        raise ValueError(f"{name}: could not parse layer spec {spec!r}: {e}")
    idxs = [_as_layer_index(x, name) for x in elems]
    return _check_layer_bounds(idxs, n_layers, name)


def resolve_layer_types(n_layers: int, attn_at, swa_at, sdm_at) -> List[str]:
    """Resolve an explicit, exhaustive per-layer sequence-mixer assignment.

    Returns a list of length ``n_layers`` with each entry in ``{"attn","swa","sdm"}``.
    ``attn_at`` / ``swa_at`` / ``sdm_at`` are each "all", a list of indices, a comma
    string, or empty; together they must partition ``range(n_layers)`` exactly. When all
    three are empty this defaults to all-"attn" (legacy behaviour for callers that do not
    set them).
    """
    sets = {
        "attn": _normalize_layer_spec(attn_at, n_layers, "attn_at"),
        "swa": _normalize_layer_spec(swa_at, n_layers, "swa_at"),
        "sdm": _normalize_layer_spec(sdm_at, n_layers, "sdm_at"),
    }
    if not (sets["attn"] or sets["swa"] or sets["sdm"]):
        return ["attn"] * n_layers

    overlaps = {}
    for a, b in (("attn", "swa"), ("attn", "sdm"), ("swa", "sdm")):
        common = sets[a] & sets[b]
        if common:
            overlaps[f"{a}&{b}"] = sorted(common)
    if overlaps:
        raise ValueError(f"layer types overlap (a layer has >1 type): {overlaps}")

    covered = sets["attn"] | sets["swa"] | sets["sdm"]
    missing = sorted(set(range(n_layers)) - covered)
    if missing:
        raise ValueError(
            f"layer-type assignment does not cover all {n_layers} layers; unassigned: "
            f"{missing}. Every layer must appear in exactly one of attn_at/swa_at/sdm_at "
            f"(or use 'all')."
        )

    types: List[Optional[str]] = [None] * n_layers
    for t in LAYER_TYPES:
        for i in sets[t]:
            types[i] = t
    return types


def cross_entropy(pred, target, **kwargs):
    return F.nll_loss(
        F.log_softmax(pred.flatten(end_dim=-2).float(), -1),
        target.flatten(end_dim=-1),
        **kwargs,
    )


def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    assert dim == 2, "Only dim=2 is supported. Check the implementation for other dims."
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    cos, sin = freqs.cos(), freqs.sin()

    return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        seq_dim (int): Sequence dimension index.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= seq_dim < ndim
    assert freqs_cis.shape == (
        x.shape[seq_dim],
        x.shape[-3],
        2,
        2,
    ), f"freqs_cis vs x: {(freqs_cis.shape, x.shape)}"
    shape = [
        d if i == seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])
    ] + [2, 2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    seq_dim: int,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    freqs_cis = reshape_for_broadcast(
        freqs_cis, xq_, seq_dim
    ).float()  # S D/2 2 2 -> 1 S 1 D/2 2 2
    xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def lengths_to_start_ids(lengths):
    doc_start = lengths.cumsum(0)
    doc_start = doc_start.roll(1)
    doc_start[0] = 0
    return doc_start


def lengths_to_local_ids(lengths):
    assert lengths.ndim == 1
    nb_seqs = lengths.size(0)
    total_seqlen = lengths.sum()
    # This gives the document id of each token
    doc_id = torch.repeat_interleave(lengths)
    # Compute document start for each document
    doc_start = lengths_to_start_ids(lengths)
    # Compute document start for each token
    doc_start = doc_start[doc_id]
    # Compute the position of each token within each document
    tok_id = torch.arange(total_seqlen, device=lengths.device) - doc_start

    return doc_id, tok_id


def generate_doc_mask_mod(
    mask_mod: _mask_mod_signature,
    lengths: torch.Tensor,
    kv_lengths: Optional[torch.Tensor] = None,
) -> _mask_mod_signature:
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        mask_mod: The mask mod to apply to the documents
        lengths: Lengths of each document

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.

    Example:

    - Square mask
      doc_mask         lengths
      a a b b b c c    2 3 2
    a 1 0 0 0 0 0 0
    a 1 1 0 0 0 0 0
    b 0 0 1 0 0 0 0
    b 0 0 1 1 0 0 0
    b 0 0 1 1 1 0 0
    c 0 0 0 0 0 1 0
    c 0 0 0 0 0 1 1

    """
    kv_lengths = kv_lengths if kv_lengths is not None else lengths
    q_document_id, q_token_id = lengths_to_local_ids(lengths)
    kv_document_id, kv_token_id = lengths_to_local_ids(kv_lengths)
    q_max_idx = lengths.sum() - 1
    kv_max_idx = kv_lengths.sum() - 1

    def doc_mask_mod(b, h, q_idx, kv_idx):
        q_idx_cap = torch.minimum(q_max_idx, q_idx)
        kv_idx_cap = torch.minimum(kv_max_idx, kv_idx)
        valid_idx = (q_idx <= q_max_idx) & (kv_idx <= kv_max_idx)
        same_doc = q_document_id[q_idx_cap] == kv_document_id[kv_idx_cap]
        q_logical = q_token_id[q_idx_cap]
        kv_logical = kv_token_id[kv_idx_cap]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask & valid_idx

    return doc_mask_mod


# Rotary embedding as in xformer, see if torchtrain implementation is not better. Also might be usefull to make it work with batch*seqlen collapsed.
class RotaryEmbedding(torch.nn.Module):
    """
    RotaryEmbedding Module
    """

    def __init__(self, theta: float, head_dim: int, max_seqlen: int = 1024):
        super().__init__()

        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(dim=head_dim, end=max_seqlen, theta=theta),
            persistent=False,
        )

    def reset_parameters(self):
        self.freqs_cis[...] = precompute_freqs_cis(
            dim=self.head_dim, end=self.max_seqlen, theta=self.theta
        )

    def forward(
        self, seqlen: Optional[int] = None, tok_idx: Optional[torch.Tensor] = None
    ):
        """
        Return freqs_cis corresponding to consecutive seqlen positions or the corresponding tok_idx positions
        Args:
            seqlen (int): Contiguous sequence length
            tok_idx (torch.Tensor[int]): Position indices of each token this overrides seqlen

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Embedded input tensor and freqs_cis
        """
        test = (seqlen is not None) or (tok_idx is not None)
        assert test, "Should provide atleast seqlen or tok_idx"
        if tok_idx is not None:
            return self.freqs_cis[tok_idx]
        elif seqlen is not None:
            return self.freqs_cis[0:seqlen]


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        x = probe.log_stats(x, "resid")
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore

class TiedLinear(nn.Module):
    def __init__(self, tied_module: nn.Module) -> None:
        super().__init__()
        self.tied_module = tied_module
        if not hasattr(tied_module, "weight"):
            raise AttributeError(
                "Provided module does not have attribute 'weight'. Please check your tied_module."
            )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.tied_module.weight)

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        attn_gate: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
        )

        # Attention output gating (as in AMAIA mixed-attention): sigmoid(w_gate(x)) applied
        # elementwise to the head-mixed attention output before the output projection. Off
        # by default; enabled per-model via BaseTransformerArgs.attn_gate.
        self.attn_gate = attn_gate
        if attn_gate:
            self.w_gate = nn.Linear(dim, n_heads * head_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        # B S D
        bsz, seq_len, dim = x.shape
        xq = self.wq(x.view_as(x))
        xk = self.wk(x.view_as(x))
        xv = self.wv(x.view_as(x))

        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])

        # This condition helps us be easily compatible
        # with inference by adding a pluggable KVCache
        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        if attn_impl == "flex_attention":
            assert mask is None or isinstance(mask, BlockMask)
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            output = flex_attention_comp(xq, xk, xv, block_mask=mask)
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        elif attn_impl == "fmha":
            assert mask is None or isinstance(mask, AttentionBias)
            output = fmha.memory_efficient_attention(xq, xk, xv, attn_bias=mask)
            # This uses B S H D instead of B H S D of pytorch

        elif attn_impl == "sdpa":
            xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
            assert mask is None or isinstance(mask, (str, torch.Tensor))
            is_causal = (mask == "causal") if isinstance(mask, str) else False
            mask = mask if isinstance(mask, torch.Tensor) else None
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                is_causal=is_causal,
                attn_mask=mask,
            )
            output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D
        else:
            raise NotImplementedError(
                f"Attention implementation {attn_impl} not supported"
            )

        output = output.reshape(output_shape)
        if self.attn_gate:
            output = torch.sigmoid(self.w_gate(x)) * output
        output = self.wo(output)

        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))

        for w in [self.wq, self.wk, self.wv]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )

        if self.attn_gate:
            nn.init.trunc_normal_(
                self.w_gate.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        mp_size: int = 1,
    ):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B S D
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        output = self.w2(F.silu(x1) * x3)
        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        in_init_std = in_init_std
        out_init_std = out_init_std / factor
        for w in [self.w1, self.w3]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=in_init_std,
                a=-3 * in_init_std,
                b=3 * in_init_std,
            )
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )


class TransformerBlock(nn.Module):
    def __init__(self, args: BaseTransformerArgs, layer_id: int = 0, layer_type: str = "attn"):
        super().__init__()

        assert layer_type in LAYER_TYPES, f"unknown layer_type {layer_type!r}"

        self.attn_type = layer_type
        self._is_sdm = layer_type == "sdm"

        if layer_type == "sdm":
            # SDM is the sequence mixer for this layer and manages its own heads/head_dim
            # internally (no base attention head config needed). head_dim=None marks that
            # this block consumes no RoPE.
            args.sdm_args.dim = args.dim
            self.attention = SparseDeltaMemory(args.sdm_args, layer_id)
            self.head_dim = None
            self.n_heads = None
            self.n_kv_heads = None
        else:
            if (args.head_dim is None) and (args.n_heads is None):
                raise ValueError(
                    f"{layer_type} layer {layer_id}: specify at least head_dim or n_heads"
                )
            if layer_type == "swa":
                # Sliding-window attention: own head configuration. Unset fields fall back
                # to the full-attention values, EXCEPT that a custom swa_n_heads defaults
                # swa_n_kv_heads to itself (MHA) rather than the base GQA n_kv_heads, which
                # need not divide swa_n_heads.
                n_heads = (
                    args.swa_n_heads if args.swa_n_heads is not None
                    else (args.n_heads or args.dim // args.head_dim)
                )
                if args.swa_n_kv_heads is not None:
                    n_kv_heads = args.swa_n_kv_heads
                elif args.swa_n_heads is not None:
                    n_kv_heads = n_heads
                else:
                    n_kv_heads = args.n_kv_heads or n_heads
                head_dim = (
                    args.swa_head_dim if args.swa_head_dim is not None
                    else args.dim // n_heads
                )
            else:  # "attn": full causal attention
                head_dim = args.head_dim if args.head_dim is not None else args.dim // args.n_heads
                n_heads = args.n_heads if args.n_heads is not None else args.dim // args.head_dim
                n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else n_heads

            if n_heads % n_kv_heads != 0:
                raise ValueError(
                    f"{layer_type} layer {layer_id}: n_heads={n_heads} not divisible by "
                    f"n_kv_heads={n_kv_heads}"
                )
            if head_dim % 2 != 0:
                raise ValueError(
                    f"{layer_type} layer {layer_id}: head_dim={head_dim} must be even for RoPE"
                )
            self.head_dim = head_dim
            self.n_heads = n_heads
            self.n_kv_heads = n_kv_heads
            self.attention = Attention(
                dim=args.dim,
                head_dim=head_dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                rope_theta=args.rope_theta,
                attn_gate=args.attn_gate,
            )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:

        if self.attn_type == "sdm":
            # SDM is causal by construction (chunk-wise recurrence); it takes no
            # rope/tok_idx/mask. During generation a decode cache (SDMLayerState) is
            # installed on the layer as `_gen_cache`; the recurrence resumes from it and
            # writes it back. In training/normal forward it is absent (None) and the
            # returned state is discarded.
            sdm_cache = getattr(self.attention, "_gen_cache", None)
            attn_out, new_cache = self.attention(
                self.attention_norm(x), freqs_cis=None, mask=None, cache=sdm_cache
            )
            if sdm_cache is not None:
                self.attention._gen_cache = new_cache
        else:
            # freq_cis is a {head_dim: tensor} mapping (per-type RoPE); accept a bare
            # tensor too for any legacy caller that passes one directly.
            fc = freq_cis[self.head_dim] if isinstance(freq_cis, dict) else freq_cis
            # mask may be a {layer_type: mask} mapping (full-causal vs sliding-window) or
            # a single mask applied to every attention layer. Index by type so a missing
            # per-type mask fails loudly rather than silently running unmasked.
            m = mask[self.attn_type] if isinstance(mask, dict) else mask
            attn_out = self.attention(
                self.attention_norm(x),
                fc,
                tok_idx=tok_idx,
                mask=m,
                attn_impl=attn_impl,
            )
        h = x + attn_out
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, init_std=None, factor=1.0):
        if self._is_sdm:
            self.attention.init_weights(init_std, factor)
        else:
            self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()

        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()


class BaseTransformer(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen

        # Resolve the explicit per-layer sequence-mixer assignment (attn / swa / sdm).
        self.layer_types = resolve_layer_types(
            args.n_layers, args.attn_at, args.swa_at, args.sdm_at
        )

        # RoPE is applied per attention layer using that layer's head_dim. Full-attention
        # layers use the default head_dim; SWA layers may use a different one. Keep the
        # default-head_dim module named `rope_embeddings` (some apps reach into it
        # directly) and hold any additional head_dim in `extra_rope`.
        if args.head_dim is not None:
            default_head_dim = args.head_dim
        elif args.n_heads is not None:
            default_head_dim = args.dim // args.n_heads
        else:
            # No base attention head config (e.g. an all-SDM model); RoPE is unused. Keep a
            # tiny placeholder table so `rope_embeddings` exists. Any attention layer would
            # already have raised in TransformerBlock for the missing head config.
            default_head_dim = 2
        self.default_head_dim = default_head_dim
        self.rope_embeddings = RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=default_head_dim,
            max_seqlen=args.max_seqlen,
        )
        self.extra_rope = nn.ModuleDict()
        if "swa" in self.layer_types:
            # Mirror TransformerBlock's SWA head_dim derivation so the key matches.
            swa_n_heads = (
                args.swa_n_heads if args.swa_n_heads is not None
                else (args.n_heads or (args.dim // args.head_dim if args.head_dim else None))
            )
            if swa_n_heads:
                swa_head_dim = (
                    args.swa_head_dim if args.swa_head_dim is not None
                    else args.dim // swa_n_heads
                )
                if swa_head_dim != default_head_dim:
                    self.extra_rope[str(swa_head_dim)] = RotaryEmbedding(
                        theta=args.rope_theta,
                        head_dim=swa_head_dim,
                        max_seqlen=args.max_seqlen,
                    )
        # Distinct head_dims for which RoPE tables exist (default + any SWA head_dim).
        self.rope_head_dims = [default_head_dim] + [
            int(k) for k in self.extra_rope.keys()
        ]

        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(
                TransformerBlock(args, layer_id, self.layer_types[layer_id])
            )

    def _rope_module(self, head_dim: int) -> "RotaryEmbedding":
        if head_dim == self.default_head_dim:
            return self.rope_embeddings
        return self.extra_rope[str(head_dim)]

    def forward(
        self,
        h,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str, dict]] = None,
        attn_impl: str = "sdpa",
    ):

        # Compute RoPE once per distinct attention head_dim, keyed for per-layer lookup.
        freq_cis = {
            hd: self._rope_module(hd)(seqlen=self.max_seqlen, tok_idx=tok_idx)
            for hd in self.rope_head_dims
        }

        for i, layer in enumerate(self.layers):
            h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
        return h

    def reset_parameters(self):
        # Either use fixed base std or sqrt model dim
        self.rope_embeddings.reset_parameters()
        for rope in self.extra_rope.values():
            rope.reset_parameters()

    def init_weights(self):
        self.reset_parameters()
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(self.init_base_std, factor)
