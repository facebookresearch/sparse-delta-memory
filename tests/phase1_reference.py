# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Plain-torch references of Phase 1's group maps and sorted rows (moved out of the package: tests only)."""
import torch

from sparse_delta_memory.cuda.phase1_cuda import (POS_COUNT_MAX, SPAY_DIST_MAX, SPAY_DIST_SHIFT,  # noqa: F401
                                                  SPAY_ENTRY_MASK, pos_pack)


def group_maps_reference(payload: torch.Tensor, C: int, W: int, R: int):
    """(pos_k, pos_q, q_last, q_only) of ``Phase1`` from the sorted payload rows [N, row].

    pos_k / pos_q: the packed position and group role of every K / Q entry (see pos_pack;
    invalid entries included, with no peers).
    q_last: the entry is a valid Q entry and no later valid Q entry shares its group (slot);
    q_only: also the group's only valid Q entry. Invalid entries get flags 0.

    Plain torch ops on integers (deterministic, no host synchronization); the reference of
    the kernel outputs.
    """
    N = payload.shape[0]
    n = C * (W + R)
    dev = payload.device
    p = payload[:, :n].to(torch.int64) & 0xFFFFFFFF
    lane = p & 0xFF
    isq = (p >> 8) & 1
    tok = (p >> 9) & 0xFF
    invalid = (p >> 17) & 1
    head = (p >> 31) & 1
    r = torch.arange(n, device=dev, dtype=torch.int64).expand(N, n)
    # Every entry (t, lane) of a chunk occurs once in its row: K entries map to [0, C*W),
    # Q entries to [C*W, C*(W+R)).
    dest = torch.where(isq == 1, C * W + tok * R + lane, tok * W + lane)
    gid = head.cumsum(1) - 1
    zeros = torch.zeros(N, 1, device=dev, dtype=torch.int64)
    peers = {}
    for kind, mine in (("k", (1 - isq) * (1 - invalid)), ("q", isq * (1 - invalid))):
        cum = mine.cumsum(1)                                   # inclusive count of peers
        start = torch.where(head == 1, r, torch.zeros_like(r)).cummax(1).values
        before = cum - mine - torch.cat([zeros, cum], 1).gather(1, start)
        total = torch.zeros(N, n, device=dev, dtype=torch.int64).scatter_add_(1, gid, mine)
        total = total.gather(1, gid)
        peers[kind] = (mine, before, total)
    # A K entry of an invalid token writes nothing, and nor does an invalid Q entry: both
    # report no peers, so a kernel sees them as singletons of their own.
    packed = torch.zeros(N, n, device=dev, dtype=torch.int64)
    for kind in ("k", "q"):
        mine, before, total = peers[kind]
        after = (total - before - 1).clamp(min=0) * mine
        packed = torch.where((isq == (kind == "q")) & (mine == 1),
                             pos_pack(r, after, total), packed)
    packed = torch.where(invalid == 1, r, packed)
    pos = torch.empty(N, n, device=dev, dtype=torch.int64).scatter_(1, dest, packed)
    mine, before, total = peers["q"]
    q_last = mine * ((total - before - 1) == 0).to(torch.int64)
    q_only = q_last * (total == 1).to(torch.int64)
    flags = torch.zeros(N, 2, n, device=dev, dtype=torch.int8)
    flags[:, 0].scatter_(1, dest, q_last.to(torch.int8))
    flags[:, 1].scatter_(1, dest, q_only.to(torch.int8))
    cw = C * W
    return (pos[:, :cw].to(torch.int32).view(N, C, W),
            pos[:, cw:].to(torch.int32).view(N, C, R),
            flags[:, 0, cw:].contiguous().view(N, C, R),
            flags[:, 1, cw:].contiguous().view(N, C, R))


def sorted_kind_rows_reference(payload: torch.Tensor, C: int, W: int, R: int):
    """(spay_k, spay_q) of ``Phase1`` from the sorted payload rows [N, row]; plain torch ops
    on integers (deterministic, static shapes), the reference of the kernel outputs."""
    N = payload.shape[0]
    n = C * (W + R)
    p = payload[:, :n].to(torch.int64) & 0xFFFFFFFF
    isq = (p >> 8) & 1
    gid = ((p >> 31) & 1).cumsum(1)          # one id per group of the row (heads start one)
    rows = []
    for q, width in ((0, W), (1, R)):
        m = C * width
        # This kind's entries in row order (a stable sort on "other kind" keeps the order).
        idx = torch.argsort((isq != q).to(torch.int8), dim=1, stable=True)[:, :m]
        e = p.gather(1, idx)
        g = gid.gather(1, idx)
        first = torch.ones(N, m, dtype=torch.int64, device=p.device)
        first[:, 1:] = (g[:, 1:] != g[:, :-1]).to(torch.int64)
        last = torch.ones_like(first)
        last[:, :-1] = first[:, 1:]
        j = torch.arange(m, device=p.device, dtype=torch.int64).expand(N, m)
        dist = j - torch.where(first == 1, j, torch.zeros_like(j)).cummax(1).values
        elem = ((e & SPAY_ENTRY_MASK) | dist.clamp(max=SPAY_DIST_MAX) << SPAY_DIST_SHIFT |
                (first << 31) | (last << 30))
        rows.append(torch.where(elem >= 1 << 31, elem - (1 << 32), elem).to(torch.int32))
    return rows[0], rows[1]


def sorted_rank_reference(payload: torch.Tensor, C: int, W: int, R: int):
    """``Phase1.spay_rank`` from the sorted payload rows [N, row]: for each block b of 32
    positions of a chunk's sorted row, (write entries at positions < 32 b, bitmask of the
    positions 32 b .. 32 b + 31 that hold a write entry). Padding tokens' write entries count;
    positions past C * (W + R) never do. An entry at position r has index
    j = kbefore + popcount(kmask & ((1 << (r % 32)) - 1)) in spay_k if it writes, and r - j
    in spay_q if it reads."""
    N = payload.shape[0]
    n = C * (W + R)
    nb = (n + 31) // 32
    p = payload[:, :n].to(torch.int64) & 0xFFFFFFFF
    blocks = torch.zeros(N, nb * 32, dtype=torch.int64, device=p.device)
    blocks[:, :n] = ((p >> 8) & 1) ^ 1
    blocks = blocks.view(N, nb, 32)
    kmask = (blocks << torch.arange(32, device=p.device, dtype=torch.int64)).sum(-1)
    count = blocks.sum(-1)
    kbefore = count.cumsum(1) - count
    kmask = torch.where(kmask >= 1 << 31, kmask - (1 << 32), kmask)
    return torch.stack([kbefore, kmask], -1).to(torch.int32)
