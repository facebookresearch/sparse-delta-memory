# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Opt-in statistics for SparseDeltaMemory (memory-access / activation / gradient stats).

Everything here is logging only, gated by ``SparseDeltaMemoryArgs.log_memory_access_stats``
and ``log_memory_norms``, and factored out of ``layer.py`` so that file holds the model
math. ``SparseDeltaMemory`` mixes in ``SDMStatsMixin`` and calls the ``_record_*`` hooks only
in eval mode outside CUDA-graph capture: they read tensor values on the host (``.item()``).
The buffers are registered on the layer (``self``) so they follow device/FSDP moves.
"""
import torch


class SDMStatsMixin:
    # ------------------------------------------------------------------ setup
    def _init_stats(self, args, num_heads: int) -> None:
        """Allocate the stats buffers/accumulators (called from the layer __init__)."""
        if args.log_memory_access_stats:
            stats_slots = num_heads * self.slots_per_head
            # Unique-index access counts for read and write (across all heads).
            for name in ("_read_idx_counts", "_write_idx_counts"):
                self.register_buffer(name, torch.zeros(stats_slots, dtype=torch.int64), persistent=False)
            # Activation statistics accumulators (for stability debugging).
            self._activation_stats: dict[str, list[float]] = {
                "read_weight_max": [],
                "write_strength_max": [],
                "write_strength_mean": [],
                "output_norm": [],
                "forget_gate_min": [],
                "forget_gate_mean": [],
                "forget_gate_max": [],
                "input_gate_min": [],
                "input_gate_mean": [],
                "input_gate_max": [],
            }
            if args.log_memory_norms:
                self._activation_stats.update({
                    "memory_norm": [],
                    "memory_max": [],
                    "updated_memory_norm": [],
                    "updated_memory_max": [],
                    "slot_norm_mean": [],
                    "slot_norm_max": [],
                    "updated_slot_norm_mean": [],
                    "updated_slot_norm_max": [],
                })
            # Full slot access histograms (accumulated, converted to entropy at log time).
            for name in ("_read_slot_access_weights", "_write_slot_access_weights"):
                self.register_buffer(name, torch.zeros(stats_slots, dtype=torch.float32), persistent=False)
            # Gradient statistics, collected via collect_grad_stats() after backward.
            self._grad_stats: dict[str, list[float]] = {
                "memory_grad_norm": [],
                "qk_proj_grad_norm": [],
                "output_proj_grad_norm": [],
            }
        else:
            self._read_idx_counts = None
            self._write_idx_counts = None
            self._read_slot_access_weights = None
            self._write_slot_access_weights = None
            self._activation_stats = None
            self._grad_stats = None

    # ------------------------------------------------------------ forward hooks
    def _record_gate_stats(self, g: torch.Tensor, beta: torch.Tensor) -> None:
        if self._activation_stats is None:
            return
        with torch.no_grad():
            decay = torch.exp(g)
            s = self._activation_stats
            s["forget_gate_min"].append(decay.min().item())
            s["forget_gate_mean"].append(decay.mean().item())
            s["forget_gate_max"].append(decay.max().item())
            s["input_gate_min"].append(beta.min().item())
            s["input_gate_mean"].append(beta.mean().item())
            s["input_gate_max"].append(beta.max().item())

    def _record_read_stats(self, q_idx, q_val, BH, H, sph) -> None:
        if self._read_idx_counts is None:
            return
        with torch.no_grad():
            # Indices are [BH, T, num_reads] in [0, sph). Add head offset for [0, H*sph).
            head_offset = (torch.arange(BH, device=q_idx.device) % H * sph).view(BH, 1, 1)
            q_idx_flat = (q_idx + head_offset).view(-1)
            self._read_idx_counts.index_add_(
                0, q_idx_flat.long(), torch.ones_like(q_idx_flat, dtype=torch.int64)
            )
            if self._activation_stats is not None:
                self._activation_stats["read_weight_max"].append(q_val.max().item())
            if self._read_slot_access_weights is not None:
                # index_put_ with accumulate=True sums duplicates in sorted order on CUDA
                # (deterministic); a float index_add_ uses atomics.
                q_val_flat = q_val.reshape(-1)
                self._read_slot_access_weights.index_put_(
                    (q_idx_flat.long(),), q_val_flat.to(self._read_slot_access_weights.dtype),
                    accumulate=True)

    def _record_memory_norms_pre(self, memory: torch.Tensor) -> None:
        if not (self.args.log_memory_norms and self._activation_stats is not None):
            return
        with torch.no_grad():
            s = self._activation_stats
            s["memory_norm"].append(memory.norm().item())
            s["memory_max"].append(memory.abs().max().item())
            slot_norms = memory.norm(dim=-1)
            s["slot_norm_mean"].append(slot_norms.mean().item())
            s["slot_norm_max"].append(slot_norms.max().item())

    def _record_write_stats(self, k_idx, k_val, memory, output, BH, H, sph) -> None:
        if self._write_idx_counts is None:
            return
        with torch.no_grad():
            # k_idx has batch offsets; mod by sph then add head offset for [0, H*sph).
            head_offset_w = (torch.arange(BH, device=k_idx.device) % H * sph).view(BH, 1, 1)
            write_idx_flat = (k_idx % sph + head_offset_w).view(-1)
            self._write_idx_counts.index_add_(
                0, write_idx_flat.long(), torch.ones_like(write_idx_flat, dtype=torch.int64)
            )
            if self._activation_stats is not None:
                s = self._activation_stats
                s["write_strength_max"].append(k_val.max().item())
                s["write_strength_mean"].append(k_val.mean().item())
                if self.args.log_memory_norms and memory is not None:
                    s["updated_memory_norm"].append(memory.norm().item())
                    s["updated_memory_max"].append(memory.abs().max().item())
                    updated_slot_norms = memory.norm(dim=-1)
                    s["updated_slot_norm_mean"].append(updated_slot_norms.mean().item())
                    s["updated_slot_norm_max"].append(updated_slot_norms.max().item())
                s["output_norm"].append(output.norm(dim=-1).mean().item())
            if self._write_slot_access_weights is not None:
                write_strength_flat = k_val.reshape(-1)
                self._write_slot_access_weights.index_put_(
                    (write_idx_flat.long(),),
                    write_strength_flat.to(self._write_slot_access_weights.dtype),
                    accumulate=True)

    # ------------------------------------------------------------ public API
    def collect_grad_stats(self) -> None:
        """Collect gradient statistics after backward pass.

        Call this after optimizer.step() to record gradient norms.
        Hooks don't work reliably with torch.compile, so we check .grad directly.
        """
        if self._grad_stats is None:
            return

        # Memory parameter gradient
        if self.memory is not None and self.memory.grad is not None:
            self._grad_stats["memory_grad_norm"].append(self.memory.grad.norm().item())

        # Q/K projection gradients
        qk_grad_norms = []
        for proj in [self.Wq_read, self.Wk_write]:
            if proj is not None:
                for param in proj.parameters():
                    if param.grad is not None:
                        qk_grad_norms.append(param.grad.norm().item())
                        break
        if qk_grad_norms:
            self._grad_stats["qk_proj_grad_norm"].append(sum(qk_grad_norms) / len(qk_grad_norms))

        # Output projection gradient
        if self.Wo is not None:
            for param in self.Wo.parameters():
                if param.grad is not None:
                    self._grad_stats["output_proj_grad_norm"].append(param.grad.norm().item())
                    break

    def get_memory_access_stats(self, reset: bool = True) -> "dict[str, float] | None":
        """Get memory access statistics and optionally reset counters.

        Returns a dict of unique-access percentages, read/write slot-access entropy, and the
        averaged activation + gradient statistics since the last reset. Returns None if
        ``log_memory_access_stats`` is disabled.
        """
        if not self.args.log_memory_access_stats:
            return None

        if self._read_idx_counts is None or self._write_idx_counts is None:
            return None

        num_slots = self._read_idx_counts.numel()

        # Compute stats - only count unique slots
        read_unique = (self._read_idx_counts > 0).sum().item()
        write_unique = (self._write_idx_counts > 0).sum().item()

        stats = {
            "read_unique_pct": 100.0 * read_unique / num_slots if num_slots > 0 else 0.0,
            "write_unique_pct": 100.0 * write_unique / num_slots if num_slots > 0 else 0.0,
        }

        # Compute entropy from full slot access histograms
        eps = 1e-10
        for kind in ("read", "write"):
            weights = getattr(self, f"_{kind}_slot_access_weights")
            if weights is not None:
                total_weight = weights.sum()
                if total_weight > eps:
                    probs = weights / total_weight
                    max_entropy = torch.log(torch.tensor(num_slots, dtype=torch.float32))
                    entropy = -(probs * probs.clamp(min=eps).log()).sum()
                    stats[f"{kind}_slot_entropy_normalized"] = (entropy / max_entropy).item()
                else:
                    stats[f"{kind}_slot_entropy_normalized"] = 0.0

        # Add activation statistics (averaged over forward passes)
        # and gradient statistics (averaged over backward passes)
        for acc in (self._activation_stats, self._grad_stats):
            for key, values in (acc or {}).items():
                stats[key] = sum(values) / len(values) if values else 0.0

        if reset:
            for buf in (self._read_idx_counts, self._write_idx_counts, self._read_slot_access_weights,
                        self._write_slot_access_weights):
                if buf is not None:
                    buf.zero_()
            for acc in (self._activation_stats, self._grad_stats):
                for key in acc or ():
                    acc[key] = []

        return stats
