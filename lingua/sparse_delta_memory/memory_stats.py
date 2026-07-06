# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Opt-in statistics for SparseDeltaMemory (memory-access / activation / gradient stats).

Everything here is logging only — gated by ``SparseDeltaMemoryArgs.log_memory_access_stats``
and ``log_memory_norms`` — and is factored out of ``layer.py`` so that file holds the model
math. ``SparseDeltaMemory`` mixes in ``SDMStatsMixin``; each ``_record_*`` hook is a no-op
unless the corresponding flag is on, so the default (stats-off) forward path is unaffected.
The buffers are still registered on the layer (``self``) so they follow device/FSDP moves.
"""
import torch


class SDMStatsMixin:
    # ------------------------------------------------------------------ setup
    def _init_stats(self, args, num_heads: int) -> None:
        """Allocate the stats buffers/accumulators (called from the layer __init__)."""
        if args.log_memory_access_stats:
            stats_slots = num_heads * self.slots_per_head
            # Unique-index access counts for read and write (across all heads).
            self.register_buffer(
                "_read_idx_counts",
                torch.zeros(stats_slots, dtype=torch.int64),
                persistent=False,
            )
            self.register_buffer(
                "_write_idx_counts",
                torch.zeros(stats_slots, dtype=torch.int64),
                persistent=False,
            )
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
            self.register_buffer(
                "_read_slot_access_weights",
                torch.zeros(stats_slots, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "_write_slot_access_weights",
                torch.zeros(stats_slots, dtype=torch.float32),
                persistent=False,
            )
            # Gradient statistics — collected via collect_grad_stats() after backward.
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
        if not (self.training and self.args.log_memory_access_stats
                and self._activation_stats is not None):
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

    def _record_read_stats(self, q_idx, q_val, BH, H, sph, ulysses_active) -> None:
        # Skip under Ulysses CP — indices are local-heads only.
        if not (self.args.log_memory_access_stats and self._read_idx_counts is not None
                and not ulysses_active):
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
                q_val_flat = q_val.view(-1)
                self._read_slot_access_weights.index_add_(
                    0, q_idx_flat.long(), q_val_flat.to(self._read_slot_access_weights.dtype)
                )

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

    def _record_write_stats(self, k_idx, k_val, memory, output, BH, H, sph, ulysses_active) -> None:
        # Skip under Ulysses CP — k_idx is local-heads only.
        if not (self.args.log_memory_access_stats and self._write_idx_counts is not None
                and not ulysses_active):
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
                write_strength_flat = k_val.view(-1)
                self._write_slot_access_weights.index_add_(
                    0, write_idx_flat.long(),
                    write_strength_flat.to(self._write_slot_access_weights.dtype),
                )

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

        num_slots = self.num_heads * self.slots_per_head

        # Compute stats - only count unique slots
        read_unique = (self._read_idx_counts > 0).sum().item()
        write_unique = (self._write_idx_counts > 0).sum().item()

        stats = {
            "read_unique_pct": 100.0 * read_unique / num_slots if num_slots > 0 else 0.0,
            "write_unique_pct": 100.0 * write_unique / num_slots if num_slots > 0 else 0.0,
        }

        # Compute entropy from full slot access histograms
        eps = 1e-10
        if self._read_slot_access_weights is not None:
            read_weights = self._read_slot_access_weights
            read_total_weight = read_weights.sum()
            if read_total_weight > eps:
                read_probs = read_weights / read_total_weight
                read_probs_safe = read_probs.clamp(min=eps)
                max_entropy = torch.log(torch.tensor(num_slots, dtype=torch.float32))
                read_entropy = -(read_probs * read_probs_safe.log()).sum()
                stats["read_slot_entropy_normalized"] = (read_entropy / max_entropy).item()
            else:
                stats["read_slot_entropy_normalized"] = 0.0

        if self._write_slot_access_weights is not None:
            write_weights = self._write_slot_access_weights
            write_total_weight = write_weights.sum()
            if write_total_weight > eps:
                write_probs = write_weights / write_total_weight
                write_probs_safe = write_probs.clamp(min=eps)
                max_entropy = torch.log(torch.tensor(num_slots, dtype=torch.float32))
                write_entropy = -(write_probs * write_probs_safe.log()).sum()
                stats["write_slot_entropy_normalized"] = (write_entropy / max_entropy).item()
            else:
                stats["write_slot_entropy_normalized"] = 0.0

        # Add activation statistics (averaged over forward passes)
        if self._activation_stats is not None:
            for key, values in self._activation_stats.items():
                stats[key] = sum(values) / len(values) if values else 0.0

        # Add gradient statistics (averaged over backward passes)
        if self._grad_stats is not None:
            for key, values in self._grad_stats.items():
                stats[key] = sum(values) / len(values) if values else 0.0

        if reset:
            self._read_idx_counts.zero_()
            self._write_idx_counts.zero_()
            if self._read_slot_access_weights is not None:
                self._read_slot_access_weights.zero_()
            if self._write_slot_access_weights is not None:
                self._write_slot_access_weights.zero_()
            if self._activation_stats is not None:
                for key in self._activation_stats:
                    self._activation_stats[key] = []
            if self._grad_stats is not None:
                for key in self._grad_stats:
                    self._grad_stats[key] = []

        return stats
