"""Selective mixed-precision for TensorNet2's all-pairs Coulomb (the memory-bound O(N^2) term).

Nsight Compute showed the Coulomb kernels are DRAM-bandwidth-bound (91% DRAM, 1.8% SM), dominated
by the per-edge [E, C] intermediates (E ~ N^2/2, C = (num_layers+1)*q_dim = 48). Halving their bytes
(fp32 -> bf16) should give a ~proportional speedup on a bandwidth-bound kernel.

Design (the NEW contribution — torchmd-net only has *uniform* dtype, and Warp has no bf16 kernels):
  * cast the bandwidth-heavy per-edge tensors (gathered charges, q_i*q_j, damping, e_ij) to `edge_dtype`
  * keep POSITIONS and the distance d_ij in fp32  -> geometric precision & exact rotational equivariance
  * accumulate (index_add) into per-atom energy in fp32 -> stable reduction / energy conservation

This is a faithful port of ScalarPlusWeightedCoulomb.pre_reduce's `all_to_all` branch
(output_modules.py L468-531 + tail L606-609). Set edge_dtype=torch.float32 to verify the port
reproduces the stock fp32 result; set torch.bfloat16 for the mixed-precision variant.
"""
from __future__ import annotations

import torch


def patch_coulomb(edge_dtype: torch.dtype = torch.bfloat16):
    """Monkeypatch ScalarPlusWeightedCoulomb.pre_reduce (all_to_all path) to use `edge_dtype`
    for the per-edge intermediates. Returns an unpatch() callable. Idempotent."""
    from torchmdnet.models.output_modules import (
        ScalarPlusWeightedCoulomb as C,
        _triu_indices,
        _exp_cutoff,
    )

    orig = getattr(C, "_orig_pre_reduce", C.pre_reduce)

    def pre_reduce_mixed(self, x, v, z, pos, batch, box=None):
        # Only the all_to_all (O(N^2)) path is bandwidth-bound; defer everything else to stock code.
        if getattr(self, "mode", None) != "all_to_all":
            return orig(self, x, v, z, pos, batch, box)

        # Cast charges to edge_dtype ONCE, BEFORE the gather, so the (memory-bound) gather of the
        # [E,C] per-edge charge tensors moves half the bytes — this is where the bandwidth win is.
        # (Casting after the gather, as a first attempt did, leaves the gather at fp32 bytes and only
        #  adds cast kernels — ncu showed no speedup. Hardware-guided fix: cast up front.)
        charges = x[:, self.hidden_channels:].to(edge_dtype)
        x = x[:, : self.hidden_channels]
        x = self.output_network(x)                       # per-atom scalar energy [N,1], fp32

        # edge_index warmup semantics (mirror stock): (re)build unless a CUDA graph is capturing
        is_capturing = x.is_cuda and torch.cuda.is_current_stream_capturing()
        if (not x.is_cuda) or (not is_capturing) or (self.edge_index.shape[1] == 0):
            self.edge_index = _triu_indices(x, batch)
        ei = self.edge_index

        # distances in fp32 (geometry precision -> equivariance); charges/products in edge_dtype
        d_ij = torch.linalg.norm(pos[ei[0]] - pos[ei[1]], dim=-1)          # fp32 [E]
        q_i = charges[ei[0]]                                               # bf16 gather [E,C]
        q_j = charges[ei[1]]
        q_ij = q_i * q_j                                                   # edge_dtype [E,C]
        fc = (1.0 - _exp_cutoff(d_ij, 4.6)).to(edge_dtype)                 # [E]
        e_ij = fc.unsqueeze(-1) * q_ij / d_ij.to(edge_dtype).unsqueeze(-1) # edge_dtype [E,C]
        e_ij = self._factor * e_ij
        w = self.qweights.to(edge_dtype)
        # channel-weighted mean -> back to fp32 before the reduction/accumulation
        e_ij = (torch.sum(e_ij * w.unsqueeze(0), dim=-1).to(torch.float32)
                / self.qweights.to(torch.float32).sum())                  # fp32 [E]

        e_i = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
        e_i = e_i.index_add(0, ei[0], e_ij)                               # fp32 accumulation
        e_i = e_i.index_add(0, ei[1], e_ij)
        return x + e_i.unsqueeze(-1)

    C._orig_pre_reduce = orig
    C.pre_reduce = pre_reduce_mixed
    C._mixed_edge_dtype = edge_dtype

    def unpatch():
        C.pre_reduce = orig
        if hasattr(C, "_orig_pre_reduce"):
            del C._orig_pre_reduce
        if hasattr(C, "_mixed_edge_dtype"):
            del C._mixed_edge_dtype

    return unpatch
