#!/usr/bin/env python3
"""Precision study for TensorNet2's forces: fp32 and selective mixed_bf16 vs an fp64 reference.

Same weights across precisions (deepcopy + cast), so this isolates NUMERICAL precision error
(not model accuracy vs DFT — that needs a trained checkpoint, orthogonal to this study).

Reports, per precision and system size:
  * relative force error   ||F_p - F_fp64|| / ||F_fp64||   (and max abs component error)
  * relative energy error
  * rotational-equivariance error  ||F(R x) - R F(x)|| / ||F(x)||   (R in SO(3))
Plus a PORT-CORRECTNESS check: mixed with edge_dtype=fp32 must match stock fp32 (~1e-6),
validating the mixed-precision reimplementation of the all-pairs Coulomb.

Run: conda run -n tn2prof python experiments/precision_accuracy.py
"""
from __future__ import annotations

import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(os.path.dirname(HERE), "harness")
sys.path.insert(0, HARNESS)

import torch  # noqa: E402
from model_build import build_model  # noqa: E402
from workloads import gen_system  # noqa: E402
from mixed_precision import patch_coulomb  # noqa: E402

torch.manual_seed(0)


def forces(model, z, pos):
    """model(z,pos,batch) -> (E, F); pass a fresh leaf each call (model sets requires_grad)."""
    batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
    E, F = model(z, pos.detach().clone(), batch=batch)
    return E.detach().double(), F.detach().double()


def rand_rotation(device):
    a = torch.randn(3, 3, dtype=torch.float64, device=device)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))          # proper orthonormal
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q  # (3,3), det +1


def rel(a, b):
    return (torch.linalg.norm((a - b).reshape(-1)) / torch.linalg.norm(b.reshape(-1))).item()


def build_ref_and_variants(n, device="cuda"):
    """fp64 reference model + inputs; return (model_fp64, model_fp32, z, pos64)."""
    m64, info = build_model(coulomb_cutoff=None, device=device, dtype=torch.float64)
    m64.eval()
    z, pos64, _ = gen_system(n, seed=1, device=device, dtype=torch.float64)
    m32 = copy.deepcopy(m64).to(torch.float32)
    m32.eval()
    return m64, m32, z, pos64, info


def run_size(n, device="cuda"):
    m64, m32, z, pos64, info = build_ref_and_variants(n, device)
    pos32 = pos64.to(torch.float32)

    # fp64 reference
    E64, F64 = forces(m64, z, pos64)

    # stock fp32 (ensure unpatched)
    E32, F32 = forces(m32, z, pos32)

    # PORT CHECK: mixed with edge_dtype=fp32 must reproduce stock fp32
    unpatch = patch_coulomb(torch.float32)
    Ep, Fp = forces(m32, z, pos32)
    unpatch()
    port_err = rel(Fp, F32)

    # selective mixed_bf16
    unpatch = patch_coulomb(torch.bfloat16)
    Emx, Fmx = forces(m32, z, pos32)
    unpatch()

    rows = []
    for name, E, F in [("fp32", E32, F32), ("mixed_bf16", Emx, Fmx)]:
        rows.append((name,
                     rel(F, F64),
                     (F - F64).abs().max().item(),
                     rel(E, E64)))
    return info, port_err, rows


def run_equivariance(n=500, device="cuda"):
    m64, m32, z, pos32_src, info = build_ref_and_variants(n, device)
    pos = pos32_src.to(torch.float32)
    R = rand_rotation(device)
    Rf = R.to(torch.float32)
    out = {}
    # fp32
    _, F = forces(m32, z, pos)
    _, F_rot = forces(m32, z, pos @ Rf.T)
    out["fp32"] = rel(F_rot, (F.to(torch.float32) @ Rf.T).double())
    # mixed_bf16
    unpatch = patch_coulomb(torch.bfloat16)
    _, Fm = forces(m32, z, pos)
    _, Fm_rot = forces(m32, z, pos @ Rf.T)
    unpatch()
    out["mixed_bf16"] = rel(Fm_rot, (Fm.to(torch.float32) @ Rf.T).double())
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print("PRECISION STUDY — forces vs fp64 reference (same weights)")
    print("=" * 78)
    for n in (250, 500, 1000):
        try:
            info, port_err, rows = run_size(n, dev)
        except RuntimeError as e:  # noqa: BLE001
            print(f"N={n}: SKIPPED ({e})")
            continue
        print(f"\nN={n}  warp_opt={info['warp_opt_active']}  "
              f"port_check(mixed@fp32 vs stock fp32 rel force err)={port_err:.2e}")
        print(f"  {'precision':>12} {'relForceErr':>12} {'maxAbsFErr':>12} {'relEnergyErr':>13}")
        for name, rf, mx, re_ in rows:
            print(f"  {name:>12} {rf:>12.3e} {mx:>12.3e} {re_:>13.3e}")

    print("\n" + "=" * 78)
    print("ROTATIONAL EQUIVARIANCE — ||F(Rx) - R F(x)|| / ||F(x)|| (N=500)")
    print("=" * 78)
    eq = run_equivariance(500, dev)
    for k, v in eq.items():
        print(f"  {k:>12}: {v:.3e}")
    print("\n(mixed_bf16 keeps positions/distances fp32 -> equivariance should match fp32)")


if __name__ == "__main__":
    main()
