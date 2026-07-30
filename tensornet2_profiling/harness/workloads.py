"""VRAM-safe synthetic molecular systems for profiling on an 8 GB RTX 5070 Laptop.

We use jittered-lattice coordinates at realistic molecular density (~0.09 atoms/Å³) so that:
  * neighbor counts within the 4.5 Å cutoff are physical (~30-40, under max_num_neighbors=64),
  * there are no atom overlaps (which would produce NaN forces),
  * the all-to-all Coulomb term still scales O(N^2) with N (independent of the local cutoff).
Coordinates need not be a real trajectory: the kernels exercised (gather/scatter message passing,
charge equilibration, all-pairs Coulomb, autograd forces) are identical to a physical input.
"""
from __future__ import annotations

import torch

# Rough protein/organic composition (H heavy-favoured), atomic numbers.
_SPECIES = torch.tensor([1, 1, 1, 6, 6, 7, 8])  # ~3:2:1:1 H:C:N:O


def gen_system(n_atoms: int, density: float = 0.09, jitter: float = 0.4,
               seed: int = 0, device: str = "cuda", dtype: torch.dtype = torch.float32):
    """Return (z[int64 N], pos[dtype N,3], batch[int64 N]) — a single molecule (batch 0)."""
    g = torch.Generator().manual_seed(seed)
    spacing = (1.0 / density) ** (1.0 / 3.0)            # grid spacing for target density
    side = int(torch.ceil(torch.tensor(float(n_atoms)) ** (1.0 / 3.0)).item())
    # cubic grid of side^3 points, take first n_atoms, add Gaussian jitter (< spacing/3)
    coords = torch.stack(torch.meshgrid(
        torch.arange(side), torch.arange(side), torch.arange(side), indexing="ij"
    ), dim=-1).reshape(-1, 3).float()[:n_atoms]
    coords = coords * spacing + jitter * torch.randn(coords.shape, generator=g)
    pos = coords.to(device=device, dtype=dtype)
    idx = torch.randint(0, len(_SPECIES), (n_atoms,), generator=g)
    z = _SPECIES[idx].to(device=device, dtype=torch.long)
    batch = torch.zeros(n_atoms, dtype=torch.long, device=device)
    return z, pos, batch


# Named presets. Workload A = small-molecule latency regime; B = scaling regime (pass --n).
PRESETS = {
    "aspirin_like": 21,
    "peptide_like": 42,
}


def gen_named(name: str, **kw):
    return gen_system(PRESETS[name], **kw)
