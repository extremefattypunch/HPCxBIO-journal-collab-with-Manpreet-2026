"""Build a TensorNet2 energy+forces model that EXERCISES the new v2 path
(charge prediction + neutral-charge equilibration + explicit Coulomb).

Verified against torchmd-net @ main:
  - torchmdnet/models/tensornet2.py : TensorNet2 (representation); OPT flag = Warp path toggle;
    output_charges=True appends per-atom charges to x for the output head.
  - torchmdnet/models/output_modules.py : ScalarPlusWeightedCoulomb
    (coulomb_cutoff=None -> mode 'all_to_all' = the O(N^2) term we profile;
     a float -> cutoff + reaction-field). Reads charges = x[:, hidden:].
  - torchmdnet/models/model.py : TorchMD_Net(rep, out, derivative=True) -> forward(z,pos,batch)
    returns (energy, forces=-dE/dpos). create_graph = self.training, so eval() = 1st-order forces.
"""
from __future__ import annotations

import torch


def _set_opt(force_eager: bool) -> bool:
    """Optionally force the pure-PyTorch (non-Warp) backbone BEFORE tensornet2 is imported.
    Returns the effective OPT flag. Must run before `import ...tensornet2`."""
    import torchmdnet.models.tensornet as _tn  # triggers the Warp try/except detection

    if force_eager:
        _tn.OPT = False
    return bool(_tn.OPT)


def build_model(
    hidden_channels: int = 128,
    q_dim: int = 16,
    num_layers: int = 2,
    num_rbf: int = 32,
    cutoff_upper: float = 4.5,
    coulomb_cutoff: float | None = None,  # None => all-to-all O(N^2) Coulomb
    max_num_neighbors: int = 64,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    force_eager: bool = False,
    static_shapes: bool | None = None,
    use_coulomb: bool = True,   # False => plain Scalar head (ablation: isolates the Coulomb cost)
):
    """Return (model, info). model(z, pos, batch=batch) -> (energy, forces)."""
    opt = _set_opt(force_eager)
    from torchmdnet.models.tensornet2 import TensorNet2
    from torchmdnet.models.output_modules import ScalarPlusWeightedCoulomb, Scalar
    from torchmdnet.models.model import TorchMD_Net

    # static_shapes=True matches AceFF/production + the Warp opt path; fall back to False on error.
    candidates = [static_shapes] if static_shapes is not None else [True, False]
    last_err = None
    for ss in candidates:
        try:
            rep = TensorNet2(
                hidden_channels=hidden_channels,
                q_dim=q_dim,
                num_layers=num_layers,
                num_rbf=num_rbf,
                cutoff_lower=0.0,
                cutoff_upper=cutoff_upper,
                max_num_neighbors=max_num_neighbors,
                max_z=128,
                equivariance_invariance_group="O(3)",
                static_shapes=ss,
                output_charges=use_coulomb,   # append charges to x only for the Coulomb head
                dtype=dtype,
            )
            if use_coulomb:
                out = ScalarPlusWeightedCoulomb(
                    hidden_channels,
                    activation="silu",
                    reduce_op="sum",
                    dtype=dtype,
                    static_shapes=ss,
                    num_hidden_layers=0,
                    num_layers=num_layers,                 # asserts len(q_weights)==num_layers+1
                    q_dim=q_dim,
                    q_weights=[1.0] * (num_layers + 1),     # equal weight per charge stage
                    coulomb_cutoff=coulomb_cutoff,          # None => O(N^2) all-to-all
                    coulomb_max_num_neighbors=None,
                    coulomb_neighbor_strategy="brute",
                )
                mode = ("all_to_all(O(N^2))" if coulomb_cutoff is None
                        else f"cutoff({coulomb_cutoff})")
            else:
                # ablation: charge prediction/equilibration still runs in the representation,
                # but the O(N^2) Coulomb energy term is removed.
                out = Scalar(hidden_channels, activation="silu", reduce_op="sum", dtype=dtype)
                mode = "none(scalar-only)"
            model = TorchMD_Net(rep, out, derivative=True, dtype=dtype).to(device)
            info = {
                "warp_opt_active": opt,
                "static_shapes": ss,
                "coulomb_mode": mode,
                "q_channels_total": (num_layers + 1) * q_dim,
            }
            return model, info
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"Failed to build TensorNet2+Coulomb model: {last_err}")


def smoke_test():
    """Build tiny model, run one forward+forces, assert finite. Prints the OPT/config."""
    from workloads import gen_system  # local import so this file is import-safe

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, info = build_model(device=dev)
    model.eval()
    z, pos, batch = gen_system(n_atoms=30, device=dev)
    energy, forces = model(z, pos, batch=batch)
    ok = torch.isfinite(energy).all().item() and torch.isfinite(forces).all().item()
    print(f"[smoke] info={info}")
    print(f"[smoke] energy.shape={tuple(energy.shape)} forces.shape={tuple(forces.shape)} "
          f"finite={ok}")
    assert ok, "non-finite energy/forces"
    return info


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    smoke_test()
