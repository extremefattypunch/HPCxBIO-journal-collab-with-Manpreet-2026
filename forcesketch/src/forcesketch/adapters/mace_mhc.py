"""MACE multi-head-committee adapter (spec SS18).

The ONLY module in the project permitted to import mace/e3nn/ase. Everything
downstream sees `energies(batch) -> [B, M]` and nothing else, which is what makes
a multi-head TensorNet2 a drop-in replacement.

Note what this adapter deliberately does NOT use: MACE's own
`get_outputs_committee`, which loops M sequential reverse passes. That loop is
the spec SS26 item-1 baseline (and is exposed here as `member_forces_reference`
for exactly that purpose), but ForceSketch computes its own VJPs from head-space
cotangents, which is the entire point.
"""

from __future__ import annotations

from pathlib import Path

import e3nn
import torch
from torch import Tensor

from forcesketch.adapters.base import BaseMHCAdapter


def configure_e3nn_for_batched_vjp() -> None:
    """Disable e3nn's TorchScript codegen. MUST run before any model is built or
    loaded, which is why it is called at import time of this module.

    e3nn 0.4.4 defaults to jit_script_fx=True, compiling every tensor product with
    torch.jit.script. A scripted backward cannot accept vmap's BatchedTensor, so
    `torch.autograd.grad(..., is_grads_batched=True)` dies with

        RuntimeError: The following operation failed in the TorchScript interpreter.
        RuntimeError: Cannot access data pointer of Tensor that doesn't have storage

    That is not a vmap *fallback* (spec SS18's warning case) -- it is a hard error,
    and it would remove the batched exact baseline that spec SS17 requires and that
    the whole SS50 systems question is measured against. Setting the flag before the
    checkpoint is unpickled fixes it: e3nn's CodeGenMixin.__setstate__ regenerates
    the forward at load time and honours the current default.

    Verified on the real 3BPA committee: batched and serial VJPs both return
    [L, N_total, 3] and agree to float64 tolerance.

    (torch.func.vjp + vmap remains unavailable for a separate reason -- MACE calls
    Tensor.requires_grad_() inside forward, which functorch forbids -- so spec
    SS17 item 4 is reported as not applicable rather than silently skipped.)
    """
    e3nn.set_optimization_defaults(jit_script_fx=False)


configure_e3nn_for_batched_vjp()


class MaceMHCAdapter(BaseMHCAdapter):
    def __init__(self, model, *, device="cuda", dtype: torch.dtype = torch.float32):
        self.model = model.to(device=device, dtype=dtype).eval()
        self._device = torch.device(device)
        self._dtype = dtype
        self._heads = list(model.heads)
        self._committee = torch.tensor(
            [i for i, h in enumerate(self._heads) if "committee-" in h],
            dtype=torch.long, device=self._device,
        )
        if len(self._committee) == 0:  # a committee-less checkpoint: use all heads
            self._committee = torch.arange(len(self._heads), device=self._device)

    # --- construction -----------------------------------------------------
    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device="cuda",
                        dtype: torch.dtype = torch.float32,
                        expect_sha256: str | None = None) -> "MaceMHCAdapter":
        """Load a pickled ScaleShiftMACE.

        `weights_only=False` is unavoidable: these checkpoints are pickled
        nn.Modules carrying embedded e3nn TorchScript codegen, not plain tensors.
        Mitigate by verifying the published digest BEFORE unpickling -- pass
        `expect_sha256`. We only ever load artifacts from Zenodo record 17829635.
        """
        path = Path(path)
        if expect_sha256 is not None:
            from forcesketch.utils.fetch import sha256_file

            got = sha256_file(path)
            if got != expect_sha256:
                raise RuntimeError(f"{path.name}: sha256 {got} != expected {expect_sha256}")
        model = torch.load(path, map_location="cpu", weights_only=False)  # noqa: S614
        if not hasattr(model, "heads"):
            raise TypeError(f"{path} is not a multi-head MACE model")
        return cls(model, device=device, dtype=dtype)

    # --- MHCAdapter surface ------------------------------------------------
    @property
    def num_heads(self) -> int:
        return len(self._committee)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def prepare(self, batch: dict) -> dict:
        batch = {
            k: (v.to(self._device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        for k, v in batch.items():
            if torch.is_tensor(v) and torch.is_floating_point(v):
                batch[k] = v.to(self._dtype)
        batch["positions"] = batch["positions"].detach().requires_grad_(True)
        return batch

    def num_structures(self, batch: dict) -> int:
        return int(batch["batch"].max().item()) + 1

    def batch_index(self, batch: dict) -> Tensor:
        return batch["batch"]

    def positions(self, batch: dict) -> Tensor:
        return batch["positions"]

    def energies(self, batch: dict, *, create_graph: bool = False) -> Tensor:
        """[B, M] committee head energies."""
        out = self.model(batch, training=False, compute_force=False,
                         committee_heads=self._committee)
        heads = out.get("heads")
        if heads is None or heads.get("energy") is None:
            raise RuntimeError(
                "model returned heads=None. The pinned fork hardcodes loss='dpose' in "
                "ScaleShiftMACE.forward, which makes the committee branch unreachable. "
                "Apply environment/mace-fork.patch."
            )
        return heads["energy"][:, self._committee]

    # --- spec SS26 item-1 baseline, as shipped by the reference paper -------
    def member_forces_reference(self, batch: dict) -> Tensor:
        """M sequential reverse passes -- MACE's own `get_outputs_committee` loop,
        which is exactly spec SS26's 'explicit per-head force loop'. Returns
        [M, N_total, 3]."""
        E = self.energies(batch)
        pos = self.positions(batch)
        return torch.stack(
            [-torch.autograd.grad(E[:, m].sum(), pos, retain_graph=True)[0]
             for m in range(self.num_heads)]
        )
