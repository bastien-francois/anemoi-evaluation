"""Derived variables a source can deliver from the variables it has: `10si = sqrt(10u^2 + 10v^2)`.

anemoi-transform filters (`uv_to_ddff`) cannot be used for this: anemoi-inference applies its post-processors in
`Runner.execute()`, not in the `Runner.run()` generator the evaluation reads, and `open_dataset` applies no filters.

Opt-in: a source keeps its `variables` as they are (so the default variable set of a run does not change) and lists
what it can derive from them in `derived_variables`; a derived name is scored only when the config asks for it in
`variables`. The source reads the components in place of it (`base_variables`) and computes it (`derive`) on numpy
arrays or torch tensors, on the device of the data.
"""

from __future__ import annotations

import numpy as np
import torch

DERIVED: dict[str, tuple[str, ...]] = {"10si": ("10u", "10v")}


def derivable(variables: list[str]) -> list[str]:
    """The derived names whose components are all in `variables` and that `variables` does not already hold."""
    return [name for name, parts in DERIVED.items() if name not in variables and all(p in variables for p in parts)]


def base_variables(requested: list[str], available: list[str]) -> list[str]:
    """`requested` with each derived name the source does not hold itself replaced by its components; no
    duplicates, order kept."""
    out: list[str] = []
    for name in requested:
        parts = DERIVED[name] if name in DERIVED and name not in available else (name,)
        for part in parts:
            if part not in out:
                out.append(part)
    return out


def derive(data: np.ndarray | torch.Tensor, base: list[str], requested: list[str]) -> np.ndarray | torch.Tensor:
    """`data` of shape (..., len(base), N) as (..., len(requested), N)."""
    index = {name: i for i, name in enumerate(base)}
    columns = []
    for name in requested:
        if name in index:
            columns.append(data[..., index[name], :])
        elif name == "10si":
            u, v = data[..., index["10u"], :], data[..., index["10v"], :]
            columns.append((u * u + v * v) ** 0.5)
        else:
            raise ValueError(f"cannot derive {name!r} from {base}")
    stack = torch.stack if isinstance(data, torch.Tensor) else np.stack
    return stack(columns, -2)


def derived_variable_info(name: str) -> dict | None:
    """`param` and `level` of a derived variable, None for any other name."""
    return {"param": name, "level": None} if name in DERIVED else None
