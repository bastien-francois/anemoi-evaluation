"""Several truths scored from one rollout (`truths:` in the config).

The model runs once per init time for the union of the truths' variables; every frame is then split per truth (its
own variables, and for a station truth the interpolation to its stations) and scored by that truth's `Evaluation`
inside a `MultiEvaluation(kind="truth")`, which writes one result file per truth. Lead 0, which needs no model call, is
produced per truth by its own evaluation (a truth may leave it out, e.g. an accumulation scored against radar).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from anemoi.evaluation.frame import Frame
from anemoi.evaluation.stations import StationForecast


class SharedRollout:
    """`multi_frames` of one forecast source for several truths: `views[name]` is the source itself or a
    `StationForecast` around it."""

    def __init__(self, source: Any, views: dict[str, Any]) -> None:
        self.source = source
        self.views = dict(views)

    def multi_frames(self, init_time, lead_time, variables: dict[str, list[str]], device) -> Iterator[dict[str, Frame]]:
        """One rollout for the union of `variables`, each frame split into one frame per truth."""
        union: list[str] = []
        for names in variables.values():
            union += [name for name in names if name not in union]
        columns = {truth: [union.index(name) for name in names] for truth, names in variables.items()}
        for frame in self.source.frames(init_time, lead_time, union, device):
            split = {}
            for truth, names in variables.items():
                part = Frame(
                    frame.init_time, frame.lead_time, frame.valid_time, list(names), frame.data[:, columns[truth]]
                )
                view = self.views[truth]
                split[truth] = view._map(part) if isinstance(view, StationForecast) else part
            yield split

    def __getattr__(self, name: str) -> Any:
        if name in ("source", "views"):
            raise AttributeError(name)
        return getattr(self.source, name)

    def close(self) -> None:
        """Release the forecast source."""
        self.source.close()
