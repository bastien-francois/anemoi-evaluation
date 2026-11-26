"""Quality control of the targets: rules that set target values to NaN before scoring.

They act on the targets only, so whether a case is dropped never depends on a forecast; a NaN target then leaves the
case by the usual rule (a case counts only where the target and every member are finite). The data stay as they are,
and every rule is defined in the configuration or by a dataset, so a run needs no side file. Rules, in the `qc:` list of
the `anemoi_dataset` targets block, each for the `variables` it names (`grid`: a sub-grid of a cutout or join, default 0):

- `only_grid: {variables, grid}`: a safeguard; every node outside sub-grid `grid` is set to NaN, so those variables are
  never scored on the other sub-grids (e.g. the IFS points of a cutout), whatever the `regions`.
- `max_distance: {variables, grid, max_km, sites: {name: [lon, lat]}}`: nodes of sub-grid `grid` farther than `max_km`
  (great circle, Earth radius 6371 km) from every site are set to NaN; other sub-grids are kept.
- `drop_where: {variables, grid, dataset, variable, above}`: `dataset` (anything `open_dataset` takes, with the nodes of
  sub-grid `grid` and the valid times of the targets) holds a QC field `variable`; where it is above `above` (strict,
  compared in the field's own dtype) the targets are set to NaN at that valid time. A valid time the QC dataset does not
  have is an error; NaN in the QC field drops nothing.
- `drop_dates: {variables, grid, start, end}`: every node of sub-grid `grid` is set to NaN for valid times in [start, end].
- `valid_range: {variables, min, max}`: values below `min` or above `max` (either may be left out) are set to NaN, e.g.
  the plausibility checks of station observations.

`counts()` gives, per rule and variable, the finite target values each rule removed over the distinct valid times read;
`attr()` is the `target_qc` result attribute (rules, totals and counts per valid time), `merge_attrs()` merges the
attributes of runs (shards) whose valid times may overlap: each valid time is counted once.
"""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any

import numpy as np

EARTH_RADIUS_KM = 6371.0


def _as_datetime(value: Any) -> datetime.datetime:
    return np.datetime64(value, "us").astype(datetime.datetime)


def _sub_grid(targets: Any, grid: int) -> tuple[int, int]:
    """(start, size) of sub-grid `grid` in the targets' dataset."""
    grids = [int(count) for count in getattr(targets.ds, "grids", [targets.ds.shape[-1]])]
    if not 0 <= grid < len(grids):
        raise ValueError(f"qc: grid {grid} out of range for {len(grids)} grids")
    return sum(grids[:grid]), grids[grid]


def _on_target_nodes(values: np.ndarray, start: int, total: int, fill: bool, targets: Any) -> np.ndarray:
    """A boolean array over the nodes of one sub-grid placed on the target's nodes (`fill` elsewhere)."""
    full = np.full(total, fill, dtype=bool)
    full[start : start + len(values)] = values
    return full[targets.grid_indices]


def distance_to_nearest_km(latitudes: np.ndarray, longitudes: np.ndarray, sites: dict) -> np.ndarray:
    """Great-circle distance (km) from each point to the nearest of `sites` ({name: [lon, lat]})."""
    lat = np.radians(np.asarray(latitudes, dtype=np.float64))
    lon = np.radians(np.asarray(longitudes, dtype=np.float64))
    nearest = np.full(lat.shape, np.inf)
    for site_lon, site_lat in sites.values():
        lat0, lon0 = np.radians(float(site_lat)), np.radians(float(site_lon))
        a = np.sin((lat0 - lat) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon0 - lon) / 2) ** 2
        nearest = np.minimum(nearest, 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))
    return nearest


class _Rule:
    def __init__(self, options: dict, targets: Any) -> None:
        self.variables = list(options["variables"])

    def drop(self, valid_time: datetime.datetime, values: np.ndarray) -> np.ndarray | None:
        """(N,) boolean, True where the target `values` (one variable at `valid_time`) are set to NaN; None when
        nothing is dropped."""
        raise NotImplementedError


class MaxDistance(_Rule):
    def __init__(self, options: dict, targets: Any) -> None:
        super().__init__(options, targets)
        sites = dict(options["sites"])
        if not sites or any(len(site) != 2 for site in sites.values()):
            raise ValueError("qc max_distance: `sites` maps each name to [lon, lat]")
        start, size = _sub_grid(targets, int(options.get("grid", 0)))
        lat = np.asarray(targets.ds.latitudes)[start : start + size]
        lon = np.asarray(targets.ds.longitudes)[start : start + size]
        far = distance_to_nearest_km(lat, lon, sites) > float(options["max_km"])
        self._drop = _on_target_nodes(far, start, int(targets.ds.shape[-1]), False, targets)

    def drop(self, valid_time: datetime.datetime, values: np.ndarray) -> np.ndarray:
        return self._drop


class DropWhere(_Rule):
    def __init__(self, options: dict, targets: Any) -> None:
        from anemoi.datasets import open_dataset

        super().__init__(options, targets)
        spec = options["dataset"]
        self.qc = open_dataset(**spec) if isinstance(spec, dict) else open_dataset(spec)
        self.start, size = _sub_grid(targets, int(options.get("grid", 0)))
        if self.qc.shape[-1] != size:
            raise ValueError(f"qc drop_where: the QC dataset has {self.qc.shape[-1]} nodes, sub-grid has {size}")
        variable = options["variable"]
        if variable not in self.qc.variables:
            raise ValueError(f"qc drop_where: {variable} is not in the QC dataset {list(self.qc.variables)}")
        self.column = list(self.qc.variables).index(variable)
        self.above = options["above"]
        self.index = {date: i for i, date in enumerate(self.qc.dates.astype("datetime64[us]").tolist())}
        self.total, self.targets = int(targets.ds.shape[-1]), targets
        self._last: tuple | None = None

    def drop(self, valid_time: datetime.datetime, values: np.ndarray) -> np.ndarray | None:
        if self._last is not None and self._last[0] == valid_time:  # several variables at one valid time: read once
            return self._last[1]
        i = self.index.get(valid_time)
        if i is None:
            raise ValueError(f"qc drop_where: the QC dataset has no valid time {valid_time}")
        field = np.asarray(self.qc[i][self.column, 0])
        bad = field > field.dtype.type(self.above)  # NaN compares False: drops nothing
        drop = _on_target_nodes(bad, self.start, self.total, False, self.targets) if bad.any() else None
        self._last = (valid_time, drop)
        return drop


class DropDates(_Rule):
    def __init__(self, options: dict, targets: Any) -> None:
        super().__init__(options, targets)
        self.start, self.end = _as_datetime(options["start"]), _as_datetime(options["end"])
        if self.end < self.start:
            raise ValueError(f"qc drop_dates: end {self.end} before start {self.start}")
        start, size = _sub_grid(targets, int(options.get("grid", 0)))
        self._all = _on_target_nodes(np.ones(size, dtype=bool), start, int(targets.ds.shape[-1]), False, targets)

    def drop(self, valid_time: datetime.datetime, values: np.ndarray) -> np.ndarray | None:
        return self._all if self.start <= valid_time <= self.end else None


class OnlyGrid(_Rule):
    """Safeguard: every node outside sub-grid `grid` is NaN, so no region (not even `all`) can score those variables
    against the other sub-grids (e.g. the IFS points of a cutout)."""

    def __init__(self, options: dict, targets: Any) -> None:
        super().__init__(options, targets)
        start, size = _sub_grid(targets, int(options.get("grid", 0)))
        self._drop = _on_target_nodes(np.zeros(size, dtype=bool), start, int(targets.ds.shape[-1]), True, targets)

    def drop(self, valid_time: datetime.datetime, values: np.ndarray) -> np.ndarray | None:
        return self._drop if self._drop.any() else None


class ValidRange(_Rule):
    def __init__(self, options: dict, targets: Any) -> None:
        super().__init__(options, targets)
        self.low, self.high = options.get("min"), options.get("max")
        if self.low is None and self.high is None:
            raise ValueError("qc valid_range: give `min`, `max` or both")

    def drop(self, valid_time: datetime.datetime, values: np.ndarray) -> np.ndarray | None:
        outside = np.zeros(values.shape, dtype=bool)
        if self.low is not None:
            outside |= values < self.low
        if self.high is not None:
            outside |= values > self.high
        return outside if outside.any() else None  # NaN compares False: left as it is


RULES = {"only_grid": OnlyGrid, "max_distance": MaxDistance, "drop_where": DropWhere, "drop_dates": DropDates, "valid_range": ValidRange}


class TargetQC:
    """The `qc` rules of a target source, applied in order to each decoded target row."""

    def __init__(self, specs: list[dict], targets: Any) -> None:
        self.specs = [dict(spec) for spec in specs]
        self.rules = []
        for spec in self.specs:
            if len(spec) != 1 or next(iter(spec)) not in RULES:
                raise ValueError(f"qc: each rule is one of {sorted(RULES)} with its options, got {spec}")
            name, options = next(iter(spec.items()))
            missing = [v for v in options["variables"] if v not in targets.variables + targets.derived_variables]
            if missing:
                raise ValueError(f"qc {name}: variables {missing} are not in the targets")
            self.rules.append(RULES[name](options, targets))
        self._counts: dict[tuple, int] = {}

    def apply(self, valid_time: datetime.datetime, variables: list[str], data: np.ndarray) -> np.ndarray:
        """`data` (V, N) for `variables` with the dropped values set to NaN; counts the finite values removed."""
        for k, rule in enumerate(self.rules):
            rows = [i for i, name in enumerate(variables) if name in rule.variables]
            if not rows:
                continue
            for i in rows:
                drop = rule.drop(valid_time, data[i])
                removed = 0 if drop is None else int((drop & np.isfinite(data[i])).sum())
                self._counts[(k, variables[i], valid_time)] = removed
                if drop is not None:
                    data[i, drop] = np.nan
        return data

    def counts(self) -> list[dict]:
        """Per rule and variable: finite target values removed and valid times seen, over distinct valid times."""
        out = []
        for k, spec in enumerate(self.specs):
            for name in self.rules[k].variables:
                seen = [n for (r, v, _), n in self._counts.items() if r == k and v == name]
                out.append({"rule": next(iter(spec)), "variable": name, "removed": sum(seen), "valid_times": len(seen)})
        return out

    def by_valid_time(self) -> list[dict]:
        """Per rule and variable: finite target values removed at each valid time read (ISO time -> count)."""
        out = []
        for k, spec in enumerate(self.specs):
            for name in self.rules[k].variables:
                seen = {t: n for (r, v, t), n in self._counts.items() if r == k and v == name}
                out.append({"rule": next(iter(spec)), "variable": name,
                            "removed": {t.isoformat(): n for t, n in sorted(seen.items())}})
        return out

    def attr(self) -> str:
        """The `target_qc` result attribute: JSON of the rules, the totals and the counts per valid time."""
        return json.dumps({"rules": self.specs, "removed": self.counts(), "by_valid_time": self.by_valid_time()}, default=str)


def merge_attrs(values: list[str]) -> str:
    """Merge `target_qc` attributes of runs with the same rules: the counts per valid time are united (a valid time
    read by several runs counts once and must have the same count in each) and the totals recomputed."""
    parsed = [json.loads(value) for value in values]
    if any("by_valid_time" not in p for p in parsed):  # results written before the counts per valid time
        logging.getLogger(__name__).warning("target_qc: inputs without counts per valid time, the first input's kept")
        return values[0]
    first = parsed[0]
    if any(json.dumps(p["rules"], sort_keys=True) != json.dumps(first["rules"], sort_keys=True) for p in parsed[1:]):
        raise ValueError("target_qc: the runs to merge have different qc rules")
    merged = []
    for i, entry in enumerate(first["by_valid_time"]):
        union: dict[str, int] = {}
        for p in parsed:
            for time, n in p["by_valid_time"][i]["removed"].items():
                if union.setdefault(time, n) != n:
                    raise ValueError(f"target_qc {entry['rule']} {entry['variable']}: {time} removed {union[time]} vs {n}")
        merged.append({"rule": entry["rule"], "variable": entry["variable"], "removed": dict(sorted(union.items()))})
    totals = [{"rule": m["rule"], "variable": m["variable"], "removed": sum(m["removed"].values()),
               "valid_times": len(m["removed"])} for m in merged]
    return json.dumps({"rules": first["rules"], "removed": totals, "by_valid_time": merged}, default=str)
