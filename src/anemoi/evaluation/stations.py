"""Scoring against point observations (stations).

- `ObstableTargets`: observations from a harp obstable sqlite (a table with `SID`, `lat`, `lon`, `elev`, `validdate` in
  Unix seconds and one column per variable), one node per station position (a station that moved is one node per
  position), NaN where a station did not report; optional `qc` rules (`valid_range`) on the observations only.
- `bilinear_cell_weights`: for each station, the cell of a structured grid (`shape` = rows, columns, row-major) that
  contains it, found by inverting the bilinear map of the stored latitudes/longitudes (Newton, the method of the
  mxalign `bilinear_cell` interpolation), and the bilinear weights of its 4 corners; stations in no cell are left out.
- `StationForecast`: a forecast source seen at the stations: every frame (members, variables) is interpolated with
  those weights, after the derived variables (10si) were computed on the grid; optional height correction
  `x + lapse_rate * (z_grid@station - station elevation)` (e.g. 2t with 0.0065 K/m and the model orography), NaN for
  stations without elevation (the case then leaves the scores).
"""

from __future__ import annotations

import datetime
import sqlite3
from typing import Any

import numpy as np
import torch

from anemoi.evaluation.frame import Frame
from anemoi.evaluation.frame import Grid
from anemoi.evaluation.sources.base import MissingTargetError
from anemoi.evaluation.sources.base import TargetSourceBase
from anemoi.evaluation.sources.base import select_variables
from anemoi.evaluation.target_qc import TargetQC

EPOCH = datetime.datetime(1970, 1, 1)
META_COLUMNS = ("SID", "lat", "lon", "elev", "validdate")
STATION_QC_RULES = ("valid_range",)


def _xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def _inv_bilinear(p00, p10, p01, p11, q, n_iter: int = 20):
    """Solve p(s,t) = q for the bilinear map of a cell (vectorised), as mxalign's `bilinear_cell`."""
    s = np.full(len(q), 0.5)
    t = np.full(len(q), 0.5)
    for _ in range(n_iter):
        f = ((1 - s) * (1 - t))[:, None] * p00 + (s * (1 - t))[:, None] * p10 + ((1 - s) * t)[:, None] * p01 \
            + (s * t)[:, None] * p11 - q
        ds_ = (1 - t)[:, None] * (p10 - p00) + t[:, None] * (p11 - p01)
        dt_ = (1 - s)[:, None] * (p01 - p00) + s[:, None] * (p11 - p10)
        det = ds_[:, 0] * dt_[:, 1] - ds_[:, 1] * dt_[:, 0]
        det = np.where(det == 0, np.nan, det)
        s = s - (f[:, 0] * dt_[:, 1] - f[:, 1] * dt_[:, 0]) / det
        t = t - (ds_[:, 0] * f[:, 1] - ds_[:, 1] * f[:, 0]) / det
    return s, t


def bilinear_cell_weights(grid_lat, grid_lon, st_lat, st_lon, shape, eps: float = 1e-9):
    """(corners (K, 4) node indices, weights (K, 4) float64, inside (S,) bool) for the K of S stations inside a cell of
    the structured grid; corners are [k, k+1, k+nx, k+nx+1] of the cell's lower-left node k."""
    from scipy.spatial import cKDTree

    ny, nx = (int(n) for n in shape)
    grid_lat, grid_lon = np.asarray(grid_lat, dtype=np.float64), np.asarray(grid_lon, dtype=np.float64)
    if grid_lat.size != ny * nx:
        raise ValueError(f"bilinear_cell: the grid has {grid_lat.size} nodes, shape {ny}x{nx} needs {ny * nx}")
    glon = np.where(grid_lon > 180, grid_lon - 360, grid_lon).reshape(ny, nx)
    glat = grid_lat.reshape(ny, nx)
    st_lat = np.asarray(st_lat, dtype=np.float64)
    slon = np.asarray(st_lon, dtype=np.float64)
    slon = np.where(slon > 180, slon - 360, slon)
    q = np.column_stack([slon, st_lat])
    # first guess: the nearest node; the containing cell has it as a corner (then a wider ring, for safety)
    _, nearest = cKDTree(_xyz(glat.ravel(), glon.ravel())).query(_xyz(st_lat, slon))
    jn, in_ = np.divmod(np.asarray(nearest, dtype=np.int64), nx)
    offsets = [(0, 0), (0, -1), (-1, 0), (-1, -1)] + [
        (dj, di) for dj in (-2, -1, 0, 1) for di in (-2, -1, 0, 1) if (dj, di) not in ((0, 0), (0, -1), (-1, 0), (-1, -1))
    ]
    n = len(q)
    best = np.full(n, -1, dtype=np.int64)
    s_out, t_out = np.full(n, np.nan), np.full(n, np.nan)
    for dj, di in offsets:
        todo = best < 0
        if not todo.any():
            break
        j0, i0 = jn[todo] + dj, in_[todo] + di
        ok = (i0 >= 0) & (i0 <= nx - 2) & (j0 >= 0) & (j0 <= ny - 2)
        idx = np.where(todo)[0][ok]
        j0, i0 = j0[ok], i0[ok]
        if len(idx) == 0:
            continue

        def corner(jj, ii):
            return np.column_stack([glon[jj, ii], glat[jj, ii]])

        s, t = _inv_bilinear(corner(j0, i0), corner(j0, i0 + 1), corner(j0 + 1, i0), corner(j0 + 1, i0 + 1), q[idx])
        hit = (s >= -eps) & (s <= 1 + eps) & (t >= -eps) & (t <= 1 + eps)
        sel = idx[hit]
        best[sel] = j0[hit] * nx + i0[hit]
        s_out[sel] = np.clip(s[hit], 0, 1)
        t_out[sel] = np.clip(t[hit], 0, 1)
    inside = best >= 0
    k, s, t = best[inside], s_out[inside], t_out[inside]
    corners = np.column_stack([k, k + 1, k + nx, k + nx + 1])
    weights = np.column_stack([(1 - s) * (1 - t), s * (1 - t), (1 - s) * t, s * t])
    return corners, weights, inside


class ObstableTargets(TargetSourceBase):
    """Station observations from a harp obstable sqlite, one node per station position."""

    def __init__(
        self, path: str, table: str = "SYNOP", variables: list[str] | None = None, qc: list | None = None
    ) -> None:
        self.path, self.table = str(path), table
        uri = f"file:{self.path}?mode=ro"
        with sqlite3.connect(uri, uri=True) as con:
            columns = [row[1] for row in con.execute(f'PRAGMA table_info("{table}")')]
            missing = [c for c in META_COLUMNS if c not in columns]
            if missing:
                raise ValueError(f"obstable {self.path}: table {table} has no columns {missing}")
            available = [c for c in columns if c not in META_COLUMNS]
            self.variables = list(variables) if variables is not None else available
            select_variables(self.variables, available)
            quoted = ", ".join(f'"{c}"' for c in (*META_COLUMNS, *self.variables))
            rows = con.execute(f'SELECT {quoted} FROM "{table}"').fetchall()
        if not rows:
            raise ValueError(f"obstable {self.path}: table {table} is empty")
        sid = np.array([r[0] for r in rows], dtype=np.int64)
        lat = np.array([r[1] for r in rows], dtype=np.float64)
        lon = np.array([r[2] for r in rows], dtype=np.float64)
        elev = np.array([np.nan if r[3] is None else r[3] for r in rows], dtype=np.float64)
        seconds = np.array([r[4] for r in rows], dtype=np.int64)
        values = np.array([[np.nan if v is None else v for v in r[5:]] for r in rows], dtype=np.float64)
        keys = np.rec.fromarrays([sid, lat, lon, np.nan_to_num(elev, nan=-1e30)])  # a station position = one node
        unique, node = np.unique(keys, return_inverse=True)
        times, slot = np.unique(seconds, return_inverse=True)
        fields = np.full((len(times), len(self.variables), len(unique)), np.nan, dtype=np.float32)
        fields[slot, :, node] = values.astype(np.float32)
        self.station_ids = unique.f0.astype(np.int64)
        self.elevations = np.where(unique.f3 == -1e30, np.nan, unique.f3)
        self.grid = Grid(unique.f1, unique.f2)
        self.dates = [EPOCH + datetime.timedelta(seconds=int(s)) for s in times]
        self._index = {date: i for i, date in enumerate(self.dates)}
        self._fields = fields
        self.derived_variables: list[str] = []
        self.dropped_stations: list[int] = []
        if qc:
            names = [next(iter(rule)) for rule in qc]
            unsupported = [name for name in names if name not in STATION_QC_RULES]
            if unsupported:
                raise ValueError(f"obstable qc: {unsupported} not supported for stations, use {list(STATION_QC_RULES)}")
        self.qc_specs = list(qc or [])
        self.qc = TargetQC(self.qc_specs, self) if qc else None

    def select_stations(self, keep: np.ndarray) -> None:
        """Keep the stations of the boolean mask `keep` (the ones inside the forecast grid), before any read."""
        keep = np.asarray(keep, dtype=bool)
        if keep.shape != (self.grid.n,):
            raise ValueError(f"select_stations: mask of {keep.shape} for {self.grid.n} stations")
        self.dropped_stations = sorted(set(self.station_ids[~keep].tolist()))
        self.station_ids, self.elevations = self.station_ids[keep], self.elevations[keep]
        self.grid = Grid(self.grid.latitudes[keep], self.grid.longitudes[keep])
        self._fields = self._fields[:, :, keep]

    def available(self, valid_time: datetime.datetime) -> bool:
        """Whether the file has observations at `valid_time`."""
        return valid_time in self._index

    def frame(self, valid_time: datetime.datetime, variables: list[str], device: torch.device) -> torch.Tensor:
        """Observations at `valid_time` as (1, V, N), NaN where a station did not report or the qc removed it."""
        index = self._index.get(valid_time)
        if index is None:
            raise MissingTargetError(f"no observations at {valid_time} in {self.path}")
        rows = select_variables(list(variables), self.variables)
        data = np.array(self._fields[index][rows], dtype=np.float32, order="C")
        if self.qc is not None:
            data = self.qc.apply(valid_time, list(variables), data)
        return torch.from_numpy(data).to(device)[None]

    def describe(self) -> dict:
        """File, table, stations, dates and qc."""
        return {
            "source": "obstable",
            "path": self.path,
            "table": self.table,
            "stations": self.grid.n,
            "dropped_stations": len(self.dropped_stations),
            "dates": {"first": self.dates[0].isoformat(), "last": self.dates[-1].isoformat(), "count": len(self.dates)},
            "variables": len(self.variables),
            **({"qc": self.qc_specs} if self.qc_specs else {}),
        }

    def to_config(self) -> dict:
        """The `targets:` block that rebuilds this source."""
        block = {"path": self.path, "table": self.table, "variables": list(self.variables)}
        if self.qc_specs:
            block["qc"] = self.qc_specs
        return {"obstable": block}


class StationForecast:
    """A forecast source seen at the stations of `targets` (an `ObstableTargets`), whose kept stations it selects."""

    has_graph = False

    def __init__(
        self,
        source: Any,
        targets: ObstableTargets,
        shape: tuple[int, int],
        first_node: int = 0,
        height_correction: dict | None = None,
    ) -> None:
        self.source = source
        ny, nx = (int(n) for n in shape)
        end = first_node + ny * nx
        if first_node < 0 or end > source.grid.n:
            raise ValueError(f"bilinear_cell: nodes {first_node}..{end} outside the forecast grid of {source.grid.n}")
        sub = slice(first_node, end)
        corners, weights, inside = bilinear_cell_weights(
            source.grid.latitudes[sub], source.grid.longitudes[sub], targets.grid.latitudes, targets.grid.longitudes,
            (ny, nx),
        )
        targets.select_stations(inside)
        self.shape, self.first_node = (ny, nx), first_node
        self.stations_in, self.stations_out = int(inside.sum()), int((~inside).sum())
        self.grid = targets.grid
        self._corners = torch.as_tensor(corners + first_node)
        self._weights = torch.as_tensor(weights, dtype=torch.float64)
        self.height_correction = dict(height_correction or {})
        self._offsets: dict[str, torch.Tensor] = {}
        for name, spec in self.height_correction.items():
            orography = self._orography(spec["orography"], ny * nx)
            at_station = (weights * orography[corners]).sum(axis=1)
            offset = float(spec["lapse_rate"]) * (at_station - targets.elevations)  # NaN without elevation
            self._offsets[name] = torch.as_tensor(offset, dtype=torch.float64)

    @staticmethod
    def _orography(spec: dict, size: int) -> np.ndarray:
        """The orography field (m) on the structured grid: `variable` of `dataset` at its first date."""
        from anemoi.datasets import open_dataset

        dataset = spec["dataset"]
        ds = open_dataset(**dataset) if isinstance(dataset, dict) else open_dataset(dataset)
        column = list(ds.variables).index(spec["variable"])
        field = np.asarray(ds[0][column, 0], dtype=np.float64)
        if field.shape != (size,):
            raise ValueError(f"height correction: orography has {field.shape[0]} nodes, the grid {size}")
        return field

    def __getattr__(self, name: str) -> Any:  # everything else is the wrapped source's
        if name == "source":
            raise AttributeError(name)
        return getattr(self.source, name)

    def _map(self, frame: Frame) -> Frame:
        data = frame.data
        corners, weights = self._corners.to(data.device), self._weights.to(data.device)
        out = (data[..., corners].double() * weights).sum(-1)  # (M, V, stations)
        for name, offset in self._offsets.items():
            if name in frame.variables:
                out[:, frame.variables.index(name)] += offset.to(data.device)
        return Frame(frame.init_time, frame.lead_time, frame.valid_time, list(frame.variables), out.float())

    def frames(self, init_time, lead_time, variables, device):
        """The wrapped source's frames, at the stations."""
        for frame in self.source.frames(init_time, lead_time, variables, device):
            yield self._map(frame)

    def initial_frame(self, init_time, variables, device):
        """The wrapped source's lead-0 frame, at the stations."""
        return self._map(self.source.initial_frame(init_time, variables, device))

    def graph_node_attribute(self, name: str, nodes: str = "data") -> np.ndarray:
        """Stations have no graph nodes."""
        raise ValueError("graph node attributes are not defined at stations; use uniform weights")

    def describe(self) -> dict:
        """The wrapped source's description, with the stations kept and left out."""
        return {
            **self.source.describe(),
            "nodes": self.grid.n,
            "interpolation": {"bilinear_cell": {"shape": list(self.shape), "first_node": self.first_node}},
            "stations": {"inside": self.stations_in, "outside": self.stations_out},
            **({"height_correction": self.height_correction} if self.height_correction else {}),
        }

    def station_config(self) -> dict:
        """The `interpolation` and `height_correction` blocks of the run config."""
        block = {"interpolation": {"bilinear_cell": {"shape": list(self.shape), "first_node": self.first_node}}}
        if self.height_correction:
            block["height_correction"] = self.height_correction
        return block
