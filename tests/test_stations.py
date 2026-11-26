import datetime
import sqlite3

import numpy as np
import pytest
import torch

from anemoi.evaluation.config import EvaluationConfig
from anemoi.evaluation.evaluate import build
from anemoi.evaluation.frame import Grid
from anemoi.evaluation.sources.base import MissingTargetError
from anemoi.evaluation.sources.fake import ArrayTargets
from anemoi.evaluation.sources.fake import FakeForecastSource
from anemoi.evaluation.stations import ObstableTargets
from anemoi.evaluation.stations import StationForecast
from anemoi.evaluation.stations import bilinear_cell_weights

HOUR = datetime.timedelta(hours=1)
T0 = datetime.datetime(2024, 1, 1)
CPU = torch.device("cpu")
NY, NX = 5, 6
LAT, LON = np.meshgrid(np.linspace(50.0, 54.0, NY), np.linspace(0.0, 5.0, NX), indexing="ij")  # 1 degree cells


def _unix(date):
    return int((date - datetime.datetime(1970, 1, 1)).total_seconds())


@pytest.fixture
def obstable(tmp_path):
    path = tmp_path / "obs.sqlite"
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE SYNOP (SID INT, lat REAL, lon REAL, elev REAL, validdate INT, "2t" REAL, "10si" REAL, msl REAL)')
    rows = []
    for k in range(3):
        date = _unix(T0 + 6 * k * HOUR)
        rows += [
            (1, 51.5, 2.25, 100.0, date, 280.0 + k, 5.0, 101000.0),  # inside
            (2, 52.0, 3.0, None, date, 281.0, 60.0, 95000.0),  # inside, no elevation, 10si and msl out of range
            (3, 60.0, 2.0, 10.0, date, 270.0, 3.0, 100000.0),  # outside the grid
        ]
    rows.append((1, 51.5, 2.5, 100.0, _unix(T0 + 12 * HOUR), 290.0, None, None))  # station 1 moved: a second node
    con.executemany("INSERT INTO SYNOP VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.commit()
    con.close()
    return str(path)


def test_weights_reproduce_a_linear_field():
    st_lat, st_lon = np.array([51.5, 50.0, 54.0, 49.0, 52.25]), np.array([2.25, 0.0, 5.0, 2.0, 4.9])
    corners, weights, inside = bilinear_cell_weights(LAT.ravel(), LON.ravel(), st_lat, st_lon, (NY, NX))
    assert inside.tolist() == [True, True, True, False, True]
    np.testing.assert_allclose(weights.sum(1), 1.0)
    field = 3.0 * LAT.ravel() - 2.0 * LON.ravel()  # bilinear interpolation is exact for a linear field
    np.testing.assert_allclose((weights * field[corners]).sum(1), 3.0 * st_lat[inside] - 2.0 * st_lon[inside], atol=1e-9)
    assert corners[0].tolist() == [1 * NX + 2, 1 * NX + 3, 2 * NX + 2, 2 * NX + 3]  # cell of (51.5 N, 2.25 E)
    np.testing.assert_allclose(weights[0], [0.375, 0.125, 0.375, 0.125])
    with pytest.raises(ValueError, match="needs"):
        bilinear_cell_weights(LAT.ravel(), LON.ravel(), st_lat, st_lon, (NY, NX + 1))


def test_obstable(obstable):
    targets = ObstableTargets(obstable, qc=[{"valid_range": {"variables": ["10si"], "max": 50.0}}])
    assert targets.variables == ["2t", "10si", "msl"] and targets.grid.n == 4  # 3 stations, one at 2 positions
    assert len(targets.dates) == 3 and targets.available(T0) and not targets.available(T0 + HOUR)
    frame = targets.frame(T0 + 12 * HOUR, ["2t", "10si"], CPU)[0].numpy()
    by_node = {(i, la, lo): k for k, (i, la, lo) in enumerate(zip(targets.station_ids, targets.grid.latitudes, targets.grid.longitudes))}
    moved, first, two = by_node[(1, 51.5, 2.5)], by_node[(1, 51.5, 2.25)], by_node[(2, 52.0, 3.0)]
    assert frame[0, moved] == 290.0 and np.isnan(frame[1, moved])  # NULL -> NaN
    assert frame[0, first] == 282.0 and np.isnan(frame[1, two])  # 10si 60 > 50 removed by the qc
    assert np.isnan(targets.elevations[two]) and targets.elevations[first] == 100.0
    with pytest.raises(MissingTargetError):
        targets.frame(T0 + HOUR, ["2t"], CPU)
    assert targets.to_config()["obstable"]["qc"] == [{"valid_range": {"variables": ["10si"], "max": 50.0}}]
    with pytest.raises(ValueError, match="not supported for stations"):
        ObstableTargets(obstable, qc=[{"drop_dates": {"variables": ["2t"], "start": "2024", "end": "2025"}}])


def _forecast(members=3):
    grid = Grid(LAT.ravel(), LON.ravel())
    field = np.stack([3.0 * LAT.ravel() - 2.0 * LON.ravel(), np.full(NY * NX, 4.0), np.full(NY * NX, 1.0e5)]).astype(np.float32)
    fields = ArrayTargets(grid, ["2t", "10si", "msl"], {T0 + 6 * k * HOUR: field for k in range(3)})
    return FakeForecastSource(fields, 6 * HOUR, members, drift=0.5, spread=0.2, device="cpu"), field


def test_station_forecast_and_height_correction(obstable, monkeypatch):
    class Orography:
        variables = ["orog"]

        def __getitem__(self, i):
            return np.full((1, 1, NY * NX), 200.0)

    monkeypatch.setattr("anemoi.datasets.open_dataset", lambda *a, **k: Orography())
    forecast, field = _forecast()
    targets = ObstableTargets(obstable)
    correction = {"2t": {"lapse_rate": 0.0065, "orography": {"dataset": "orog.zarr", "variable": "orog"}}}
    stations = StationForecast(forecast, targets, (NY, NX), height_correction=correction)
    assert stations.stations_out == 1 and targets.grid.n == 3 and stations.grid is targets.grid  # station 3 dropped
    assert targets.dropped_stations == [3] and stations.members == 3 and stations.supports_lead_zero
    frame = next(stations.frames(T0, 6 * HOUR, ["2t", "msl"], CPU))
    assert frame.data.shape == (3, 2, 3) and frame.data.dtype == torch.float32
    linear = 3.0 * targets.grid.latitudes - 2.0 * targets.grid.longitudes
    expected = linear + 0.5 + 0.0065 * (200.0 - targets.elevations)  # drift 0.5 at step 1, member 1 has no offset
    np.testing.assert_allclose(frame.data[1, 0].numpy(), expected, rtol=1e-6)  # NaN where no elevation
    assert np.isnan(frame.data[1, 0].numpy()).tolist() == np.isnan(targets.elevations).tolist()
    np.testing.assert_allclose(frame.data[1, 1].numpy(), 1.0e5 + 0.5)  # msl: no correction
    assert stations.station_config()["height_correction"] == correction


def test_run_from_config(obstable):
    forecast, _ = _forecast()
    config = {
        "forecast": {"persistence": {"timestep": "6h"}},
        "targets": {"obstable": {"path": obstable, "qc": [{"valid_range": {"variables": ["msl"], "min": 96000.0}}]}},
        "interpolation": {"bilinear_cell": {"shape": [NY, NX]}},
        "lead_time": "12h",
        "init_times": {"dates": [T0.isoformat()]},
        "variables": ["2t", "10si", "msl"],
        "metrics": ["rmse", "fair_crps", "spread"],
        "weights": {"uniform": {}},
        "regions": {"stations": "all"},
    }
    evaluation = build(config, forecast=forecast)
    result = evaluation.run().to_xarray(evaluation.metrics).sel(bin="all", region="stations")
    # msl: station 2 (95000 < 96000) removed by the qc, the moved position has no msl -> 1 case per lead
    assert result["weight_sum"].sel(variable="msl").values.tolist() == [1.0, 1.0]
    # 2t: stations 1 and 2 at +6 h; the moved position also reports at +12 h -> 2, 3
    assert result["weight_sum"].sel(variable="2t").values.tolist() == [2.0, 3.0]
    assert evaluation.to_config()["interpolation"] == {"bilinear_cell": {"shape": [NY, NX], "first_node": 0}}
    with pytest.raises(ValueError, match="go together"):
        EvaluationConfig.model_validate({**config, "interpolation": None})
