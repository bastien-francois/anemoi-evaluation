import datetime
import sqlite3

import numpy as np
import pytest
import xarray as xr

from anemoi.evaluation.__main__ import format_output_path
from anemoi.evaluation.config import EvaluationConfig
from anemoi.evaluation.evaluate import MultiEvaluation
from anemoi.evaluation.evaluate import build
from anemoi.evaluation.frame import Grid
from anemoi.evaluation.sources.fake import ArrayTargets
from anemoi.evaluation.sources.fake import FakeForecastSource

HOUR = datetime.timedelta(hours=1)
T0 = datetime.datetime(2024, 1, 1)
NY, NX = 5, 6
LAT, LON = np.meshgrid(np.linspace(50.0, 54.0, NY), np.linspace(0.0, 5.0, NX), indexing="ij")
VARIABLES = ["2t", "10si", "msl", "tp"]
DATES = [T0 + 6 * k * HOUR for k in range(5)]


def _fields(offset=0.0):
    rng = np.random.default_rng(1)
    base = np.stack(
        [280 + rng.normal(0, 2, NY * NX), 5 + rng.normal(0, 1, NY * NX), 1e5 + rng.normal(0, 200, NY * NX),
         np.abs(rng.normal(0, 1e-3, NY * NX)) - 2e-4]  # tp with a few negative values
    ).astype(np.float32)
    return ArrayTargets(Grid(LAT.ravel(), LON.ravel()), VARIABLES, {d: base + offset for d in DATES})


class CountingForecast(FakeForecastSource):
    rollouts = 0

    def frames(self, *args, **kwargs):
        type(self).rollouts += 1
        yield from super().frames(*args, **kwargs)


@pytest.fixture
def sources(monkeypatch, tmp_path):
    truth_fields = {"an": _fields(0.3), "radar": _fields(-0.1)}
    monkeypatch.setattr(
        "anemoi.evaluation.evaluate.DatasetTargets", lambda dataset, **settings: truth_fields[dataset]
    )
    path = tmp_path / "obs.sqlite"
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE SYNOP (SID INT, lat REAL, lon REAL, elev REAL, validdate INT, "2t" REAL, "10si" REAL, msl REAL)')
    epoch = datetime.datetime(1970, 1, 1)
    for k, date in enumerate(DATES):
        t = int((date - epoch).total_seconds())
        con.execute("INSERT INTO SYNOP VALUES (1, 51.5, 2.25, 100.0, ?, ?, 5.0, 101000.0)", (t, 281.0 + k))
        con.execute("INSERT INTO SYNOP VALUES (2, 53.2, 4.1, 20.0, ?, ?, 7.0, 100500.0)", (t, 279.0 - k))
    con.commit()
    con.close()
    CountingForecast.rollouts = 0
    return str(path)


def _forecast():
    return CountingForecast(_fields(), 6 * HOUR, members=3, drift=0.5, spread=0.2, device="cpu")


TRUTHS = {
    "analysis": {"targets": {"anemoi_dataset": {"dataset": "an"}}, "variables": ["2t", "msl"], "include_lead_zero": True},
    "radar": {"targets": {"anemoi_dataset": {"dataset": "radar"}}, "variables": ["tp"], "forecast_minimum": {"tp": 0.0}},
    "synop": {"variables": ["2t", "10si", "msl"], "include_lead_zero": True, "regions": {"lam": "all"},
              "interpolation": {"bilinear_cell": {"shape": [NY, NX]}}},
}
RUN = {
    "forecast": {"persistence": {"timestep": "6h"}},
    "lead_time": "18h",
    "init_times": {"dates": [T0.isoformat(), (T0 + 6 * HOUR).isoformat()]},
    "metrics": ["rmse", "fair_crps", "spread"],
    "weights": {"uniform": {}},
    "regions": {"lam": "all"},
}


def _truths(obstable):
    truths = {name: dict(block) for name, block in TRUTHS.items()}
    truths["synop"]["targets"] = {"obstable": {"path": obstable}}
    return truths


def _alone(name, block):
    """The run config of one truth on its own, as before `truths`."""
    config = {**RUN, "targets": block["targets"], "variables": block["variables"]}
    for key in ("include_lead_zero", "forecast_minimum", "interpolation", "regions"):
        if key in block:
            config[key] = block[key]
    return config


def test_truths_equal_separate_runs_with_one_rollout(sources):
    evaluation = build({**RUN, "truths": _truths(sources)}, forecast=_forecast())
    assert isinstance(evaluation, MultiEvaluation) and evaluation.kind == "truth"
    states = evaluation.run()
    assert CountingForecast.rollouts == 2  # one rollout per init time, shared by the three truths
    for name, block in _truths(sources).items():
        assert states[name].attrs["truth"] == name and "dataset" not in states[name].attrs
        alone = build(_alone(name, block), forecast=_forecast())
        expected = alone.run().to_xarray(alone.metrics)
        got = states[name].to_xarray(evaluation.metrics[name])
        assert got.lead_time.values.tolist() == expected.lead_time.values.tolist()
        for key in ("rmse", "fair_crps", "spread", "weight_sum"):
            xr.testing.assert_equal(got[key], expected[key])
    hours = lambda state: [int(h / np.timedelta64(1, "h")) for h in state.to_xarray(evaluation.metrics["radar"]).lead_time.values]  # noqa: E731
    assert hours(states["radar"]) == [6, 12, 18] and hours(states["analysis"])[0] == 0  # lead 0 per truth
    assert '"tp"' in states["radar"].attrs["forecast_minimum"]


def test_config_checks(sources):
    with pytest.raises(ValueError, match="belong to each truth"):
        EvaluationConfig.model_validate({**RUN, "truths": _truths(sources), "variables": ["2t"]})
    with pytest.raises(ValueError, match="truth"):
        EvaluationConfig.model_validate({**RUN, "truths": _truths(sources), "output": {"path": "out/run-{shard}.nc"}})
    bad = _truths(sources)
    bad["synop"].pop("interpolation")
    with pytest.raises(ValueError, match="go together"):
        EvaluationConfig.model_validate({**RUN, "truths": bad})
    assert format_output_path("out/{truth}/run-{shard}.nc", 1, 4, "radar", "truth") == "out/radar/run-1.nc"
    with pytest.raises(ValueError, match="truth"):
        format_output_path("out/run-{shard}.nc", 1, 4, "radar", "truth")
