import datetime
import json

import numpy as np
import pytest
import torch

from anemoi.evaluation.config import TargetsConfig
from anemoi.evaluation.sources.anemoi_dataset import DatasetTargets
from anemoi.evaluation.target_qc import distance_to_nearest_km

from test_real_sources import FakeDataset

HOUR = datetime.timedelta(hours=1)
T0 = datetime.datetime(2024, 1, 1)
CPU = torch.device("cpu")


class FakeQC:
    """A QC dataset on sub-grid 1 (6 nodes) of FakeDataset: field `x`, NaN except a few values."""

    def __init__(self, n_dates: int = 4) -> None:
        self.dates = np.array([T0 + 6 * k * HOUR for k in range(n_dates)], dtype="datetime64[s]")
        self.variables = ["x"]
        self.shape = (n_dates, 1, 1, 6)
        self.values = np.full((n_dates, 6), np.nan, dtype=np.float32)
        self.values[:, 1] = 0.01
        self.values[1, 4] = 0.06  # above 0.05 at T0 + 6 h only
        self.values[2, 4] = 0.05  # equal at T0 + 12 h: kept (strict)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.values[i][None, None, :]


@pytest.fixture
def qc_dataset(monkeypatch):
    opened = []

    def fake_open_dataset(*args, **kwargs):
        opened.append((args, kwargs))
        return FakeQC()

    monkeypatch.setattr("anemoi.datasets.open_dataset", fake_open_dataset)
    return opened


def _targets():
    dataset = FakeDataset()
    site = [float(dataset.longitudes[6]), float(dataset.latitudes[6])]  # node 2 of sub-grid 1
    qc = [
        {"max_distance": {"variables": ["a"], "grid": 1, "max_km": 1.0, "sites": {"here": site}}},
        {"drop_where": {"variables": ["a", "b"], "grid": 1, "dataset": "qc.zarr", "variable": "x", "above": 0.05}},
        {"drop_dates": {"variables": ["b"], "grid": 1, "start": "2024-01-01T12", "end": "2024-01-01T18"}},
    ]
    return DatasetTargets.from_dataset(dataset, kwargs={"dataset": "fake"}, qc=qc)


def test_distance():
    sites = {"a": [0.0, 0.0], "b": [90.0, 0.0]}  # lon, lat
    d = distance_to_nearest_km(np.array([0.0, 0.0, 90.0]), np.array([0.0, 45.0, 0.0]), sites)
    quarter = np.pi * 6371.0 / 2
    np.testing.assert_allclose(d, [0.0, quarter / 2, quarter], rtol=1e-12, atol=1e-9)


def test_rules(qc_dataset):
    targets = _targets()
    assert qc_dataset == [(("qc.zarr",), {})]
    nan = np.isnan
    t0 = targets.frame(T0, ["a", "b", "c"], CPU)[0].numpy()
    # max_distance on a: sub-grid 0 (nodes 0-3) untouched, in sub-grid 1 (nodes 4-9) only node 6 is near the site
    assert nan(t0[0]).tolist() == [False] * 4 + [True, True, False, True, True, True]
    assert not nan(t0[1]).any() and not nan(t0[2]).any()  # b: nothing at T0; c: no rule
    t6 = targets.frame(T0 + 6 * HOUR, ["b", "a"], CPU)[0].numpy()
    assert nan(t6[0]).tolist() == [False] * 8 + [True, False]  # drop_where: cell 4 of sub-grid 1 = node 8
    t12 = targets.frame(T0 + 12 * HOUR, ["b", "a"], CPU)[0].numpy()
    assert nan(t12[0]).tolist() == [False] * 4 + [True] * 6  # drop_dates, inclusive, sub-grid 1 only
    assert nan(t12[1]).sum() == 5  # a: only max_distance; 0.05 is not above 0.05
    np.testing.assert_array_equal(t12[1][[0, 6]], [200.0, 206.0])  # values kept are unchanged
    counts = {(c["rule"], c["variable"]): (c["removed"], c["valid_times"]) for c in targets.qc.counts()}
    assert counts[("max_distance", "a")] == (15, 3)
    assert counts[("drop_where", "a")] == (0, 3)  # node 8 was already NaN
    assert counts[("drop_where", "b")] == (1, 3)
    assert counts[("drop_dates", "b")] == (6, 3)


def test_counts_not_doubled_by_rereads(qc_dataset):
    targets = _targets()
    for _ in range(3):
        targets.frame(T0, ["a"], CPU)
    assert targets.qc.counts()[0] == {"rule": "max_distance", "variable": "a", "removed": 5, "valid_times": 1}


def test_errors(qc_dataset):
    sites = {"s": [0.0, 0.0]}
    where = {"variables": ["a"], "grid": 1, "dataset": "q", "variable": "x", "above": 0}
    with pytest.raises(ValueError, match="not in the targets"):
        DatasetTargets.from_dataset(FakeDataset(), qc=[{"max_distance": {"variables": ["x"], "max_km": 1, "sites": sites}}])
    with pytest.raises(ValueError, match="one of"):
        DatasetTargets.from_dataset(FakeDataset(), qc=[{"clip": {"variables": ["a"]}}])
    with pytest.raises(ValueError, match="out of range"):
        DatasetTargets.from_dataset(
            FakeDataset(), qc=[{"max_distance": {"variables": ["a"], "grid": 2, "max_km": 1, "sites": sites}}]
        )
    with pytest.raises(ValueError, match="lon, lat"):
        DatasetTargets.from_dataset(FakeDataset(), qc=[{"max_distance": {"variables": ["a"], "max_km": 1, "sites": {}}}])
    with pytest.raises(ValueError, match="nodes, sub-grid has"):
        DatasetTargets.from_dataset(FakeDataset(), qc=[{"drop_where": {**where, "grid": 0}}])
    with pytest.raises(ValueError, match="not in the QC dataset"):
        DatasetTargets.from_dataset(FakeDataset(), qc=[{"drop_where": {**where, "variable": "y"}}])
    with pytest.raises(ValueError, match="before start"):
        DatasetTargets.from_dataset(
            FakeDataset(), qc=[{"drop_dates": {"variables": ["a"], "start": "2024-01-02", "end": "2024-01-01"}}]
        )
    late = DatasetTargets.from_dataset(FakeDataset(n_dates=5), qc=[{"drop_where": where}])
    with pytest.raises(ValueError, match="no valid time"):
        late.frame(T0 + 24 * HOUR, ["a"], CPU)


def test_config_round_trip(qc_dataset):
    targets = _targets()
    block = TargetsConfig.model_validate(targets.to_config()).anemoi_dataset
    assert block.qc == targets.qc.specs and "qc" not in block.open_dataset_kwargs()
    assert json.dumps(targets.describe()["qc"], default=str)


def test_no_qc_unchanged():
    plain = DatasetTargets.from_dataset(FakeDataset())
    assert plain.qc is None and "qc" not in plain.to_config()["anemoi_dataset"]
    assert not np.isnan(plain.frame(T0, ["a"], CPU).numpy()).any()


def test_only_grid():
    targets = DatasetTargets.from_dataset(FakeDataset(), qc=[{"only_grid": {"variables": ["c"], "grid": 1}}])
    frame = targets.frame(T0, ["a", "c"], CPU)[0].numpy()
    assert not np.isnan(frame[0]).any()  # a: no rule
    assert np.isnan(frame[1]).tolist() == [True] * 4 + [False] * 6  # c: only sub-grid 1 left
    assert targets.qc.counts() == [{"rule": "only_grid", "variable": "c", "removed": 4, "valid_times": 1}]
    whole = DatasetTargets.from_dataset(FakeDataset(grids=(10,)), qc=[{"only_grid": {"variables": ["c"]}}])
    assert not np.isnan(whole.frame(T0, ["c"], CPU).numpy()).any()  # a single grid: nothing to remove


def test_attr_and_merge_across_shards(qc_dataset):
    from anemoi.evaluation.target_qc import merge_attrs

    first, second = _targets(), _targets()  # two shards; both read T0 + 6 h (an overlapping valid time)
    for date in (T0, T0 + 6 * HOUR):
        first.frame(date, ["a", "b"], CPU)
    for date in (T0 + 6 * HOUR, T0 + 12 * HOUR):
        second.frame(date, ["a", "b"], CPU)
    attr = json.loads(first.qc.attr())
    assert attr["removed"] == first.qc.counts() and attr["rules"] == first.qc.specs
    merged = json.loads(merge_attrs([first.qc.attr(), second.qc.attr()]))
    totals = {(c["rule"], c["variable"]): (c["removed"], c["valid_times"]) for c in merged["removed"]}
    assert totals[("max_distance", "a")] == (15, 3)  # 5 per valid time, T0 + 6 h counted once
    assert totals[("drop_where", "b")] == (1, 3) and totals[("drop_dates", "b")] == (6, 3)
    single = _targets()
    for date in (T0, T0 + 6 * HOUR, T0 + 12 * HOUR):
        single.frame(date, ["a", "b"], CPU)
    assert merged["removed"] == single.qc.counts()  # = one run over the three valid times
    other = DatasetTargets.from_dataset(FakeDataset(), qc=[{"only_grid": {"variables": ["c"], "grid": 1}}])
    with pytest.raises(ValueError, match="different qc rules"):
        merge_attrs([first.qc.attr(), other.qc.attr()])
    legacy = json.dumps({"rules": first.qc.specs, "removed": first.qc.counts()})  # written before by_valid_time
    assert merge_attrs([legacy, second.qc.attr()]) == legacy


def test_valid_range():
    targets = DatasetTargets.from_dataset(
        FakeDataset(), qc=[{"valid_range": {"variables": ["a"], "min": 3.0, "max": 7.0}}, {"valid_range": {"variables": ["b"], "max": 15.0}}]
    )
    frame = targets.frame(T0, ["a", "b"], CPU)[0].numpy()  # a = 0..9, b = 10..19 at T0
    assert np.isnan(frame[0]).tolist() == [True] * 3 + [False] * 5 + [True] * 2
    assert np.isnan(frame[1]).tolist() == [False] * 6 + [True] * 4
    assert [c["removed"] for c in targets.qc.counts()] == [5, 4]
    with pytest.raises(ValueError, match="min"):
        DatasetTargets.from_dataset(FakeDataset(), qc=[{"valid_range": {"variables": ["a"]}}])
