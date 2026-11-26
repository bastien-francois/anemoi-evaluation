import datetime
import json

import numpy as np
import pytest
import torch
from anemoi.utils.dates import frequency_to_string

from anemoi.evaluation.config import EvaluationConfig
from anemoi.evaluation.evaluate import Evaluation
from anemoi.evaluation.frame import Frame

HOUR = datetime.timedelta(hours=1)
T0 = datetime.datetime(2024, 1, 1)
INITS = [T0, T0 + 12 * HOUR]


def _evaluation(forecast, targets, **kwargs):
    return Evaluation(forecast, targets, INITS, "24h", ["rmse", "fair_crps", "spread"], **kwargs)


def test_values_raised_counted_and_scored(fake_sources):
    forecast, targets = fake_sources(members=3, drift=-0.5, spread=0.2)  # q drifts below 0
    plain = _evaluation(forecast, targets)
    bounded = _evaluation(forecast, targets, forecast_minimum={"q": 0.0})
    expected = {}
    for (frame, target), (clipped, same_target) in zip(list(plain.pairs()), list(bounded.pairs()), strict=True):
        t, q = frame.variables.index("t"), frame.variables.index("q")
        assert torch.equal(clipped.data[:, t], frame.data[:, t])  # other variables untouched
        torch.testing.assert_close(clipped.data[:, q], frame.data[:, q].clamp(min=0.0))
        assert torch.equal(target, same_target)
        lead = frame.lead_time
        below = frame.data[:, q][frame.data[:, q] < 0]
        count, smallest = expected.get(lead, (0, np.inf))
        expected[lead] = (count + below.numel(), min(smallest, float(below.min())) if below.numel() else smallest)
    counts = bounded.forecast_minimum_counts()["q"]
    assert sum(entry["raised"] for entry in counts.values()) == sum(c for c, _ in expected.values()) > 0
    for lead, (count, smallest) in expected.items():
        entry = counts[frequency_to_string(lead)]
        assert entry["raised"] == count and (entry["smallest"] == pytest.approx(smallest) if count else entry["smallest"] is None)

    result = _evaluation(forecast, targets, forecast_minimum={"q": 0.0})
    state = result.run()
    stored = json.loads(state.attrs["forecast_minimum"])
    assert stored["minimum"] == {"q": 0.0} and stored["raised"] == json.loads(json.dumps({"q": counts}))
    scores = state.to_xarray(result.metrics).sel(bin="all", region="global")
    reference = plain.run().to_xarray(plain.metrics).sel(bin="all", region="global")
    np.testing.assert_array_equal(scores["rmse"].sel(variable="t"), reference["rmse"].sel(variable="t"))
    assert not np.allclose(scores["rmse"].sel(variable="q"), reference["rmse"].sel(variable="q"))


def test_copy_not_in_place_and_nan_kept(fake_sources):
    forecast, targets = fake_sources(members=2)
    evaluation = _evaluation(forecast, targets, forecast_minimum={"q": 0.0})
    data = torch.tensor([[[1.0, -1.0, float("nan")], [-2.0, -3.0, 4.0]]] * 2)  # (M=2, V=2 [t, q], N=3)
    frame = Frame(T0, 6 * HOUR, T0 + 6 * HOUR, ["t", "q"], data.clone())
    out = evaluation.apply_forecast_minimum(frame)
    assert torch.allclose(frame.data, data, equal_nan=True)  # the input frame is not written
    assert out.data[:, 1].tolist() == [[0.0, 0.0, 4.0]] * 2  # q raised to 0
    assert torch.allclose(out.data[:, 0], data[:, 0], equal_nan=True)  # t (and its NaN) untouched
    positive = Frame(T0, 6 * HOUR, T0 + 6 * HOUR, ["t", "q"], data.abs())
    assert evaluation.apply_forecast_minimum(positive) is positive  # nothing to raise: no copy
    assert evaluation.forecast_minimum_counts()["q"][frequency_to_string(6 * HOUR)] == {"raised": 4, "smallest": -3.0}


def test_unknown_variable_ignored_and_config(fake_sources, caplog):
    forecast, targets = fake_sources()
    evaluation = _evaluation(forecast, targets, forecast_minimum={"tp": 0.0})
    assert evaluation.forecast_minimum == {} and "not scored" in caplog.text
    assert "forecast_minimum" not in evaluation.plan()
    config = {"forecast": {"persistence": {"timestep": "6h"}}, "lead_time": "24h", "init_times": {"dates": ["2024-01-01"]}}
    assert EvaluationConfig.model_validate(config).forecast_minimum is None
    assert EvaluationConfig.model_validate({**config, "forecast_minimum": {"tp": 0}}).forecast_minimum == {"tp": 0.0}
