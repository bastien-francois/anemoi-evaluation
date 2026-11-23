import datetime
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from anemoi.evaluation.config import AnemoiInferenceConfig
from anemoi.evaluation.derived import base_variables
from anemoi.evaluation.derived import derivable
from anemoi.evaluation.derived import derive
from anemoi.evaluation.derived import derived_variable_info
from anemoi.evaluation.sources.anemoi_dataset import DatasetTargets
from anemoi.evaluation.sources.anemoi_inference import InferenceForecastSource
from anemoi.evaluation.sources.anemoi_inference import format_init_time

HOUR = datetime.timedelta(hours=1)
T0 = datetime.datetime(2026, 7, 2)
CPU = torch.device("cpu")
N = 5


def test_derived_helpers():
    assert derivable(["2t", "10u", "10v"]) == ["10si"]
    assert derivable(["2t", "10u"]) == [] and derivable(["10u", "10v", "10si"]) == []
    assert base_variables(["10si", "2t", "10u"], ["10u", "10v", "2t"]) == ["10u", "10v", "2t"]
    assert base_variables(["10si"], ["10si", "10u", "10v"]) == ["10si"]  # held by the source: read, not derived
    data = np.array([[[3.0, 0.0], [4.0, 2.0], [7.0, 8.0]]], dtype=np.float32)  # (1, [10u, 10v, 2t], 2)
    out = derive(data, ["10u", "10v", "2t"], ["2t", "10si"])
    np.testing.assert_allclose(out, [[[7.0, 8.0], [5.0, 2.0]]])
    on_torch = derive(torch.from_numpy(data), ["10u", "10v", "2t"], ["10si"])
    torch.testing.assert_close(on_torch, torch.tensor([[[5.0, 2.0]]]))
    with pytest.raises(ValueError, match="cannot derive"):
        derive(data, ["10u", "10v", "2t"], ["msl"])
    assert derived_variable_info("10si") == {"param": "10si", "level": None} and derived_variable_info("2t") is None


class WindDataset:
    """What DatasetTargets reads; 10u = 3, 10v = 4 (+ date index), 2t = 280."""

    def __init__(self) -> None:
        self.dates = np.array([T0 + 6 * k * HOUR for k in range(3)], dtype="datetime64[s]")
        self.missing = set()
        self.variables = ["10u", "10v", "2t"]
        self.name_to_index = {name: i for i, name in enumerate(self.variables)}
        self.typed_variables = {}
        self.latitudes, self.longitudes = np.linspace(40, 50, N), np.linspace(0, 10, N)
        self.shape = (3, 3, 1, N)

    def __getitem__(self, i: int) -> np.ndarray:
        values = np.array([3.0 + i, 4.0 + i, 280.0])[:, None] + np.zeros(N)[None, :]
        return values.astype(np.float32)[:, None, :]


def test_dataset_targets_derive_10si_only_when_asked():
    targets = DatasetTargets.from_dataset(WindDataset())
    assert targets.variables == ["10u", "10v", "2t"] and targets.derived_variables == ["10si"]
    frame = targets.frame(T0 + 6 * HOUR, ["2t", "10si"], CPU)
    assert frame.shape == (1, 2, N) and frame.dtype == torch.float32
    torch.testing.assert_close(frame[0, 1], torch.full((N,), float(np.hypot(4.0, 5.0))))
    torch.testing.assert_close(frame[0, 0], torch.full((N,), 280.0))
    assert targets.variable_info("10si") == {"param": "10si", "level": None}
    ahead = DatasetTargets.from_dataset(WindDataset(), prefetch=2)
    ahead.prefetch([T0, T0 + 6 * HOUR], ["10si"])
    torch.testing.assert_close(ahead.frame(T0, ["10si"], CPU)[0, 0], torch.full((N,), 5.0))
    ahead.close()
    with pytest.raises(ValueError):
        targets.frame(T0, ["msl"], CPU)


def test_format_init_time_and_member_inputs_config():
    block = {"dataset": {"cutout": ["/a/mbr001.zarr", "/lbc/{yyyy}/{mm}/{dd}/lbc_{yyyymmdd}_{hh}_mbr001.zarr"]}, "n": 1}
    assert format_init_time(block, T0 + 12 * HOUR) == {
        "dataset": {"cutout": ["/a/mbr001.zarr", "/lbc/2026/07/02/lbc_20260702_12_mbr001.zarr"]},
        "n": 1,
    }
    assert format_init_time("{yyyymmddhh}", T0) == "2026070200"
    config = AnemoiInferenceConfig.model_validate({"checkpoint": "x", "members": 2, "member_inputs": [block, block]})
    assert config.member_inputs == [block, block]
    with pytest.raises(ValueError, match="2 input blocks for 3 members"):
        AnemoiInferenceConfig.model_validate({"checkpoint": "x", "members": 3, "member_inputs": [block, block]})


class FakeInput:
    """One member's prognostics at an init time: 10u = 3 (m + 1), 10v = 4 (m + 1), 2t = init hour + m."""

    def __init__(self, member: int, init_time: datetime.datetime) -> None:
        self.member, self.init_time = member, init_time

    def create_input_state(self, *, dates, **kwargs):
        m = self.member
        values = {"10u": 3.0 * (m + 1), "10v": 4.0 * (m + 1), "2t": float(self.init_time.hour + m)}
        fields = {v: np.full((len(dates), N), x, dtype=np.float32) for v, x in values.items()}
        return {"date": dates[-1], "fields": fields}


class EmptyInput:
    def create_input_state(self, *, dates, **kwargs):
        return {"fields": {}}


def combine(*states):
    combined = {"date": states[0]["date"], "fields": {}}
    for state in states:
        combined["fields"].update(state["fields"])
    return combined


def fake_run(input_states, lead_time, return_numpy):
    """Each step adds the step count to the member's initial fields: members stay distinguishable."""
    state = input_states["ds"]
    for k in range(1, int(lead_time / (6 * HOUR)) + 1):
        out = {v: torch.as_tensor(x[-1]) + k for v, x in state["fields"].items()}
        yield {"ds": {"step": 6 * k * HOUR, "date": state["date"] + 6 * k * HOUR, "fields": out}}


def member_inputs_source(members: int = 2) -> InferenceForecastSource:
    source = InferenceForecastSource.__new__(InferenceForecastSource)
    source.dataset_names, source.dataset_name = ["ds"], "ds"
    source.datasets = {"ds": SimpleNamespace(variables=["10u", "10v", "2t"], grid=SimpleNamespace(n=N))}
    source.member_inputs = [{"dataset": f"/mbr00{m + 1}/{{yyyymmdd}}{{hh}}.zarr"} for m in range(members)]
    source.members, source.member_group, source.seed, source.quiet = members, None, 0, False
    source.rollout_seeds = None
    source.temporal_downscaling = False
    source.output_offsets, source.output_horizon = [6 * HOUR], 6 * HOUR
    source._initial = {}
    source.runner = SimpleNamespace(
        checkpoint=SimpleNamespace(multi_dataset_metadata={"ds": SimpleNamespace(lagged=[-6 * HOUR, 0 * HOUR])}),
        constant_forcings_inputs={"ds": EmptyInput()},
        dynamic_forcings_inputs={"ds": EmptyInput()},
        _combine_states=combine,
        run=fake_run,
        reference_date=None,
        device=torch.device("cpu"),
    )
    source._member_input = lambda member, init_time: FakeInput(member, init_time)
    return source


def test_member_inputs_give_each_member_its_own_initial_state_and_10si():
    source = member_inputs_source()
    assert source.member_input_config(1, T0 + 12 * HOUR) == {"dataset": "/mbr002/2026070212.zarr"}
    frames = list(source.frames(T0 + 12 * HOUR, 12 * HOUR, ["2t", "10si"], CPU))
    assert [f.lead_time for f in frames] == [6 * HOUR, 12 * HOUR] and frames[0].variables == ["2t", "10si"]
    for k, frame in enumerate(frames, start=1):
        assert frame.data.shape == (2, 2, N) and frame.data.dtype == torch.float32
        for m in range(2):
            torch.testing.assert_close(frame.data[m, 0], torch.full((N,), 12.0 + m + k))
            expected = float(np.hypot(3.0 * (m + 1) + k, 4.0 * (m + 1) + k))
            torch.testing.assert_close(frame.data[m, 1], torch.full((N,), expected))
    lead0 = source.initial_frame(T0, ["2t", "10si"], CPU)
    torch.testing.assert_close(lead0.data[:, 0, 0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(lead0.data[:, 1, 0], torch.tensor([5.0, 10.0]))
    source.member_group = SimpleNamespace()
    with pytest.raises(ValueError, match="need an unsplit run"):
        next(source.frames(T0, 6 * HOUR, ["10si"], CPU))


def test_member_inputs_are_opened_once(monkeypatch):
    import anemoi.inference.inputs
    import anemoi.inference.variables

    opened = []

    def create_input(runner, config, metadata, variables, purpose):
        opened.append(config)
        return SimpleNamespace(config=config)

    monkeypatch.setattr(anemoi.inference.inputs, "create_input", create_input)
    monkeypatch.setattr(
        anemoi.inference.variables, "Variables", lambda metadata: SimpleNamespace(retrieved_prognostic_variables=list)
    )
    source = member_inputs_source()
    del source._member_input  # the real method, not the fake
    first = source._member_input(0, T0)
    assert source._member_input(0, T0) is first and source._member_input(1, T0) is not first
    assert source._member_input(0, T0 + 24 * HOUR).config == {"dataset": "/mbr001/2026070300.zarr"}
    assert len(opened) == 3


# the inference script's seeds: BASE_SEED + init_hours * 10 + k, k = member slot 1..5 (from its output file names)
INFERENCE_SEEDS = {"base": 16014075, "init_hours_factor": 10, "member_offsets": [1, 2, 3, 4, 5]}


def test_rollout_seed_reproduces_the_inference_script_seeds():
    from anemoi.evaluation.sources.anemoi_inference import hours_since_epoch
    from anemoi.evaluation.sources.anemoi_inference import rollout_seed

    assert hours_since_epoch(datetime.datetime(1970, 1, 2, 6)) == 30
    assert rollout_seed(INFERENCE_SEEDS, datetime.datetime(2026, 7, 2), 0) == 20966716  # 2026070200_mbr001_k1
    assert rollout_seed(INFERENCE_SEEDS, datetime.datetime(2026, 7, 4), 1) == 20967197  # 2026070400_mbr002_k2
    assert rollout_seed(INFERENCE_SEEDS, datetime.datetime(2026, 7, 7), 4) == 20967920  # 2026070700_mbr005_k5
    aware = datetime.datetime(2026, 7, 4, 2, tzinfo=datetime.timezone(datetime.timedelta(hours=2)))
    assert hours_since_epoch(aware) == hours_since_epoch(datetime.datetime(2026, 7, 4))
    config = AnemoiInferenceConfig.model_validate({"checkpoint": "x", "members": 5, "rollout_seeds": INFERENCE_SEEDS})
    assert config.rollout_seeds.member_offsets == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="2 offsets for 5 members"):
        AnemoiInferenceConfig.model_validate(
            {"checkpoint": "x", "members": 5, "rollout_seeds": {"member_offsets": [1, 2]}}
        )


def noisy_run(input_states, lead_time, return_numpy):
    """Each step draws its noise from the global generator, as the model's noise injector does."""
    state = input_states["ds"]
    for k in range(1, int(lead_time / (6 * HOUR)) + 1):
        noise = torch.randn(N)
        yield {"ds": {"step": 6 * k * HOUR, "date": state["date"] + 6 * k * HOUR, "fields": {"2t": noise}}}


def test_rollout_seeds_give_each_lockstep_member_the_stream_of_its_own_run():
    source = member_inputs_source(members=3)
    source.runner.run = noisy_run
    source.rollout_seeds = {"base": 16014075, "init_hours_factor": 10, "member_offsets": [1, 2, 3]}
    init = datetime.datetime(2026, 7, 4)
    frames = list(source.frames(init, 18 * HOUR, ["2t"], CPU))  # 3 steps, members interleaved step by step
    for m in range(3):
        torch.manual_seed(16014075 + 10 * 495312 + m + 1)  # the seed of that member's own anemoi-inference run
        alone = [torch.randn(N) for _ in range(3)]
        for frame, expected in zip(frames, alone):
            torch.testing.assert_close(frame.data[m, 0], expected, rtol=0, atol=0)
    # a member's stream depends on its own seed only: dropping a member leaves the others unchanged
    source.members, source.member_inputs = 2, source.member_inputs[:2]
    source.rollout_seeds = {**source.rollout_seeds, "member_offsets": [1, 2]}
    pair = list(source.frames(init, 18 * HOUR, ["2t"], CPU))
    for frame, three in zip(pair, frames):
        torch.testing.assert_close(frame.data, three.data[:2], rtol=0, atol=0)
    # the default per-call reseeding draws different noise
    source.rollout_seeds = None
    default = list(source.frames(init, 18 * HOUR, ["2t"], CPU))
    assert not torch.equal(default[0].data, pair[0].data)
