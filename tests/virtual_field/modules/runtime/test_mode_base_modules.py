import numpy as np
import pytest

from virtual_field.runtime.mode_base import OctoArmSimulationBase
from virtual_field.runtime.two_gcr_simulation import TwoGCRSimulation

pytestmark = pytest.mark.modules


class _DummyStepper:
    def step(
        self, simulator: object, time_value: float, step_dt: float
    ) -> float:
        return time_value + step_dt


class _DummySimulation(OctoArmSimulationBase):
    def build_simulation(self) -> None:
        self.simulator = object()
        self.timestepper = _DummyStepper()
        self.rods = {
            arm_id: self._make_rod(base)
            for arm_id, base in self.arm_bases.items()
        }

    def _make_rod(self, base: list[float]) -> object:
        return type(
            "DummyRod",
            (),
            {
                "position_collection": np.array(
                    [
                        [base[0], base[0], base[0]],
                        [base[1], base[1], base[1]],
                        [base[2], base[2] - 0.2, base[2] - 0.4],
                    ],
                    dtype=np.float64,
                ),
                "radius": np.array([0.03, 0.02], dtype=np.float64),
                "lengths": np.array([0.2, 0.2], dtype=np.float64),
                "director_collection": np.repeat(
                    np.eye(3, dtype=np.float64)[..., None], 2, axis=2
                ),
            },
        )()

    def handle_commands(self, command) -> None:  # noqa: ANN001
        _ = command


def _make_dummy_rod(base: list[float]) -> object:
    return type(
        "DummyRod",
        (),
        {
            "position_collection": np.array(
                [
                    [base[0], base[0], base[0]],
                    [base[1], base[1], base[1]],
                    [base[2], base[2] - 0.2, base[2] - 0.4],
                ],
                dtype=np.float64,
            ),
            "radius": np.array([0.03, 0.02], dtype=np.float64),
            "lengths": np.array([0.2, 0.2], dtype=np.float64),
            "director_collection": np.repeat(
                np.eye(3, dtype=np.float64)[..., None], 2, axis=2
            ),
        },
    )()


class _TwoGCRInitRegression(TwoGCRSimulation):
    def build_simulation(self) -> None:
        self.wave_event_before_post_setup = self._grip_wave_event.get(
            self.arm_ids[0], 0
        )
        self.simulator = object()
        self.timestepper = _DummyStepper()
        self.rods = {
            arm_id: _make_dummy_rod(base)
            for arm_id, base in self.arm_bases.items()
        }


def _base_position() -> tuple[float, float, float]:
    return (0.0, 1.0, -0.15)


def test_initialize_octo_arm_targets_sets_rest_pose() -> None:
    simulation = _DummySimulation(
        user_id="user_dummy",
        arm_ids=tuple(f"arm_{index}" for index in range(8)),
        base_position=_base_position(),
        dt_internal=0.1,
    )

    assert len(simulation._target_position) == 8
    assert len(simulation._rest_target_position) == 8
    assert all(
        np.isclose(position[1], 1.0)
        for position in simulation._target_position.values()
    )
    assert len(simulation._attached) == 8
    assert all(simulation._attached.values())


def test_base_arm_states_include_lengths_and_directors() -> None:
    simulation = _DummySimulation(
        user_id="user_dummy",
        arm_ids=tuple(f"arm_{index}" for index in range(8)),
        base_position=_base_position(),
        dt_internal=0.1,
    )

    state = simulation.arm_states()["arm_0"]
    assert state.owner_user_id == "user_dummy"
    assert state.element_lengths == [0.2, 0.2]
    assert len(state.directors) == 2
    assert state.contact_points == []


def test_base_step_uses_internal_substeps() -> None:
    simulation = _DummySimulation(
        user_id="user_dummy",
        arm_ids=tuple(f"arm_{index}" for index in range(8)),
        base_position=_base_position(),
        dt_internal=0.1,
    )

    simulation.step(0.25)

    assert np.isclose(simulation._time, 0.25)


def test_arm_states_use_active_rod_slices_when_available() -> None:
    simulation = _DummySimulation(
        user_id="user_dummy",
        arm_ids=tuple(f"arm_{index}" for index in range(8)),
        base_position=_base_position(),
        dt_internal=0.1,
    )
    rod = simulation.rods["arm_0"]
    rod._active_node_slice = lambda: slice(1, None)
    rod._active_elem_slice = lambda: slice(1, None)

    state = simulation.arm_states()["arm_0"]

    assert np.allclose(
        np.asarray(state.centerline, dtype=np.float64),
        np.asarray(rod.position_collection.T[1:], dtype=np.float64),
    )
    assert state.radii == [0.02]
    assert state.element_lengths == [0.2]
    assert len(state.directors) == 1


def test_two_gcr_control_state_exists_during_build_simulation() -> None:
    simulation = _TwoGCRInitRegression(
        user_id="user_dummy",
        arm_ids=("left_arm", "right_arm"),
        base_left=[-0.15, 0.2, 0.15],
        base_right=[0.15, 0.2, 0.15],
        dt_internal=0.1,
    )

    assert simulation.wave_event_before_post_setup == 0
    assert simulation._grip_wave_event == {"left_arm": 0, "right_arm": 0}
