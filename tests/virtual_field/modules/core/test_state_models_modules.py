import pytest

from virtual_field.core.state import (
    ArmState,
    MeshEntity,
    SceneState,
    SphereEntity,
    Transform,
)

pytestmark = pytest.mark.modules


def _make_arm(arm_id: str) -> ArmState:
    return ArmState(
        arm_id=arm_id,
        owner_user_id="user_a",
        base=Transform(
            translation=[0.0, 0.0, 0.0], rotation_xyzw=[0.0, 0.0, 0.0, 1.0]
        ),
        tip=Transform(
            translation=[0.0, 0.0, 1.0], rotation_xyzw=[0.0, 0.0, 0.0, 1.0]
        ),
        centerline=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        radii=[0.1],
        directors=[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
    )


def test_scene_state_round_trip() -> None:
    mesh = MeshEntity(
        mesh_id="mesh_1",
        owner_id="publisher_1",
        asset_uri="data:model/gltf-binary;base64,AA==",
    )
    sphere = SphereEntity(
        sphere_id="sphere_1",
        owner_id="user_a",
        translation=[1.0, 2.0, 3.0],
        radius=0.25,
        color_rgb=[0.95, 0.45, 0.08],
    )
    state = SceneState(
        timestamp=1.25,
        arms={
            "arm_a": _make_arm("arm_a"),
            "arm_b": _make_arm("arm_b"),
        },
        meshes={"mesh_1": mesh},
        spheres={"sphere_1": sphere},
    )

    encoded = state.to_dict()
    decoded = SceneState.from_dict(encoded)

    assert decoded.timestamp == pytest.approx(1.25)
    assert set(decoded.arms.keys()) == {"arm_a", "arm_b"}
    assert decoded.arms["arm_a"].tip.translation[2] == pytest.approx(1.0)
    assert decoded.arms["arm_a"].owner_user_id == "user_a"
    assert decoded.arms["arm_a"].directors[0][0][0] == pytest.approx(1.0)
    assert decoded.meshes["mesh_1"].owner_id == "publisher_1"
    assert decoded.spheres["sphere_1"].radius == pytest.approx(0.25)
    assert decoded.spheres["sphere_1"].color_rgb[0] == pytest.approx(0.95)


def test_scene_state_allows_empty_collections() -> None:
    state = SceneState(timestamp=0.0, arms={})
    assert state.arms == {}


def test_static_mesh_scene_state_omits_asset_uri_after_first_client_payload() -> (
    None
):
    mesh = MeshEntity(
        mesh_id="terrain",
        owner_id="user_1",
        asset_uri="data:model/gltf-binary;base64,QUJD",
        static_asset=True,
    )
    state = SceneState(timestamp=0.0, arms={}, meshes={"terrain": mesh})
    sent: set[str] = set()
    first = state.to_dict_for_client(sent)
    assert "asset_uri" in first["meshes"]["terrain"]
    assert first["meshes"]["terrain"]["static_asset"] is True
    assert "terrain" in sent

    second = state.to_dict_for_client(sent)
    assert "asset_uri" not in second["meshes"]["terrain"]
    assert second["meshes"]["terrain"]["mesh_id"] == "terrain"


def test_static_mesh_removed_clears_sent_ids_for_resend() -> None:
    mesh = MeshEntity(
        mesh_id="terrain",
        owner_id="user_1",
        asset_uri="data:model/gltf-binary;base64,QUJD",
        static_asset=True,
    )
    sent: set[str] = set()
    state = SceneState(timestamp=0.0, arms={}, meshes={"terrain": mesh})
    state.to_dict_for_client(sent)
    assert sent == {"terrain"}

    empty_meshes = SceneState(timestamp=0.1, arms={}, meshes={})
    empty_meshes.to_dict_for_client(sent)
    assert sent == set()

    again = SceneState(timestamp=0.2, arms={}, meshes={"terrain": mesh})
    payload = again.to_dict_for_client(sent)
    assert "asset_uri" in payload["meshes"]["terrain"]


def test_arm_state_rejects_invalid_centerline() -> None:
    with pytest.raises(ValueError, match="centerline"):
        ArmState(
            arm_id="a",
            owner_user_id=None,
            base=Transform(),
            tip=Transform(),
            centerline=[[0.0, 0.0]],
            radii=[],
        )


def test_arm_state_rejects_invalid_director_shape() -> None:
    with pytest.raises(ValueError, match="directors"):
        ArmState(
            arm_id="a",
            owner_user_id=None,
            base=Transform(),
            tip=Transform(),
            centerline=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            radii=[0.1],
            directors=[[[1.0, 0.0], [0.0, 1.0]]],
        )
