from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


JSONDict = dict[str, Any]


def _validate_vector_shape(values: list[float], size: int, name: str) -> None:
    if len(values) != size:
        raise ValueError(f"{name} must have size {size}, got {len(values)}")


@dataclass(slots=True)
class Transform:
    """Rigid pose in world space for wire protocol and rendering.

    ``translation`` is ``[x, y, z]``. ``rotation_xyzw`` is a unit quaternion
    ``[x, y, z, w]`` (scalar-last), matching common VR/controller conventions.
    """

    translation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_xyzw: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 1.0]
    )

    def __post_init__(self) -> None:
        _validate_vector_shape(self.translation, 3, "translation")
        _validate_vector_shape(self.rotation_xyzw, 4, "rotation_xyzw")

    def to_dict(self) -> JSONDict:
        return {
            "translation": self.translation,
            "rotation_xyzw": self.rotation_xyzw,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> "Transform":
        return cls(
            translation=list(data["translation"]),
            rotation_xyzw=list(data["rotation_xyzw"]),
        )


@dataclass(slots=True)
class Twist:
    linear: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self) -> None:
        _validate_vector_shape(self.linear, 3, "linear")
        _validate_vector_shape(self.angular, 3, "angular")

    def to_dict(self) -> JSONDict:
        return {"linear": self.linear, "angular": self.angular}

    @classmethod
    def from_dict(cls, data: JSONDict) -> "Twist":
        return cls(linear=list(data["linear"]), angular=list(data["angular"]))


@dataclass(slots=True)
class ArmState:
    arm_id: str
    owner_user_id: str | None
    base: Transform
    tip: Transform
    centerline: list[list[float]]
    radii: list[float]
    element_lengths: list[float] = field(default_factory=list)
    directors: list[list[list[float]]] = field(default_factory=list)
    contact_points: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.arm_id:
            raise ValueError("arm_id cannot be empty")
        for idx, point in enumerate(self.centerline):
            _validate_vector_shape(point, 3, f"centerline[{idx}]")
        for radius in self.radii:
            if radius < 0:
                raise ValueError("radii cannot contain negative values")
        for length in self.element_lengths:
            if length < 0:
                raise ValueError("element_lengths cannot contain negative values")
        for idx, director in enumerate(self.directors):
            if len(director) != 3:
                raise ValueError(f"directors[{idx}] must be 3x3")
            for jdx, row in enumerate(director):
                _validate_vector_shape(row, 3, f"directors[{idx}][{jdx}]")
        for idx, point in enumerate(self.contact_points):
            _validate_vector_shape(point, 3, f"contact_points[{idx}]")

    def to_dict(self) -> JSONDict:
        return {
            "arm_id": self.arm_id,
            "owner_user_id": self.owner_user_id,
            "base": self.base.to_dict(),
            "tip": self.tip.to_dict(),
            "centerline": self.centerline,
            "radii": self.radii,
            "element_lengths": self.element_lengths,
            "directors": self.directors,
            "contact_points": self.contact_points,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> "ArmState":
        return cls(
            arm_id=str(data["arm_id"]),
            owner_user_id=(
                str(data["owner_user_id"])
                if data.get("owner_user_id") is not None
                else None
            ),
            base=Transform.from_dict(data["base"]),
            tip=Transform.from_dict(data["tip"]),
            centerline=[list(point) for point in data["centerline"]],
            radii=list(data["radii"]),
            element_lengths=list(data.get("element_lengths", [])),
            directors=[
                [list(row) for row in director]
                for director in data.get("directors", [])
            ],
            contact_points=[
                list(point) for point in data.get("contact_points", [])
            ],
        )


@dataclass(slots=True)
class MeshEntity:
    mesh_id: str
    owner_id: str
    asset_uri: str
    translation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_xyzw: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 1.0]
    )
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    visible: bool = True

    def __post_init__(self) -> None:
        if not self.mesh_id:
            raise ValueError("mesh_id cannot be empty")
        if not self.owner_id:
            raise ValueError("owner_id cannot be empty")
        if not self.asset_uri:
            raise ValueError("asset_uri cannot be empty")
        _validate_vector_shape(self.translation, 3, "translation")
        _validate_vector_shape(self.rotation_xyzw, 4, "rotation_xyzw")
        _validate_vector_shape(self.scale, 3, "scale")

    def to_dict(self) -> JSONDict:
        return {
            "mesh_id": self.mesh_id,
            "owner_id": self.owner_id,
            "asset_uri": self.asset_uri,
            "translation": self.translation,
            "rotation_xyzw": self.rotation_xyzw,
            "scale": self.scale,
            "visible": self.visible,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> "MeshEntity":
        return cls(
            mesh_id=str(data["mesh_id"]),
            owner_id=str(data["owner_id"]),
            asset_uri=str(data["asset_uri"]),
            translation=list(data.get("translation", [0.0, 0.0, 0.0])),
            rotation_xyzw=list(data.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0])),
            scale=list(data.get("scale", [1.0, 1.0, 1.0])),
            visible=bool(data.get("visible", True)),
        )


@dataclass(slots=True)
class OverlayPointsEntity:
    overlay_id: str
    owner_id: str
    points: list[list[float]] = field(default_factory=list)
    point_size: float = 0.008
    visible: bool = True

    def __post_init__(self) -> None:
        if not self.overlay_id:
            raise ValueError("overlay_id cannot be empty")
        if not self.owner_id:
            raise ValueError("owner_id cannot be empty")
        if self.point_size <= 0.0:
            raise ValueError("point_size must be > 0")
        for idx, point in enumerate(self.points):
            _validate_vector_shape(point, 3, f"points[{idx}]")

    def to_dict(self) -> JSONDict:
        return {
            "overlay_id": self.overlay_id,
            "owner_id": self.owner_id,
            "points": self.points,
            "point_size": self.point_size,
            "visible": self.visible,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> "OverlayPointsEntity":
        return cls(
            overlay_id=str(data["overlay_id"]),
            owner_id=str(data["owner_id"]),
            points=[list(point) for point in data.get("points", [])],
            point_size=float(data.get("point_size", 0.008)),
            visible=bool(data.get("visible", True)),
        )


@dataclass(slots=True)
class SphereEntity:
    sphere_id: str
    owner_id: str
    translation: list[float]
    radius: float
    color_rgb: list[float] = field(default_factory=lambda: [0.95, 0.45, 0.08])
    visible: bool = True

    def __post_init__(self) -> None:
        if not self.sphere_id:
            raise ValueError("sphere_id cannot be empty")
        if not self.owner_id:
            raise ValueError("owner_id cannot be empty")
        _validate_vector_shape(self.translation, 3, "translation")
        _validate_vector_shape(self.color_rgb, 3, "color_rgb")
        if self.radius <= 0.0:
            raise ValueError("radius must be > 0")

    def to_dict(self) -> JSONDict:
        return {
            "sphere_id": self.sphere_id,
            "owner_id": self.owner_id,
            "translation": self.translation,
            "radius": self.radius,
            "color_rgb": self.color_rgb,
            "visible": self.visible,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> "SphereEntity":
        return cls(
            sphere_id=str(data["sphere_id"]),
            owner_id=str(data["owner_id"]),
            translation=list(data.get("translation", [0.0, 0.0, 0.0])),
            radius=float(data["radius"]),
            color_rgb=list(data.get("color_rgb", [0.95, 0.45, 0.08])),
            visible=bool(data.get("visible", True)),
        )


@dataclass(slots=True)
class SceneState:
    timestamp: float
    arms: dict[str, ArmState]
    scenery: dict[str, Transform] = field(default_factory=dict)
    user_arms: dict[str, list[str]] = field(default_factory=dict)
    meshes: dict[str, MeshEntity] = field(default_factory=dict)
    overlay_points: dict[str, OverlayPointsEntity] = field(default_factory=dict)
    spheres: dict[str, SphereEntity] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, arm in self.arms.items():
            if key != arm.arm_id:
                raise ValueError("arms keys must match ArmState.arm_id")

    def to_dict(self) -> JSONDict:
        return {
            "timestamp": self.timestamp,
            "arms": {arm_id: arm.to_dict() for arm_id, arm in self.arms.items()},
            "scenery": {
                name: transform.to_dict()
                for name, transform in self.scenery.items()
            },
            "user_arms": self.user_arms,
            "meshes": {
                mesh_id: mesh.to_dict() for mesh_id, mesh in self.meshes.items()
            },
            "overlay_points": {
                overlay_id: overlay.to_dict()
                for overlay_id, overlay in self.overlay_points.items()
            },
            "spheres": {
                sphere_id: sphere.to_dict()
                for sphere_id, sphere in self.spheres.items()
            },
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> "SceneState":
        return cls(
            timestamp=float(data["timestamp"]),
            arms={
                arm_id: ArmState.from_dict(arm_data)
                for arm_id, arm_data in data["arms"].items()
            },
            scenery={
                name: Transform.from_dict(transform)
                for name, transform in data.get("scenery", {}).items()
            },
            user_arms={
                user_id: list(arm_ids)
                for user_id, arm_ids in data.get("user_arms", {}).items()
            },
            meshes={
                mesh_id: MeshEntity.from_dict(mesh_data)
                for mesh_id, mesh_data in data.get("meshes", {}).items()
            },
            overlay_points={
                overlay_id: OverlayPointsEntity.from_dict(overlay_data)
                for overlay_id, overlay_data in data.get("overlay_points", {}).items()
            },
            spheres={
                sphere_id: SphereEntity.from_dict(sphere_data)
                for sphere_id, sphere_data in data.get("spheres", {}).items()
            },
        )
