from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np


# TODO: Test code to create arbitrary cylinder in gltf data format
def build_cylinder_gltf_data_uri(
    start: np.ndarray,
    direction: np.ndarray,
    normal: np.ndarray,
    length: float,
    radius: float,
    color_rgba: tuple[float, float, float, float] = (0.2, 0.75, 0.25, 1.0),
    radial_segments: int = 20,
) -> str:
    positions, normals, indices = _build_cylinder_geometry(
        np.asarray(start, dtype=np.float32).reshape(3),
        np.asarray(direction, dtype=np.float32).reshape(3),
        np.asarray(normal, dtype=np.float32).reshape(3),
        float(length),
        float(radius),
        radial_segments=radial_segments,
    )

    position_bytes = positions.astype("<f4", copy=False).tobytes()
    normal_bytes = normals.astype("<f4", copy=False).tobytes()
    index_bytes = indices.astype("<u4", copy=False).tobytes()
    buffer_blob = position_bytes + normal_bytes + index_bytes
    buffer_uri = (
        "data:application/octet-stream;base64,"
        + base64.b64encode(buffer_blob).decode("ascii")
    )

    position_offset = 0
    normal_offset = len(position_bytes)
    index_offset = normal_offset + len(normal_bytes)
    primitive_count = int(indices.size)

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorFactor": list(color_rgba),
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
                "doubleSided": True,
            }
        ],
        "buffers": [{"byteLength": len(buffer_blob), "uri": buffer_uri}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": position_offset, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": normal_offset, "byteLength": len(normal_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": len(index_bytes), "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": int(positions.shape[0]),
                "type": "VEC3",
                "min": positions.min(axis=0).astype(float).tolist(),
                "max": positions.max(axis=0).astype(float).tolist(),
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": int(normals.shape[0]),
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": 5125,
                "count": primitive_count,
                "type": "SCALAR",
                "min": [int(indices.min())],
                "max": [int(indices.max())],
            },
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1},
                        "indices": 2,
                        "material": 0,
                    }
                ]
            }
        ],
    }
    encoded = base64.b64encode(
        json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    return f"data:model/gltf+json;base64,{encoded}"


def _vertex_normals_from_triangles(
    points: np.ndarray, tri_idx: np.ndarray
) -> np.ndarray:
    """Accumulate face normals per vertex; ``tri_idx`` is (F, 3) vertex indices."""
    pts = np.asarray(points, dtype=np.float64)
    idx = np.asarray(tri_idx, dtype=np.int64)
    n = pts.shape[0]
    normals = np.zeros((n, 3), dtype=np.float64)
    for i, j, k in idx:
        p0, p1, p2 = pts[i], pts[j], pts[k]
        fn = np.cross(p1 - p0, p2 - p0)
        ln = float(np.linalg.norm(fn))
        if ln > 1.0e-12:
            fn /= ln
        for vi in (i, j, k):
            normals[vi] += fn
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1.0e-12] = 1.0
    normals /= norms
    return normals.astype(np.float32)


def build_pyvista_polydata_gltf_data_uri(
    mesh: Any,
    color_rgba: tuple[float, float, float, float] = (0.42, 0.48, 0.55, 1.0),
    base_color_texture_path: str | Path | None = None,
) -> str:
    """Encode a PyVista surface mesh as an embedded GLTF JSON data URI.

    The VR client loads mesh entities via ``GLTFLoader`` (see
    ``VR/client/entities/meshes/mesh_entity_manager.js``), same as
    :func:`build_cylinder_gltf_data_uri`. Use the **same** PyVista transforms
    (scale, translate, etc.) as the simulation mesh so the render matches physics.
    """
    import pyvista as pv

    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    tri = mesh.triangulate()
    try:
        tri = tri.compute_normals(point_normals=True, cell_normals=False, inplace=False)
    except RuntimeError:
        pass
    points = np.asarray(tri.points, dtype=np.float32)
    if points.size == 0:
        raise ValueError("mesh has no points")

    faces_flat = np.asarray(tri.faces, dtype=np.int64)
    if faces_flat.size == 0 or faces_flat.size % 4 != 0:
        raise ValueError("mesh faces must be a triangle list [3,i,j,k,...]")
    faces_reshaped = faces_flat.reshape(-1, 4)
    if not np.all(faces_reshaped[:, 0] == 3):
        raise ValueError("only triangular faces are supported")

    if "Normals" in tri.point_data:
        normals = np.asarray(tri.point_data["Normals"], dtype=np.float32)
    else:
        normals = _vertex_normals_from_triangles(points, faces_reshaped[:, 1:4])

    if normals.shape != points.shape:
        raise ValueError("vertex normals must match points")

    indices = faces_reshaped[:, 1:4].astype(np.uint32, copy=False).reshape(-1)

    position_bytes = points.astype("<f4", copy=False).tobytes()
    normal_bytes = normals.astype("<f4", copy=False).tobytes()
    texcoord_bytes: bytes | None = None
    if base_color_texture_path is not None:
        texcoords = getattr(tri, "active_texture_coordinates", None)
        if texcoords is not None:
            texcoords_array = np.asarray(texcoords, dtype=np.float32)
            if texcoords_array.shape == (points.shape[0], 2):
                texcoords_array = texcoords_array.copy()
                # GLTF UV origin is upper-left for most textures in Three.js workflows.
                texcoords_array[:, 1] = 1.0 - texcoords_array[:, 1]
                texcoord_bytes = texcoords_array.astype("<f4", copy=False).tobytes()
    index_bytes = indices.tobytes()
    buffer_blob = position_bytes + normal_bytes
    if texcoord_bytes is not None:
        buffer_blob += texcoord_bytes
    buffer_blob += index_bytes
    buffer_uri = (
        "data:application/octet-stream;base64,"
        + base64.b64encode(buffer_blob).decode("ascii")
    )

    position_offset = 0
    normal_offset = len(position_bytes)
    texcoord_offset = normal_offset + len(normal_bytes)
    index_offset = texcoord_offset + (len(texcoord_bytes) if texcoord_bytes is not None else 0)
    primitive_count = int(indices.size)

    alpha = float(color_rgba[3])
    material = {
        "pbrMetallicRoughness": {
            "baseColorFactor": list(color_rgba),
            "metallicFactor": 0.0,
            "roughnessFactor": 1.0,
        },
        "doubleSided": True,
    }
    if alpha < 0.999:
        # GLTF alpha in baseColorFactor only applies when alphaMode is BLEND/MASK.
        material["alphaMode"] = "BLEND"

    image_entry = None
    texture_entry = None
    texture_path = Path(base_color_texture_path) if base_color_texture_path is not None else None
    if texture_path is not None and texture_path.exists() and texcoord_bytes is not None:
        texture_mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
        }.get(texture_path.suffix.lower())
        if texture_mime is not None:
            texture_encoded = base64.b64encode(texture_path.read_bytes()).decode("ascii")
            image_entry = {"uri": f"data:{texture_mime};base64,{texture_encoded}"}
            texture_entry = {"source": 0}
            material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}

    buffer_views = [
        {"buffer": 0, "byteOffset": position_offset, "byteLength": len(position_bytes), "target": 34962},
        {"buffer": 0, "byteOffset": normal_offset, "byteLength": len(normal_bytes), "target": 34962},
    ]
    accessors = [
        {
            "bufferView": 0,
            "componentType": 5126,
            "count": int(points.shape[0]),
            "type": "VEC3",
            "min": points.min(axis=0).astype(float).tolist(),
            "max": points.max(axis=0).astype(float).tolist(),
        },
        {
            "bufferView": 1,
            "componentType": 5126,
            "count": int(normals.shape[0]),
            "type": "VEC3",
        },
    ]
    attributes = {"POSITION": 0, "NORMAL": 1}
    if texcoord_bytes is not None:
        buffer_views.append(
            {"buffer": 0, "byteOffset": texcoord_offset, "byteLength": len(texcoord_bytes), "target": 34962}
        )
        accessors.append(
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": int(points.shape[0]),
                "type": "VEC2",
            }
        )
        attributes["TEXCOORD_0"] = 2
        index_accessor_buffer_view = 3
    else:
        index_accessor_buffer_view = 2
    buffer_views.append(
        {"buffer": 0, "byteOffset": index_offset, "byteLength": len(index_bytes), "target": 34963}
    )
    accessors.append(
        {
            "bufferView": index_accessor_buffer_view,
            "componentType": 5125,
            "count": primitive_count,
            "type": "SCALAR",
            "min": [int(indices.min())],
            "max": [int(indices.max())],
        }
    )

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "materials": [material],
        "buffers": [{"byteLength": len(buffer_blob), "uri": buffer_uri}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": len(accessors) - 1,
                        "material": 0,
                    }
                ]
            }
        ],
    }
    if image_entry is not None and texture_entry is not None:
        gltf["images"] = [image_entry]
        gltf["textures"] = [texture_entry]
    encoded = base64.b64encode(
        json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    return f"data:model/gltf+json;base64,{encoded}"


def build_sphere_gltf_data_uri(
    radius: float,
    color_rgba: tuple[float, float, float, float] = (0.95, 0.82, 0.25, 1.0),
    lat_segments: int = 12,
    lon_segments: int = 18,
) -> str:
    positions, normals, indices = _build_sphere_geometry(
        float(radius),
        lat_segments=lat_segments,
        lon_segments=lon_segments,
    )

    position_bytes = positions.astype("<f4", copy=False).tobytes()
    normal_bytes = normals.astype("<f4", copy=False).tobytes()
    index_bytes = indices.astype("<u4", copy=False).tobytes()
    buffer_blob = position_bytes + normal_bytes + index_bytes
    buffer_uri = (
        "data:application/octet-stream;base64,"
        + base64.b64encode(buffer_blob).decode("ascii")
    )

    position_offset = 0
    normal_offset = len(position_bytes)
    index_offset = normal_offset + len(normal_bytes)
    primitive_count = int(indices.size)

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorFactor": list(color_rgba),
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
                "doubleSided": True,
            }
        ],
        "buffers": [{"byteLength": len(buffer_blob), "uri": buffer_uri}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": position_offset, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": normal_offset, "byteLength": len(normal_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": len(index_bytes), "target": 34963},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": int(positions.shape[0]),
                "type": "VEC3",
                "min": positions.min(axis=0).astype(float).tolist(),
                "max": positions.max(axis=0).astype(float).tolist(),
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": int(normals.shape[0]),
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": 5125,
                "count": primitive_count,
                "type": "SCALAR",
                "min": [int(indices.min())],
                "max": [int(indices.max())],
            },
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1},
                        "indices": 2,
                        "material": 0,
                    }
                ]
            }
        ],
    }
    encoded = base64.b64encode(
        json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    return f"data:model/gltf+json;base64,{encoded}"


def _build_cylinder_geometry(
    start: np.ndarray,
    direction: np.ndarray,
    normal: np.ndarray,
    length: float,
    radius: float,
    radial_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if radial_segments < 3:
        raise ValueError("radial_segments must be >= 3")

    axis = direction / np.linalg.norm(direction)
    normal_dir = normal - axis * np.dot(normal, axis)
    normal_norm = np.linalg.norm(normal_dir)
    if normal_norm < 1.0e-6:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(fallback, axis))) > 0.95:
            fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        normal_dir = fallback - axis * np.dot(fallback, axis)
        normal_norm = np.linalg.norm(normal_dir)
    normal_dir /= normal_norm
    binormal = np.cross(axis, normal_dir)
    end = start + axis * length

    positions: list[list[float]] = []
    normals: list[list[float]] = []
    indices: list[int] = []

    angles = np.linspace(0.0, 2.0 * np.pi, radial_segments, endpoint=False)

    for angle in angles:
        radial = np.cos(angle) * normal_dir + np.sin(angle) * binormal
        bottom = start + radius * radial
        top = end + radius * radial
        positions.append(bottom.astype(float).tolist())
        positions.append(top.astype(float).tolist())
        normals.append(radial.astype(float).tolist())
        normals.append(radial.astype(float).tolist())

    for idx in range(radial_segments):
        next_idx = (idx + 1) % radial_segments
        bottom_a = 2 * idx
        top_a = bottom_a + 1
        bottom_b = 2 * next_idx
        top_b = bottom_b + 1
        indices.extend([bottom_a, top_a, top_b, bottom_a, top_b, bottom_b])

    bottom_center_index = len(positions)
    positions.append(start.astype(float).tolist())
    normals.append((-axis).astype(float).tolist())
    top_center_index = len(positions)
    positions.append(end.astype(float).tolist())
    normals.append(axis.astype(float).tolist())

    bottom_ring_offset = len(positions)
    for angle in angles:
        radial = np.cos(angle) * normal_dir + np.sin(angle) * binormal
        positions.append((start + radius * radial).astype(float).tolist())
        normals.append((-axis).astype(float).tolist())

    top_ring_offset = len(positions)
    for angle in angles:
        radial = np.cos(angle) * normal_dir + np.sin(angle) * binormal
        positions.append((end + radius * radial).astype(float).tolist())
        normals.append(axis.astype(float).tolist())

    for idx in range(radial_segments):
        next_idx = (idx + 1) % radial_segments
        indices.extend(
            [
                bottom_center_index,
                bottom_ring_offset + next_idx,
                bottom_ring_offset + idx,
            ]
        )
        indices.extend(
            [
                top_center_index,
                top_ring_offset + idx,
                top_ring_offset + next_idx,
            ]
        )

    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
    )


def _build_sphere_geometry(
    radius: float,
    *,
    lat_segments: int,
    lon_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lat_segments < 2:
        raise ValueError("lat_segments must be >= 2")
    if lon_segments < 3:
        raise ValueError("lon_segments must be >= 3")

    positions: list[list[float]] = []
    normals: list[list[float]] = []
    indices: list[int] = []

    for lat_idx in range(lat_segments + 1):
        theta = np.pi * lat_idx / lat_segments
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        for lon_idx in range(lon_segments):
            phi = 2.0 * np.pi * lon_idx / lon_segments
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            normal = np.array(
                [sin_theta * cos_phi, cos_theta, sin_theta * sin_phi],
                dtype=np.float32,
            )
            positions.append((radius * normal).astype(float).tolist())
            normals.append(normal.astype(float).tolist())

    ring = lon_segments
    for lat_idx in range(lat_segments):
        for lon_idx in range(lon_segments):
            next_lon = (lon_idx + 1) % lon_segments
            a = lat_idx * ring + lon_idx
            b = lat_idx * ring + next_lon
            c = (lat_idx + 1) * ring + lon_idx
            d = (lat_idx + 1) * ring + next_lon
            if lat_idx > 0:
                indices.extend([a, c, b])
            if lat_idx < lat_segments - 1:
                indices.extend([b, c, d])

    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
    )
