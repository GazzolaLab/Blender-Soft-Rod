from __future__ import annotations

import base64
import json

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