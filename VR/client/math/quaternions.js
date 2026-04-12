import * as THREE from "three";

import { transpose3x3 } from "./matrix3.js";

export function quaternionFromRowwiseDirector(matrixRows) {
  const m = new THREE.Matrix4();
  m.set(
    matrixRows[0][0], matrixRows[1][0], matrixRows[2][0], 0.0,
    matrixRows[0][1], matrixRows[1][1], matrixRows[2][1], 0.0,
    matrixRows[0][2], matrixRows[1][2], matrixRows[2][2], 0.0,
    0.0, 0.0, 0.0, 1.0
  );
  return new THREE.Quaternion().setFromRotationMatrix(m);
}

export function columnwiseDirectorFromQuaternion(quaternion) {
  // Convert quaternion to column-wise director for elastica representation.
  // Maybe consider using three.js's conversion, but we just want 3x3 orientation.
  const x = quaternion.x;
  const y = quaternion.y;
  const z = quaternion.z;
  const w = quaternion.w;
  const xx = x * x;
  const yy = y * y;
  const zz = z * z;
  const xy = x * y;
  const xz = x * z;
  const yz = y * z;
  const wx = w * x;
  const wy = w * y;
  const wz = w * z;
  return [
    [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
    [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
    [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
  ];
}

export function rowwiseDirectorFromQuaternion(quaternion) {
  // Typically quaternion from controller is in world space,
  // We need to convert to row-wise director for elastica representation.
  return transpose3x3(columnwiseDirectorFromQuaternion(quaternion));
}
