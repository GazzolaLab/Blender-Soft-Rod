export function setObjectTranslation(object3d, translation) {
  object3d.position.set(
    translation?.[0] ?? 0.0,
    translation?.[1] ?? 0.0,
    translation?.[2] ?? 0.0
  );
}

export function setObjectRotationXyzw(object3d, rotationXyzw) {
  object3d.quaternion.set(
    rotationXyzw?.[0] ?? 0.0,
    rotationXyzw?.[1] ?? 0.0,
    rotationXyzw?.[2] ?? 0.0,
    rotationXyzw?.[3] ?? 1.0
  );
}

export function applyPoseToObject3D(object3d, poseState) {
  setObjectTranslation(object3d, poseState?.translation);
  setObjectRotationXyzw(object3d, poseState?.rotation_xyzw);
}

