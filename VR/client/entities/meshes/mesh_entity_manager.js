import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

import { applyPoseToObject3D } from "../../math/se3.js";

function setObjectTransform(object3d, meshState) {
  applyPoseToObject3D(object3d, meshState);
  object3d.scale.set(
    meshState.scale?.[0] ?? 1.0,
    meshState.scale?.[1] ?? 1.0,
    meshState.scale?.[2] ?? 1.0
  );
  object3d.visible = meshState.visible !== false;
}

export function createMeshEntityManager(worldRoot) {
  const gltfLoader = new GLTFLoader();
  const meshEntities = new Map();

  return {
    update(meshStates) {
      const activeMeshIds = new Set(Object.keys(meshStates || {}));

      for (const [meshId, entry] of meshEntities.entries()) {
        if (!activeMeshIds.has(meshId)) {
          worldRoot.remove(entry.object3d);
          meshEntities.delete(meshId);
        }
      }

      for (const [meshId, meshState] of Object.entries(meshStates || {})) {
        const existing = meshEntities.get(meshId);
        if (existing) {
          setObjectTransform(existing.object3d, meshState);
          continue;
        }

        const uri = meshState.asset_uri;
        if (!uri) {
          console.warn(
            `Mesh ${meshId}: missing asset_uri on first appearance (static mesh must include URI once)`,
          );
          continue;
        }

        gltfLoader.load(
          uri,
          (gltf) => {
            if (!activeMeshIds.has(meshId)) return;
            const object3d = gltf.scene;
            setObjectTransform(object3d, meshState);
            worldRoot.add(object3d);
            meshEntities.set(meshId, { object3d, assetUri: uri });
          },
          undefined,
          (error) => {
            console.error(`Failed to load mesh ${meshId}`, error);
          }
        );
      }
    },
  };
}
