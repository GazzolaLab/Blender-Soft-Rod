import * as THREE from "three";

const SUPPORTED_REPRESENTATIONS = new Set([
  "segmented-pipe",
  "torus-stack",
]);


export function renderArmGeometry(demoArm, options = {}) {
  const representation = options.builder ?? options.representation;
  clearArmBodyGroup(demoArm.armBodyGroup);

  switch (representation) {
    case "segmented-pipe":
      buildSegmentedPipe(demoArm);
      break;
    case "torus-stack":
      buildArmWithTorusStack(demoArm);
      break;
    default:
      buildSegmentedPipe(demoArm);
  }
}

function clearArmBodyGroup(group) {
  while (group.children.length > 0) {
    const child = group.children[group.children.length - 1];
    group.remove(child);
    disposeObject3D(child);
  }
}

function disposeObject3D(object3d) {
  object3d.traverse((child) => {
    if (child.geometry) {
      child.geometry.dispose();
    }
    if (child.material) {
      if (Array.isArray(child.material)) {
        for (const material of child.material) material.dispose();
      } else {
        child.material.dispose();
      }
    }
  });
}

function buildSegmentedPipe(demoArm) {
  const points = demoArm.centerline;
  const radii = demoArm.radii;
  if (!Array.isArray(points) || points.length < 2) return;
  if (!Array.isArray(radii) || radii.length < 1) return;

  const material = new THREE.MeshStandardMaterial({
    color: demoArm.color,
    roughness: 0.45,
    metalness: 0.05,
  });
  const yAxis = new THREE.Vector3(0, 1, 0);

  for (let i = 0; i < points.length - 1; i += 1) {
    const p0 = points[i];
    const p1 = points[i + 1];
    const segment = p1.clone().sub(p0);
    const length = segment.length();
    if (length <= 1e-6) continue;

    const r0 = Math.max(0.001, radii[Math.min(i, radii.length - 1)]);
    const r1 = Math.max(0.001, radii[Math.min(i + 1, radii.length - 1)]);
    const geom = new THREE.CylinderGeometry(r1, r0, length, 14, 1, true);
    const mesh = new THREE.Mesh(geom, material.clone());
    mesh.position.copy(p0.clone().add(p1).multiplyScalar(0.5));
    mesh.quaternion.setFromUnitVectors(yAxis, segment.normalize());
    demoArm.armBodyGroup.add(mesh);
  }

  for (let i = 0; i < points.length; i += 1) {
    let radius;
    if (i === 0) {
      radius = radii[0];
    } else if (i === points.length - 1) {
      radius = radii[radii.length - 1];
    } else {
      const left = radii[Math.max(0, i - 1)];
      const right = radii[Math.min(radii.length - 1, i)];
      radius = 0.5 * (left + right);
    }
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(Math.max(0.001, radius), 14, 10),
      material.clone()
    );
    sphere.position.copy(points[i]);
    demoArm.armBodyGroup.add(sphere);
  }
}

function buildArmWithTorusStack(demoArm) {
  const points = demoArm.centerline;
  const radii = demoArm.radii;
  const elementLengths = demoArm.elementLengths;
  const directors = demoArm.directors;
  if (!Array.isArray(points) || points.length < 2) return;
  if (!Array.isArray(radii) || radii.length < 1) return;

  const zAxis = new THREE.Vector3(0, 0, 1);

  for (let i = 0; i < points.length - 1; i += 1) {
    const p0 = points[i];
    const p1 = points[i + 1];
    const segment = p1.clone().sub(p0);
    const segmentLength = segment.length();
    if (segmentLength <= 1e-6) continue;

    const majorRadius = Math.max(0.001, radii[Math.min(i, radii.length - 1)]);
    const rawElementLength =
      Array.isArray(elementLengths) && i < elementLengths.length
        ? elementLengths[i]
        : segmentLength;
    const minorRadius = Math.max(
      0.001,
      Math.min(majorRadius * 0.9, 0.2 * Math.max(0.0, rawElementLength))
    );

    const torus = new THREE.Mesh(
      new THREE.TorusGeometry(majorRadius, minorRadius, 10, 20),
      new THREE.MeshStandardMaterial({
        color: demoArm.color,
        roughness: 0.42,
        metalness: 0.06,
      })
    );
    torus.position.copy(p0.clone().add(p1).multiplyScalar(0.5));

    const hasDirector =
      Array.isArray(directors) &&
      directors.length === points.length - 1 &&
      Array.isArray(directors[i]) &&
      directors[i].length === 3;
    if (hasDirector) {
      const m = new THREE.Matrix4();
      const rows = directors[i];
      // directors are row-wise [normal, binormal, tangent] in world coords.
      // For object rotation, columns must be basis vectors in world => D^T.
      m.set(
        rows[0][0], rows[1][0], rows[2][0], 0.0,
        rows[0][1], rows[1][1], rows[2][1], 0.0,
        rows[0][2], rows[1][2], rows[2][2], 0.0,
        0.0, 0.0, 0.0, 1.0
      );
      torus.quaternion.setFromRotationMatrix(m);
    } else {
      torus.quaternion.setFromUnitVectors(zAxis, segment.normalize());
    }

    demoArm.armBodyGroup.add(torus);
  }
}
