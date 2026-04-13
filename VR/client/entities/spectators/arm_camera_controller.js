import * as THREE from "three";

const DEFAULT_FORWARD_OFFSET = 0.025;
const DEFAULT_POSITION_LERP = 0.2;
const DEFAULT_ROTATION_SLERP = 0.18;
const WORLD_UP = new THREE.Vector3(0.0, 1.0, 0.0);
const ALT_UP = new THREE.Vector3(1.0, 0.0, 0.0);

function setOptions(select, items, selectedValue, placeholder) {
  const previousValue = selectedValue ?? "";
  const signature = items.map((item) => `${item.value}\u0000${item.label}`).join("\u0001");
  const placeholderSignature = `placeholder\u0000${placeholder}`;
  const nextSignature = items.length === 0 ? placeholderSignature : signature;

  if (select.dataset.optionSignature !== nextSignature) {
    select.innerHTML = "";

    if (items.length === 0) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = placeholder;
      select.append(option);
      select.dataset.optionSignature = nextSignature;
      select.value = "";
      select.disabled = true;
      return "";
    }

    for (const item of items) {
      const option = document.createElement("option");
      option.value = item.value;
      option.textContent = item.label;
      select.append(option);
    }
    select.dataset.optionSignature = nextSignature;
  }

  if (items.length === 0) {
    select.value = "";
    select.disabled = true;
    return "";
  }

  select.disabled = false;
  const nextValue = items.some((item) => item.value === previousValue)
    ? previousValue
    : items[0].value;
  select.value = nextValue;
  return nextValue;
}

function chooseUpHint(forward, preferredUp) {
  if (preferredUp.lengthSq() > 1.0e-8) {
    const normalized = preferredUp.clone().normalize();
    if (Math.abs(normalized.dot(forward)) < 0.98) {
      return normalized;
    }
  }

  if (Math.abs(WORLD_UP.dot(forward)) < 0.98) {
    return WORLD_UP;
  }
  return ALT_UP;
}

function derivePoseFromArmState(armState, scratch) {
  const {
    position,
    quaternion,
    matrix,
    xAxis,
    yAxis,
    zAxis,
    forward,
    upHint,
    previousPoint,
  } = scratch;
  const tip = armState?.tip?.translation;
  if (!Array.isArray(tip) || tip.length !== 3) {
    return false;
  }

  let hasForward = false;
  upHint.set(0.0, 0.0, 0.0);
  const directors = armState?.directors;
  if (Array.isArray(directors) && directors.length > 0) {
    const lastDirector = directors[directors.length - 1];
    if (
      Array.isArray(lastDirector) &&
      lastDirector.length === 3 &&
      Array.isArray(lastDirector[1]) &&
      Array.isArray(lastDirector[2])
    ) {
      forward.set(lastDirector[2][0], lastDirector[2][1], lastDirector[2][2]);
      upHint.set(lastDirector[1][0], lastDirector[1][1], lastDirector[1][2]);
      hasForward = forward.lengthSq() > 1.0e-8;
    }
  }

  if (!hasForward && Array.isArray(armState?.centerline) && armState.centerline.length >= 2) {
    const { centerline } = armState;
    forward.set(
      centerline[centerline.length - 1][0],
      centerline[centerline.length - 1][1],
      centerline[centerline.length - 1][2]
    );
    previousPoint.set(
      centerline[centerline.length - 2][0],
      centerline[centerline.length - 2][1],
      centerline[centerline.length - 2][2]
    );
    forward.sub(previousPoint);
    hasForward = forward.lengthSq() > 1.0e-8;
  }

  if (!hasForward) {
    return false;
  }

  forward.normalize();
  const chosenUp = chooseUpHint(forward, upHint);
  zAxis.copy(forward).multiplyScalar(-1.0);
  xAxis.crossVectors(chosenUp, zAxis);
  if (xAxis.lengthSq() <= 1.0e-8) {
    return false;
  }
  xAxis.normalize();
  yAxis.crossVectors(zAxis, xAxis).normalize();

  matrix.makeBasis(xAxis, yAxis, zAxis);
  quaternion.setFromRotationMatrix(matrix);
  position.set(tip[0], tip[1], tip[2]).addScaledVector(forward, DEFAULT_FORWARD_OFFSET);
  return true;
}

export function createArmCameraController({
  panelEl,
  userSelectEl,
  armSelectEl,
  positionLerp = DEFAULT_POSITION_LERP,
  rotationSlerp = DEFAULT_ROTATION_SLERP,
} = {}) {
  const scratch = {
    position: new THREE.Vector3(),
    quaternion: new THREE.Quaternion(),
    matrix: new THREE.Matrix4(),
    xAxis: new THREE.Vector3(),
    yAxis: new THREE.Vector3(),
    zAxis: new THREE.Vector3(),
    forward: new THREE.Vector3(),
    upHint: new THREE.Vector3(),
    previousPoint: new THREE.Vector3(),
  };
  const smoothedPosition = new THREE.Vector3();
  const smoothedQuaternion = new THREE.Quaternion();
  let selectedUserId = "";
  let selectedArmId = "";
  let initialized = false;

  function setPanelVisible(visible) {
    if (panelEl) {
      panelEl.style.display = visible ? "grid" : "none";
    }
  }

  function setSelection(userId, armId) {
    selectedUserId = userId ?? "";
    selectedArmId = armId ?? "";
  }

  function clear() {
    selectedUserId = "";
    selectedArmId = "";
    initialized = false;
    smoothedPosition.set(0.0, 0.0, 0.0);
    smoothedQuaternion.identity();
    if (userSelectEl) {
      userSelectEl.innerHTML = "";
      userSelectEl.dataset.optionSignature = "";
      userSelectEl.disabled = true;
    }
    if (armSelectEl) {
      armSelectEl.innerHTML = "";
      armSelectEl.dataset.optionSignature = "";
      armSelectEl.disabled = true;
    }
  }

  function syncSelection({ userArms, armStates }) {
    const users = Object.entries(userArms || {})
      .map(([userId, armIds]) => ({
        userId,
        armIds: (armIds || []).filter((armId) => {
          const state = armStates?.[armId];
          return !!state && !String(armId).toLowerCase().includes("head");
        }),
      }))
      .filter((entry) => entry.armIds.length > 0);

    const nextUserId = userSelectEl
      ? setOptions(
          userSelectEl,
          users.map((entry) => ({
            value: entry.userId,
            label: `${entry.userId} (${entry.armIds.length} arms)`,
          })),
          selectedUserId,
          "No users available"
        )
      : users[0]?.userId ?? "";

    selectedUserId = nextUserId;
    const armItems =
      users.find((entry) => entry.userId === selectedUserId)?.armIds.map((armId) => ({
        value: armId,
        label: armId,
      })) ?? [];
    const nextArmId = armSelectEl
      ? setOptions(armSelectEl, armItems, selectedArmId, "No arms available")
      : armItems[0]?.value ?? "";
    selectedArmId = nextArmId;
    return { userId: selectedUserId, armId: selectedArmId };
  }

  if (userSelectEl) {
    userSelectEl.addEventListener("change", () => {
      selectedUserId = userSelectEl.value;
      selectedArmId = "";
    });
  }
  if (armSelectEl) {
    armSelectEl.addEventListener("change", () => {
      selectedArmId = armSelectEl.value;
    });
  }

  return {
    show() {
      setPanelVisible(true);
    },

    hide() {
      setPanelVisible(false);
    },

    clear,

    getSelection() {
      return { userId: selectedUserId, armId: selectedArmId };
    },

    syncSelection,

    applyToCamera(camera, armState) {
      if (!derivePoseFromArmState(armState, scratch)) {
        initialized = false;
        return false;
      }

      if (!initialized) {
        smoothedPosition.copy(scratch.position);
        smoothedQuaternion.copy(scratch.quaternion);
        initialized = true;
      } else {
        smoothedPosition.lerp(scratch.position, positionLerp);
        smoothedQuaternion.slerp(scratch.quaternion, rotationSlerp);
      }

      camera.position.copy(smoothedPosition);
      camera.quaternion.copy(smoothedQuaternion);
      return true;
    },

    setSelection,
  };
}

