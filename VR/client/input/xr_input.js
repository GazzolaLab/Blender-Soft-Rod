import * as THREE from "three";

import { identity3x3, multiply3x3, transpose3x3 } from "../math/matrix3.js";
import {
  quaternionFromRowwiseDirector,
  rowwiseDirectorFromQuaternion,
} from "../math/so3.js";
import {
  setWaypointPreviewVisible,
  updateWaypointPreview,
} from "./waypoint_preview.js";

const SIMULATED_ROD_MODES = new Set([
  "two-cr",
  "two-gcr",
  "spirobs",
  "cathy-throw",
  "cathy-foraging",
  "octo-waypoint",
  "noel-c4",
  "coomm-octopus",
]);

function isSimulatedRodMode(mode) {
  return SIMULATED_ROD_MODES.has(mode);
}

function applyBaseJoystickControl(baseControl, demoArm, joystickX, joystickY) {
  const xInput = Math.abs(joystickX) >= baseControl.deadband ? joystickX : 0.0;
  const zInput = Math.abs(joystickY) >= baseControl.deadband ? joystickY : 0.0;
  demoArm.base.x = Math.max(
    baseControl.minX,
    Math.min(baseControl.maxX, demoArm.base.x + xInput * baseControl.step)
  );
  demoArm.base.z = Math.max(
    baseControl.minZ,
    Math.min(baseControl.maxZ, demoArm.base.z - zInput * baseControl.step)
  );
}

function applyBaseHeightButtons(
  state,
  baseControl,
  handedness,
  demoArm,
  primaryPressed,
  secondaryPressed
) {
  const edgeState = state.buttonEdgeState[handedness];

  if (primaryPressed && !edgeState.primary) {
    demoArm.base.y = Math.min(baseControl.maxY, demoArm.base.y + baseControl.heightStep);
  }
  if (secondaryPressed && !edgeState.secondary) {
    demoArm.base.y = Math.max(baseControl.minY, demoArm.base.y - baseControl.heightStep);
  }

  edgeState.primary = primaryPressed;
  edgeState.secondary = secondaryPressed;
}

export function sendXRInput({
  state,
  config,
  renderer,
  camera,
  frame,
  demoArms,
  clearControllerDrivenArmState,
  getBaseArmDirectorRowsForHand,
  waypointPreviewByHand,
}) {
  if (state.sessionMode !== "vr_client") return;
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return;

  const session = renderer.xr.getSession();
  const referenceSpace = renderer.xr.getReferenceSpace();
  if (!session || !referenceSpace) return;

  const nowSeconds = performance.now() / 1000;
  if (nowSeconds - state.lastInputSentAt < 1.0 / config.inputSendHz) return;
  state.lastInputSentAt = nowSeconds;

  const input = {
    timestamp: nowSeconds,
    head_pose: {
      translation: [camera.position.x, camera.position.y, camera.position.z],
      rotation_xyzw: [
        camera.quaternion.x,
        camera.quaternion.y,
        camera.quaternion.z,
        camera.quaternion.w,
      ],
    },
    controllers: {},
    actions: { crawl: false },
  };

  const leftArm = demoArms.get("left_arm");
  const rightArm = demoArms.get("right_arm");
  const allowsBaseControl =
    config.characterModes[state.selectedCharacterMode].allowsBaseControl;
  const simulatedRodMode = isSimulatedRodMode(state.selectedCharacterMode);

  leftArm.controllerRawPose.valid = false;
  leftArm.controllerAlignedPose.valid = false;
  rightArm.controllerRawPose.valid = false;
  rightArm.controllerAlignedPose.valid = false;

  let leftPreviewUpdated = false;
  let rightPreviewUpdated = false;

  for (const source of session.inputSources) {
    if (!source.handedness) continue;

    const poseSpace =
      state.selectedCharacterMode === "octo-waypoint" && source.targetRaySpace
        ? source.targetRaySpace
        : source.gripSpace;
    if (!poseSpace) continue;

    const pose = frame.getPose(poseSpace, referenceSpace);
    if (!pose) continue;

    const t = pose.transform.position;
    const r = pose.transform.orientation;
    const gamepad = source.gamepad;
    const buttons = gamepad?.buttons ?? [];
    const axes = gamepad?.axes ?? [0, 0, 0, 0];
    const triggerClickPressed = !!buttons[0]?.pressed;
    const gripClickPressed = !!buttons[1]?.pressed;
    const primaryPressed = !!buttons[4]?.pressed;
    const secondaryPressed = !!buttons[5]?.pressed;
    const rawControllerQuat = new THREE.Quaternion(r.x, r.y, r.z, r.w);
    const controllerDirectorRows = rowwiseDirectorFromQuaternion(rawControllerQuat);

    if (simulatedRodMode) {
      const wasSecondaryPressed = state.recalibrationEdgeState[source.handedness];
      if (secondaryPressed && !wasSecondaryPressed) {
        state.controllerAlignmentOffsetByHand[source.handedness] = multiply3x3(
          getBaseArmDirectorRowsForHand(source.handedness),
          transpose3x3(controllerDirectorRows)
        );
      }
      state.recalibrationEdgeState[source.handedness] = secondaryPressed;
    } else {
      state.recalibrationEdgeState[source.handedness] = secondaryPressed;
      state.controllerAlignmentOffsetByHand[source.handedness] = identity3x3();
    }

    const demoArm = source.handedness === "left" ? leftArm : rightArm;

    const alignedControllerQuat = quaternionFromRowwiseDirector(
      multiply3x3(
        state.controllerAlignmentOffsetByHand[source.handedness],
        controllerDirectorRows
      )
    );

    input.controllers[source.handedness] = {
      pose: {
        translation: [t.x, t.y, t.z],
        rotation_xyzw: [
          rawControllerQuat.x,
          rawControllerQuat.y,
          rawControllerQuat.z,
          rawControllerQuat.w,
        ],
      },
      velocity: {
        linear: [0, 0, 0],
        angular: [0, 0, 0],
      },
      grip: buttons[1]?.value ?? 0,
      trigger: buttons[0]?.value ?? 0,
      joystick: [axes[2] ?? 0, axes[3] ?? 0],
      buttons: {
        trigger_click: triggerClickPressed,
        grip_click: gripClickPressed,
        primary: primaryPressed,
        secondary: secondaryPressed,
      },
    };
    input.actions.crawl = input.actions.crawl || triggerClickPressed;

    demoArm.controllerRawPose.position.set(t.x, t.y, t.z);
    demoArm.controllerRawPose.quaternion.copy(rawControllerQuat);
    demoArm.controllerRawPose.valid = true;
    demoArm.controllerAlignedPose.position.set(t.x, t.y, t.z);
    demoArm.controllerAlignedPose.quaternion.copy(alignedControllerQuat);
    demoArm.controllerAlignedPose.valid = true;

    const previewUpdated = updateWaypointPreview({
      waypointPreviewByHand,
      handedness: source.handedness,
      selectedCharacterMode: state.selectedCharacterMode,
      controllerPosition: new THREE.Vector3(t.x, t.y, t.z),
      controllerQuaternion: rawControllerQuat,
      initialControllerForward: config.initialControllerForward,
      waypointPlaneConfig: config.waypointPlaneConfig,
    });
    if (source.handedness === "left") leftPreviewUpdated = previewUpdated;
    if (source.handedness === "right") rightPreviewUpdated = previewUpdated;

    if (allowsBaseControl) {
      applyBaseJoystickControl(config.baseControl, demoArm, axes[2] ?? 0.0, axes[3] ?? 0.0);
      applyBaseHeightButtons(
        state,
        config.baseControl,
        source.handedness,
        demoArm,
        primaryPressed,
        secondaryPressed
      );
      demoArm.tip.set(t.x, t.y, t.z);
      demoArm.tipQuat.set(r.x, r.y, r.z, r.w);
      clearControllerDrivenArmState(demoArm);
    }
  }

  if (!leftPreviewUpdated) {
    setWaypointPreviewVisible(waypointPreviewByHand, "left", false);
  }
  if (!rightPreviewUpdated) {
    setWaypointPreviewVisible(waypointPreviewByHand, "right", false);
  }

  state.socket.send(JSON.stringify({ version: 1, type: "xr_input", payload: input }));
}
