import * as THREE from "three";
import { VRButton } from "three/addons/webxr/VRButton.js";

import {
  createArms,
  updateArmPosture,
  applyArmSceneState,
} from "../modes/arms.js";

import { CHARACTER_MODES_REGISTRY } from "../modes/modes_registry.js";

import { createWindowConfig, createRunConfig } from "./config.js";
import { createAppState } from "./state.js";

export function createApp({ document, window }) {
  const configWindow = createWindowConfig({ window });
  const configRun = createRunConfig({ windowConfig: configWindow });
  const state = createAppState();

  const dom = {
    statusEl: document.getElementById("status"),
    startButton: document.getElementById("start-btn"),
    joinPanel: document.getElementById("join-panel"),
    joinButton: document.getElementById("join-btn"),
    characterModeSelect: document.getElementById("character-mode"),
  };

  /* XR Scene Configuration */
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1f2b);
  const worldRoot = new THREE.Group();
  worldRoot.position.copy(configRun.calibrationOffset);
  scene.add(worldRoot);

  const camera = new THREE.PerspectiveCamera(
    70,
    window.innerWidth / window.innerHeight,
    0.01,
    50
  );
  camera.position.set(0, 1.4, 2.0);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.xr.enabled = true;
  document.body.appendChild(renderer.domElement);

  const hemi = new THREE.HemisphereLight(0xffffff, 0x223344, 1.2);
  scene.add(hemi);

  const grid = new THREE.GridHelper(8, 16, 0x6c757d, 0x495057);
  grid.position.y = 0;
  worldRoot.add(grid);
  const worldAxes = new THREE.AxesHelper(0.3);
  worldRoot.add(worldAxes);
  /* XR Configuration */

  const demoArms = createArms({
    armIds: configRun.armIds,
    armConfig: configRun.armConfig,
    worldRoot,
    curveSamples: configRun.curveSamples,
    tipDefaultRadius: configRun.tipDefaultRadius,
    maxContactPoints: configRun.maxContactPoints,
  });

  function setStatus(text) {
    dom.statusEl.textContent = text;
  }

  function updateDesktopCamera(dt) {
    // Ignore if XR is presenting
    if (renderer.xr.isPresenting) return;

    // Keyboard control
    const direction = new THREE.Vector3();
    if (state.desktopControls.keys.w) direction.z += 1;
    if (state.desktopControls.keys.s) direction.z -= 1;
    if (state.desktopControls.keys.a) direction.x -= 1;
    if (state.desktopControls.keys.d) direction.x += 1;
    if (state.desktopControls.keys.e) direction.y += 1;
    if (state.desktopControls.keys.q) direction.y -= 1;

    if (direction.lengthSq() > 0) {

      direction.normalize();
      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
      if (forward.lengthSq() > 0) {
        forward.normalize();
      }
      const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion).normalize();
      const right = new THREE.Vector3().crossVectors(forward, up).normalize();
      const delta = new THREE.Vector3()
        .addScaledVector(forward, direction.z * state.desktopControls.moveSpeed * dt)
        .addScaledVector(right, direction.x * state.desktopControls.moveSpeed * dt)
        .addScaledVector(up, direction.y * state.desktopControls.moveSpeed * dt);
      camera.position.add(delta);
    }
  }

  function clearRenderedArms() {
    for (const demoArm of demoArms.values()) {
      demoArm.group.visible = false;
    }
  }

  function applyCharacterMode(modeKey) {
    const mode = configRun.characterModes[modeKey] ?? configRun.characterModes["demo-spline"];
    state.selectedCharacterMode =
      modeKey in configRun.characterModes ? modeKey : "demo-spline";
    demoArms.get("left_arm").base.copy(mode.leftBase);
    demoArms.get("right_arm").base.copy(mode.rightBase);
    clearRenderedArms();
  }

  function connect(joinConfig) {
    // Connect to server via WebSocket
    state.socket = new WebSocket(configWindow.serverHost);
    state.sessionRole = joinConfig.serverRole ?? joinConfig.role;
    state.sessionMode = joinConfig.role;

    // Prepare connection
    clearRenderedArms();

    /* Connection event handlers */
    state.socket.onopen = () => {
      setStatus(`connected: ${configWindow.serverHost}`);
      state.socket.send(
        JSON.stringify({
          version: 1,
          type: "hello",
          payload: {
            client: "sparc-webxr",
            role: joinConfig.serverRole ?? joinConfig.role,
            requested_arm_count: joinConfig.requestedArmCount,
            character_mode: joinConfig.characterMode,
          },
        })
      );
    };

    state.socket.onclose = (event) => {
      setStatus(`disconnected: ${configWindow.serverHost} (code=${event.code})`);
      clearRenderedArms();
    };

    state.socket.onerror = () => {
      setStatus(`error: failed to connect ${configWindow.serverHost}`);
      clearRenderedArms();
    };

    state.socket.onmessage = (event) => {
      const message = JSON.parse(event.data);

      // Hello acknowledgement
      // User ID, session role, character mode, controlled arms, arm IDs
      if (message.type === "hello_ack") {
        state.userId = message.payload.user_id ?? null;
        state.sessionRole = message.payload.role ?? state.sessionRole;
        const resolvedMode = message.payload.character_mode;
        if (typeof resolvedMode === "string" && resolvedMode in configRun.characterModes) {
          state.selectedCharacterMode = resolvedMode;
          dom.characterModeSelect.value = resolvedMode;
          applyCharacterMode(resolvedMode);
        }

        // Controlled arms
        const controlled = message.payload.controlled_arm_ids ?? [];
        if (state.sessionMode === "vr_client" && controlled.length >= 2) {
          state.controlledArmByHand.left = controlled[0];
          state.controlledArmByHand.right = controlled[1];
        }

        // Current user arm IDs
        state.currentUserRenderArmIds = message.payload.arm_ids ?? [];
        state.currentUserArmIds = (state.currentUserRenderArmIds || []).filter(
          (armId) => typeof armId !== "string" || !armId.toLowerCase().includes("head")
        );

        // Status message
        setStatus(
          `connected: ${configWindow.serverHost} | role=${state.sessionRole} | user=${state.userId ?? "unknown"} | arms=${(
            message.payload.arm_ids ?? []
          ).length}`
        );
      }

      if (message.type === "asset_manifest") {
        state.armManifest = message.payload?.arms || {};
      }

      if (message.type === "error") {
        setStatus(`server error: ${message.payload?.reason ?? "unknown"}`);
      }

      if (message.type === "scene_state") {
        applyArmSceneState({
          demoArms,
          controlledArmByHand: state.controlledArmByHand,
          armIdByDemoKey: null,
          armStates: message.payload.arms || {},
          selectedCharacterMode: state.selectedCharacterMode,
          forceGenericRod: false,
        });
      }
    };
    /* Connection event handlers */
  }

  function sendXRInput(frame) {
    // Ignore if not in VR client mode
    if (state.sessionMode !== "vr_client") return;
    // Ignore if not connected to server
    if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return;

    // Fetch XR session and reference space
    const session = renderer.xr.getSession();
    const referenceSpace = renderer.xr.getReferenceSpace();
    if (!session || !referenceSpace) return;

    // TODO: Implement XR input sending (controller, head pose, action, etc.)
    return;
  }

  function animate() {
    // Heartbeat
    let lastTime = performance.now() / 1000;
    renderer.setAnimationLoop((_, frame) => {
      // Update time
      const now = performance.now() / 1000;
      const dt = Math.min(0.05, now - lastTime);
      lastTime = now;

      // Update desktop camera
      updateDesktopCamera(dt);

      // Send XR input
      if (frame) sendXRInput(frame);

      // Update arm posture
      updateArmPosture({
        demoArms,
        armIds: configRun.armIds,
        selectedCharacterMode: state.selectedCharacterMode,
        forceGenericRod: false,
        curveSamples: configRun.curveSamples,
        tipDefaultRadius: configRun.tipDefaultRadius,
        maxContactPoints: configRun.maxContactPoints,
      });

      // Render scene
      renderer.render(scene, camera);
    });
  }

  function populateCharacterModeSelect() {
    dom.characterModeSelect.replaceChildren();
    for (const { id, label } of CHARACTER_MODES_REGISTRY) {
      if (!(id in configRun.characterModes)) {
        console.warn(
          `modes_registry: "${id}" is listed in CHARACTER_MODES_REGISTRY but missing from characterModes`
        );
        continue;
      }
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = label;
      dom.characterModeSelect.appendChild(opt);
    }
  }

  function setupJoinPanel() {
    const modeFromQuery = configWindow.searchParams.get("mode");
    if (modeFromQuery && modeFromQuery in configRun.characterModes) {
      dom.characterModeSelect.value = modeFromQuery;
    }

    const refreshModeUI = () => {
      const mode = dom.characterModeSelect.value;
      applyCharacterMode(mode);
    };

    // Refresh mode UI when character mode select changes
    dom.characterModeSelect.addEventListener("change", refreshModeUI);
    refreshModeUI();

    dom.joinButton.addEventListener("click", () => {
      const joinConfig = {
        role: "vr_client",
        characterMode: dom.characterModeSelect.value,
        requestedArmCount:
          dom.characterModeSelect.value === "cathy-foraging" ? 8 : 2,
      };
      connect(joinConfig);
      dom.joinButton.disabled = true;
      dom.characterModeSelect.disabled = true;
      dom.joinPanel.style.display = "none";
    });
  }

  function setupEvents() {
    dom.startButton.addEventListener("click", () => {
      document.body.appendChild(VRButton.createButton(renderer));
      dom.startButton.remove();
    });

    renderer.xr.addEventListener("sessionstart", () => {
      connectionIndicator.visible = false;
    });

    window.addEventListener("resize", () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });

    window.addEventListener("keydown", (event) => {
      const key = event.key.toLowerCase();
      if (key in state.desktopControls.keys) {
        state.desktopControls.keys[key] = true;
      }
    });

    window.addEventListener("keyup", (event) => {
      const key = event.key.toLowerCase();
      if (key in state.desktopControls.keys) {
        state.desktopControls.keys[key] = false;
      }
    });

    renderer.domElement.addEventListener("click", () => {
      if (!renderer.xr.isPresenting && document.pointerLockElement !== renderer.domElement) {
        renderer.domElement.requestPointerLock();
      }
    });

    document.addEventListener("mousemove", (event) => {
      // Ignore if not in desktop mode
      if (document.pointerLockElement !== renderer.domElement) return;

      // Ignore if in XR session
      if (renderer.xr.isPresenting) return;

      // Update desktop camera controls based on mouse movement
      // ref: https://github.com/mrdoob/three.js/blob/dev/examples/webxr_xr_ballshooter.html
      state.desktopControls.yaw -= event.movementX * state.desktopControls.lookSensitivity;
      state.desktopControls.pitch -= event.movementY * state.desktopControls.lookSensitivity;
      const maxPitch = Math.PI * 0.49;
      state.desktopControls.pitch = Math.max(
        -maxPitch,
        Math.min(maxPitch, state.desktopControls.pitch)
      );
      camera.quaternion.setFromEuler(
        new THREE.Euler(
          state.desktopControls.pitch,
          state.desktopControls.yaw,
          0,
          "YXZ"
        )
      );
    });
  }

  return {
    start() {
      populateCharacterModeSelect();
      setupJoinPanel();
      setupEvents();
      clearRenderedArms();
      animate();
    },
  };
}
