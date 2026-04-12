import * as THREE from "three";
import { VRButton } from "three/addons/webxr/VRButton.js";
import { CHARACTER_MODES_REGISTRY } from "./modes/modes_registry.js";

const statusEl = document.getElementById("status");
const startButton = document.getElementById("start-btn");
const joinPanel = document.getElementById("join-panel");
const joinButton = document.getElementById("join-btn");
const characterModeSelect = document.getElementById("character-mode");
let selectedCharacterMode = "demo-spline";

const searchParams = new URLSearchParams(window.location.search);
const defaultWsScheme = window.location.protocol === "https:" ? "wss" : "ws";
const defaultWsHost = window.location.hostname || "127.0.0.1";
const defaultWsPort = searchParams.get("ws_port") ?? "8765";
const serverHost =
  searchParams.get("ws") ??
  `${defaultWsScheme}://${defaultWsHost}:${defaultWsPort}`;

const calibrationOffset = new THREE.Vector3(
  Number(searchParams.get("ox") ?? 0.0),
  Number(searchParams.get("oy") ?? 0.0),
  Number(searchParams.get("oz") ?? 0.0)
);

let socket = null;
let userId = null;
let sessionRole = "vr_client";
let sessionMode = "vr_client";

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1f2b);
const worldRoot = new THREE.Group();
worldRoot.position.copy(calibrationOffset);
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

function setStatus(text) {
  statusEl.textContent = text;
}

function applyCharacterMode(modeKey) {
  // TODO: Implement character mode selection
}

function connect(joinConfig) {
  socket = new WebSocket(serverHost);
  sessionRole = joinConfig.serverRole ?? joinConfig.role;
  sessionMode = joinConfig.role;

  socket.onopen = () => {
    setStatus(`connected: ${serverHost}`);
    socket.send(
      JSON.stringify({
        version: 1,
        type: "hello",
        payload: {
          client: "sparc-webxr",
          role: sessionRole,
        },
      })
    );
  };

  socket.onclose = (event) => {
    setStatus(`disconnected: ${serverHost} (code=${event.code})`);
  };
  socket.onerror = () => {
    setStatus(`error: failed to connect ${serverHost}`);
  };

  socket.onmessage = (event) => {
    // TODO: Implement message handling

    const message = JSON.parse(event.data);

    if (message.type === "error") {
      setStatus(`server error: ${message.payload?.reason ?? "unknown"}`);
    }

  };
}

function sendXRInput(frame) {
  // TODO: Implement XR input sending
}

function animate() {
  // TODO: Implement animation loop
  let lastTime = performance.now() / 1000;
  renderer.setAnimationLoop((_, frame) => {
    const now = performance.now() / 1000;
    const dt = Math.min(0.05, now - lastTime);
    lastTime = now;

    if (frame) sendXRInput(frame);

    renderer.render(scene, camera);
  });
}

startButton.addEventListener("click", () => {
  document.body.appendChild(VRButton.createButton(renderer));
  startButton.remove();
});

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

/* TODO: Implement key handling for desktop mode */
window.addEventListener("keydown", (event) => {
  const key = event.key.toLowerCase();
  // TODO: Implement key handling
});

window.addEventListener("keyup", (event) => {
  const key = event.key.toLowerCase();
  // TODO: Implement key handling
});
/* TODO */

renderer.domElement.addEventListener("click", () => {
  if (!renderer.xr.isPresenting && document.pointerLockElement !== renderer.domElement) {
    renderer.domElement.requestPointerLock();
  }
});

document.addEventListener("mousemove", (event) => {
  if (document.pointerLockElement !== renderer.domElement) return;
  if (renderer.xr.isPresenting) return;

  // TODO: Implement desktop mode mouse movement handling
});

function setupJoinPanel() {
  // TODO: Implement join panel setup
}

function populateCharacterModeSelect() {
  characterModeSelect.replaceChildren();
  for (const { id, label } of CHARACTER_MODES_REGISTRY) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = label;
    characterModeSelect.appendChild(opt);
  }
}

populateCharacterModeSelect();
setupJoinPanel();
animate();
